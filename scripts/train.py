#!/usr/bin/env python3
"""Train an autoregressive image generator with the MASC coarse-vocabulary prior.

This entry point implements the *MASC-specific* parts of Stage 2 training
(Figure 2): it loads a pre-built MASC mapping, relabels the tokenizer codes to
coarse cluster indices on the fly, resizes the backbone's embedding/head to the
coarse vocabulary, and optimises the ``k``-way next-cluster cross-entropy (Eq. 5)
with the schedule reported in Appendix A.2 / Table 8.

What this script owns vs. what it expects from the backbone repo
----------------------------------------------------------------
MASC is a plug-and-play module.  To keep the change surface minimal and avoid
re-distributing third-party code, the *backbone* (its transformer, class
conditioning, CFG, KV-cache and AR sampling loop) is obtained from the official
repository of whichever generator you are reproducing and exposed through a thin
adapter implementing :class:`masc.integration.ARBackbone`.  Concretely you
provide a ``backbones`` package with::

    from backbones import build_backbone   # -> (model, ARBackbone adapter)

and a directory of pre-extracted tokenizer codes (the standard LlamaGen/VAR
practice of caching ``z^q`` per image).  See ``docs/reproduction.md`` for how to
produce both from the official LlamaGen / VAR / RandAR / IAR / CTF / GigaTok /
RAR releases.  Everything below this line is generic and backbone-independent.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def build_optimizer(model, cfg):
    import torch
    opt_cfg = cfg["optimizer"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["peak_lr"],
        betas=tuple(opt_cfg.get("betas", (0.9, 0.95))),
        eps=float(opt_cfg.get("eps", 1e-8)),
        weight_decay=opt_cfg.get("weight_decay", 0.05),
    )


def cosine_lr(step, total_steps, peak_lr, final_lr, warmup_steps=0):
    """Cosine-annealing schedule used by all backbones (Table 8)."""
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    return final_lr + 0.5 * (peak_lr - final_lr) * (1 + math.cos(math.pi * progress))


class CodeDataset:
    """Pre-extracted tokenizer codes ``z^q`` (+ class labels) on disk.

    Supports:
    1. Multi-sample shards (recommended): each ``shard_*.npy`` is ``[S, L]``,
       with parallel ``labels.npy`` of shape ``[total_samples]``.
       Dataset exposes sample-level indexing, enabling standard batching and shuffling.
    2. Pre-batched shards (legacy): each ``.npy`` is ``[B, L]``, with
       parallel ``labels.npy`` of shape ``[num_shards, B]``.
    """

    def __init__(self, code_dir: str):
        self.files = sorted(
            os.path.join(code_dir, f) for f in os.listdir(code_dir)
            if f.endswith(".npy") and f != "labels.npy"
        )
        if not self.files:
            raise FileNotFoundError(
                f"No code shards in {code_dir}. Extract tokenizer codes from your "
                "backbone's tokenizer first (see docs/reproduction.md)."
            )
        self.labels = np.load(os.path.join(code_dir, "labels.npy"))

        first_shard = np.load(self.files[0], mmap_mode="r")
        if first_shard.ndim == 2 and first_shard.shape[0] > 1 and len(self.labels) > len(self.files):
            self.is_sharded = True
            self.mmaps = [np.load(f, mmap_mode="r") for f in self.files]
            self.shard_lengths = [m.shape[0] for m in self.mmaps]
            self.cum_lengths = np.cumsum([0] + self.shard_lengths)
            self.total_len = int(self.cum_lengths[-1])
        else:
            self.is_sharded = False
            self.total_len = len(self.files)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if self.is_sharded:
            shard_idx = int(np.searchsorted(self.cum_lengths[1:], idx, side="right"))
            local_idx = idx - self.cum_lengths[shard_idx]
            return self.mmaps[shard_idx][local_idx].astype(np.int64), int(self.labels[idx])
        return np.load(self.files[idx]), self.labels[idx]


def main() -> None:
    ap = argparse.ArgumentParser(description="Train AR generator + MASC prior.")
    ap.add_argument("--config",   required=True, help="YAML config (see configs/).")
    ap.add_argument("--mapping",  required=True, help="MASC mapping .npz from build_masc_tree.py")
    ap.add_argument("--codes",    required=True, help="Directory of pre-extracted tokenizer codes.")
    ap.add_argument("--out",      required=True, help="Checkpoint output directory.")
    # UPipe flags (override config; 0 = disable)
    ap.add_argument("--upipe-chunk-heads", type=int, default=None,
                    help="Q-heads per UPipe stage (None = disable UPipe, use MASC only).")
    ap.add_argument("--no-flash",  action="store_true",
                    help="Disable Flash-Attention 2 within each UPipe chunk.")
    # Gradient checkpointing (saves ~30%% VRAM at ~20%% speed cost)
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="Enable gradient checkpointing on transformer layers.")
    # Resume from a previous checkpoint
    ap.add_argument("--resume", default=None,
                    help="Path to checkpoint .pt file to resume from.")
    args = ap.parse_args()

    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler

    from masc import load_mapping
    from masc.integration import resize_token_embedding, resize_output_head, MASCObjective

    cfg = load_config(args.config)

    # --- Distributed setup ---
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        is_ddp = world_size > 1
        is_main = local_rank == 0
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        local_rank = 0
        world_size = 1
        is_ddp = False
        is_main = True

    # --- MASC prior ---------------------------------------------------------
    mp = load_mapping(args.mapping)
    assert mp.k == cfg["masc"]["k"], (mp.k, cfg["masc"]["k"])
    objective = MASCObjective(mp.mapping, device=device)
    if is_main:
        print(f"[train] MASC mapping: N={mp.n} -> k={mp.k}")

    # --- Backbone (from the official repo, via your adapter) ----------------
    try:
        from backbones import build_backbone
    except ImportError as e:
        raise SystemExit(
            "Could not import `backbones.build_backbone`. MASC is plug-and-play: "
            "supply a thin adapter around the official backbone you are "
            "reproducing (LlamaGen/VAR/RandAR/IAR/CTF/GigaTok/RAR). "
            "See docs/reproduction.md."
        ) from e

    model, adapter = build_backbone(cfg["backbone"])
    model = model.to(device)

    # --- Gradient checkpointing (optional, saves ~30% VRAM) -----------------
    if args.grad_ckpt:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            if is_main:
                print("[train] Gradient checkpointing enabled (HuggingFace API).")
        else:
            if is_main:
                print("[train] gradient_checkpointing_enable() not found on model.")

    # --- MASC + UPipe surgery (in correct order) ----------------------------
    chunk_heads = args.upipe_chunk_heads
    if chunk_heads is None:
        upipe_cfg = cfg.get("upipe", {})
        chunk_heads = upipe_cfg.get("chunk_heads", None) if upipe_cfg.get("enabled", False) else None
    use_flash = not args.no_flash

    from masc.integration import apply_masc_and_upipe
    _upipe_registry = apply_masc_and_upipe(
        model=model,
        adapter=adapter,
        mapping=mp.mapping,
        k=mp.k,
        upipe_chunk_heads=chunk_heads,
        upipe_use_flash=use_flash,
    )

    # --- DDP wrapper --------------------------------------------------------
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # --- Data / optim -------------------------------------------------------
    ds = CodeDataset(args.codes)
    local_batch = cfg["train"].get("local_batch_size", 8)
    if ds.is_sharded:
        sampler = DistributedSampler(ds, shuffle=True) if is_ddp else None
        loader = DataLoader(
            ds, batch_size=local_batch,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=True,
            drop_last=True
        )
    else:
        sampler = DistributedSampler(ds, shuffle=True) if is_ddp else None
        loader = DataLoader(
            ds, batch_size=None,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=True,
            persistent_workers=True
        )

    opt = build_optimizer(model, cfg)
    train_cfg = cfg["train"]
    total_steps = train_cfg["total_steps"]
    peak_lr = cfg["optimizer"]["peak_lr"]
    final_lr = cfg["optimizer"]["final_lr"]
    grad_clip = train_cfg.get("grad_clip", 1.0)
    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.get("amp", True))

    # --- Resume from checkpoint if requested --------------------------------
    start_step = 0
    if args.resume and os.path.exists(args.resume) and "c2i_" not in os.path.basename(args.resume):
        try:
            ckpt = torch.load(args.resume, map_location=device)
            raw_model = model.module if hasattr(model, "module") else model
            raw_model.load_state_dict(ckpt["model"])
            start_step = ckpt.get("step", 0)
            if "optimizer" in ckpt:
                opt.load_state_dict(ckpt["optimizer"])
            if "scaler" in ckpt:
                scaler.load_state_dict(ckpt["scaler"])
            if is_main:
                print(f"[train] Resumed from step {start_step} ({args.resume})")
        except Exception as e:
            if is_main:
                print(f"[train] WARNING: Failed to load {args.resume}: {e}")
                all_ckpts = sorted(glob.glob(os.path.join(args.out, "ckpt_[0-9]*.pt")))
                candidates = [c for c in all_ckpts if os.path.abspath(c) != os.path.abspath(args.resume)]
                if candidates:
                    fallback = candidates[-1]
                    print(f"[train] Falling back to previous valid checkpoint: {fallback}")
                    ckpt = torch.load(fallback, map_location=device)
                    raw_model = model.module if hasattr(model, "module") else model
                    raw_model.load_state_dict(ckpt["model"])
                    start_step = ckpt.get("step", 0)
                    if "optimizer" in ckpt:
                        opt.load_state_dict(ckpt["optimizer"])
                    if "scaler" in ckpt:
                        scaler.load_state_dict(ckpt["scaler"])
                    print(f"[train] Fallback successful! Resumed from step {start_step}")
                else:
                    raise

    if is_main:
        os.makedirs(args.out, exist_ok=True)
    model.train()
    step = start_step

    while step < total_steps:
        if is_ddp and sampler is not None:
            sampler.set_epoch(step)
        for codes, label in loader:
            codes = torch.as_tensor(codes, device=device)   # [B, L] fine z^q
            label = torch.as_tensor(label, device=device)
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, total_steps, peak_lr, final_lr)
            with torch.cuda.amp.autocast(enabled=train_cfg.get("amp", True)):
                coarse_in = objective.coarse_targets(codes)
                # DDP-compatible forward call
                if hasattr(model, "forward_for_masc"):
                    logits = model.forward_for_masc(coarse_in, class_labels=label)
                else:
                    out = model(coarse_in, label)
                    logits = out[0] if isinstance(out, tuple) else out
                    if logits.shape[1] > coarse_in.shape[1]:
                        logits = logits[:, :coarse_in.shape[1], :]
                loss = objective.loss(logits, codes)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            if is_main and step % cfg.get("log_every", 50) == 0:
                print(f"[train] step {step}/{total_steps}  loss={loss.item():.4f}",
                      flush=True)
            if is_main and step > 0 and step % train_cfg.get("ckpt_every", 10000) == 0:
                raw_model = model.module if hasattr(model, "module") else model
                ckpt_dest = os.path.join(args.out, f"ckpt_{step}.pt")
                ckpt_tmp = ckpt_dest + ".tmp"
                torch.save({
                    "model":     raw_model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scaler":    scaler.state_dict(),
                    "step":      step,
                    "upipe_chunk_heads": chunk_heads,
                    "masc_k":    mp.k,
                }, ckpt_tmp)
                os.replace(ckpt_tmp, ckpt_dest)
                print(f"[train] Saved checkpoint: {ckpt_dest}")

                # Keep only the last 2 checkpoints to avoid exceeding Kaggle disk quota
                all_ckpts = sorted(glob.glob(os.path.join(args.out, "ckpt_[0-9]*.pt")))
                for old_ck in all_ckpts[:-2]:
                    try:
                        os.remove(old_ck)
                        print(f"[train] Removed old {os.path.basename(old_ck)} to free disk space")
                    except Exception:
                        pass
            step += 1
            if step >= total_steps:
                break

    if is_main:
        raw_model = model.module if hasattr(model, "module") else model
        torch.save({"model": raw_model.state_dict(), "step": step},
                   os.path.join(args.out, "ckpt_final.pt"))
        print("[train] done.")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
