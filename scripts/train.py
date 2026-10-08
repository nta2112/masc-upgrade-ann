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
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        trainable_params,
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
    # Epochs override (e.g. --epochs 15.0)
    ap.add_argument("--epochs", type=float, default=None,
                    help="Number of epochs to train (overrides config).")
    ap.add_argument("--ckpt-every", type=int, default=None,
                    help="Checkpoint interval in steps (overrides config).")
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

    # --- Freeze backbone layers (optional fast fine-tuning) -----------------
    freeze_layers = cfg.get("backbone", {}).get("freeze_layers", 0)
    if freeze_layers > 0 and hasattr(model, "layers"):
        num_layers = len(model.layers)
        freeze_count = min(freeze_layers, num_layers)
        for i in range(freeze_count):
            for p in model.layers[i].parameters():
                p.requires_grad = False
        trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_p = sum(p.numel() for p in model.parameters())
        if is_main:
            print(f"[train] Frozen first {freeze_count}/{num_layers} layers! Trainable: {trainable_p/1e6:.1f}M / {total_p/1e6:.1f}M ({trainable_p/total_p*100:.1f}%)")

    # --- DDP wrapper --------------------------------------------------------
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # --- Data / optim -------------------------------------------------------
    ds = CodeDataset(args.codes)
    local_batch = cfg["train"].get("local_batch_size", 8)
    num_workers = min(4, os.cpu_count() or 1)
    if ds.is_sharded:
        sampler = DistributedSampler(ds, shuffle=True) if is_ddp else None
        loader = DataLoader(
            ds, batch_size=local_batch,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if (num_workers > 0) else None,
        )
    else:
        sampler = DistributedSampler(ds, shuffle=True) if is_ddp else None
        loader = DataLoader(
            ds, batch_size=None,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if (num_workers > 0) else None,
        )

    opt = build_optimizer(model, cfg)
    train_cfg = cfg["train"]
    epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs")
    ckpt_every = args.ckpt_every if args.ckpt_every is not None else train_cfg.get("ckpt_every", 1000)
    global_batch = local_batch * (world_size if is_ddp else 1)
    if epochs is not None and epochs > 0:
        steps_per_epoch = max(1, len(ds) // max(1, global_batch))
        total_steps = int(epochs * steps_per_epoch)
        if is_main:
            print(f"[train] Epoch-based training: {epochs} epoch(s) | {len(ds):,} samples | global batch {global_batch} -> {total_steps:,} total steps ({steps_per_epoch:,} steps/epoch)")
    else:
        total_steps = train_cfg.get("total_steps", 40000)
        steps_per_epoch = max(1, len(ds) // max(1, global_batch))
        if is_main:
            print(f"[train] Step-based training: {total_steps:,} total steps (batch {global_batch}, ~{total_steps/steps_per_epoch:.2f} epochs)")
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

            # Nếu checkpoint đã đạt hoặc vượt total_steps gốc (ví dụ đã xong 1 epoch = 40,036 steps)
            # Tự động cộng dồn số bước cần train thêm để chạy tiếp đúng số epochs mong muốn
            if start_step >= total_steps:
                added_steps = int((epochs or 1.0) * steps_per_epoch)
                total_steps = start_step + added_steps
                if is_main:
                    print(f"[train] Checkpoint đã hoàn thành epoch trước. Tự động train THÊM {epochs or 1.0} epoch ({added_steps:,} steps) -> Mục tiêu mới: {total_steps:,} steps!")
            elif epochs is not None:
                # Nếu muốn train thêm đủ 1 epoch tính từ điểm resume
                target_from_now = start_step + int(epochs * steps_per_epoch)
                if target_from_now > total_steps:
                    total_steps = target_from_now
                    if is_main:
                        print(f"[train] Mở rộng mục tiêu train thêm {epochs} epoch từ checkpoint -> Mục tiêu mới: {total_steps:,} steps!")
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
        cur_epoch = int(step / max(1, steps_per_epoch))
        if is_ddp and sampler is not None:
            sampler.set_epoch(cur_epoch)
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
                epoch_prog = (step * global_batch) / max(1, len(ds))
                vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                print(f"[train] step {step}/{total_steps} | epoch {epoch_prog:.2f} | loss={loss.item():.4f} | lr={opt.param_groups[0]['lr']:.2e} | VRAM: {vram_gb:.2f} GB",
                      flush=True)
            if is_main and step > 0 and step % ckpt_every == 0:
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

                # Keep only the latest 1 checkpoint to avoid exceeding Kaggle disk quota
                all_ckpts = sorted(glob.glob(os.path.join(args.out, "ckpt_[0-9]*.pt")))
                for old_ck in all_ckpts[:-1]:
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
        final_dest = os.path.join(args.out, "ckpt_final.pt")
        torch.save({"model": raw_model.state_dict(), "step": step}, final_dest)
        print(f"[train] Saved final checkpoint: {final_dest}")

        # Clean up all intermediate checkpoints since final checkpoint is successfully saved
        for old_ck in glob.glob(os.path.join(args.out, "ckpt_[0-9]*.pt")):
            try:
                os.remove(old_ck)
                print(f"[train] Removed intermediate {os.path.basename(old_ck)} to free disk space")
            except Exception:
                pass
        print("[train] done.")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
