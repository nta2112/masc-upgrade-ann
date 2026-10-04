"""Plugging the MASC prior into an autoregressive backbone.

MASC is a *plug-and-play* module: it changes the prediction target of an
otherwise unmodified AR image generator from an ``N``-way token classification
to a ``k``-way coarse-cluster classification (Section 3.3).  Concretely, three
things change relative to a vanilla backbone:

1.  The **token-embedding table** is indexed by coarse cluster ids (size ``k``)
    instead of fine token ids (size ``N``).
2.  The **output classification head** produces ``k`` logits instead of ``N``.
3.  The **targets** fed to the cross-entropy loss are relabelled with ``M``
    (:func:`masc.relabel.relabel_targets`).

The two helpers below perform (1) and (2) for a *generic* decoder-only
transformer and are written against a minimal duck-typed interface
(:class:`ARBackbone`).  They are intentionally backbone-agnostic: every
published AR generator we evaluate (LlamaGen, VAR, RandAR, IAR, CTF, GigaTok,
RAR) exposes its embedding/head under different attribute names and couples them
to its own class-conditioning, CFG and sampling logic.  Wiring MASC into a
specific backbone therefore means implementing the small :class:`ARBackbone`
adapter against that project's source tree — see ``docs/reproduction.md`` and
the per-backbone notes.  We deliberately do **not** vendor third-party backbone
code here; please obtain it from the respective official repositories.

This file requires PyTorch; the numpy MASC core (clustering / mapping / relabel
/ random-sampling decode) does not.
"""

from __future__ import annotations

from typing import Optional, Protocol, Union, runtime_checkable

import numpy as np

try:  # torch is only needed for the model-surgery helpers
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover - exercised only without torch
    raise ImportError(
        "masc.integration requires PyTorch. Install torch to use the AR-backbone "
        "helpers; the numpy MASC core (masc.clustering / mapping / relabel / decode) "
        "is usable without it."
    ) from exc

__all__ = [
    "ARBackbone",
    "resize_token_embedding",
    "resize_output_head",
    "MASCObjective",
    "apply_masc_and_upipe",
]


@runtime_checkable
class ARBackbone(Protocol):
    """Minimal interface a backbone must expose to receive the MASC prior.

    Implement a thin adapter that returns the embedding / head submodules of the
    *specific* model you are using.  Everything else (causal masking, class
    conditioning, KV-cache, classifier-free guidance, the autoregressive sampling
    loop) is owned by the backbone and is left untouched by MASC.
    """

    def get_token_embedding(self) -> "nn.Embedding":  # vocab embedding table
        ...

    def set_token_embedding(self, emb: "nn.Embedding") -> None:
        ...

    def get_output_head(self) -> "nn.Linear":          # final vocab projection
        ...

    def set_output_head(self, head: "nn.Linear") -> None:
        ...


def resize_token_embedding(
    backbone: "ARBackbone",
    k: int,
    mapping: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
) -> None:
    """Replace the backbone's token-embedding table with a ``k``-row table.

    If ``mapping`` (shape [N]) is provided and matches the old embedding's vocab size,
    the new embedding rows are initialised as cluster centroids (the average of
    pretrained token embeddings belonging to each cluster). This preserves pretrained
    representations for fast fine-tuning.
    Otherwise, falls back to normal initialisation N(0, 0.02).
    """
    old = backbone.get_token_embedding()
    new = nn.Embedding(k, old.embedding_dim)

    initialized = False
    if mapping is not None:
        try:
            m = torch.as_tensor(np.asarray(mapping), dtype=torch.int64, device=old.weight.device)
            if m.shape[0] == old.weight.shape[0]:
                counts = torch.bincount(m, minlength=k).float().unsqueeze(1)
                old_w = old.weight.detach().to(dtype=torch.float32)
                sum_w = torch.zeros(k, old.embedding_dim, device=old.weight.device, dtype=torch.float32)
                sum_w.index_add_(0, m, old_w)
                mask = (counts > 0)
                centroid_w = torch.where(mask, sum_w / torch.clamp(counts, min=1.0), torch.zeros_like(sum_w))
                if (~mask.squeeze(1)).any():
                    rand_w = torch.randn_like(centroid_w) * 0.02
                    centroid_w = torch.where(mask, centroid_w, rand_w)
                new.weight.data.copy_(centroid_w.to(dtype=old.weight.dtype))
                initialized = True
                print(f"[MASC] Initialized token embeddings via cluster centroids (N={m.shape[0]} -> k={k}).")
            else:
                print(f"[MASC] Mapping size ({m.shape[0]}) != old vocab ({old.weight.shape[0]}), fallback to N(0, 0.02).")
        except Exception as e:
            print(f"[MASC] Warning: Centroid initialization failed ({e}), fallback to N(0, 0.02).")

    if not initialized:
        nn.init.normal_(new.weight, mean=0.0, std=0.02)
        print(f"[MASC] Initialized token embeddings with N(0, 0.02) (k={k}).")

    new = new.to(old.weight.device, old.weight.dtype)
    backbone.set_token_embedding(new)


def resize_output_head(
    backbone: "ARBackbone",
    k: int,
    mapping: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
) -> None:
    """Replace the final projection so it emits ``k`` logits.

    If ``mapping`` (shape [N]) is provided and matches the old head's out_features,
    the new head weights and bias are initialised as cluster centroid averages of the
    pretrained head. This preserves pretrained classification capabilities for fine-tuning.
    Otherwise, falls back to normal initialisation N(0, 0.02).
    """
    old = backbone.get_output_head()
    in_features = old.in_features
    has_bias = old.bias is not None
    new = nn.Linear(in_features, k, bias=has_bias)

    initialized = False
    if mapping is not None:
        try:
            m = torch.as_tensor(np.asarray(mapping), dtype=torch.int64, device=old.weight.device)
            if m.shape[0] == old.weight.shape[0]:
                counts = torch.bincount(m, minlength=k).float().unsqueeze(1)
                old_w = old.weight.detach().to(dtype=torch.float32)
                sum_w = torch.zeros(k, in_features, device=old.weight.device, dtype=torch.float32)
                sum_w.index_add_(0, m, old_w)
                mask = (counts > 0)
                centroid_w = torch.where(mask, sum_w / torch.clamp(counts, min=1.0), torch.zeros_like(sum_w))
                if (~mask.squeeze(1)).any():
                    rand_w = torch.randn_like(centroid_w) * 0.02
                    centroid_w = torch.where(mask, centroid_w, rand_w)
                new.weight.data.copy_(centroid_w.to(dtype=old.weight.dtype))

                if has_bias:
                    old_b = old.bias.detach().to(dtype=torch.float32)
                    sum_b = torch.zeros(k, device=old.weight.device, dtype=torch.float32)
                    sum_b.index_add_(0, m, old_b)
                    counts_b = counts.squeeze(1)
                    mask_b = (counts_b > 0)
                    centroid_b = torch.where(mask_b, sum_b / torch.clamp(counts_b, min=1.0), torch.zeros_like(sum_b))
                    new.bias.data.copy_(centroid_b.to(dtype=old.bias.dtype))

                initialized = True
                print(f"[MASC] Initialized output head via cluster centroids (N={m.shape[0]} -> k={k}).")
            else:
                print(f"[MASC] Mapping size ({m.shape[0]}) != old head ({old.weight.shape[0]}), fallback to N(0, 0.02).")
        except Exception as e:
            print(f"[MASC] Warning: Centroid initialization failed ({e}), fallback to N(0, 0.02).")

    if not initialized:
        nn.init.normal_(new.weight, mean=0.0, std=0.02)
        if has_bias:
            nn.init.zeros_(new.bias)
        print(f"[MASC] Initialized output head with N(0, 0.02) (k={k}).")

    new = new.to(old.weight.device, old.weight.dtype)
    backbone.set_output_head(new)


class MASCObjective:
    """Relabel targets with ``M`` and compute the coarse cross-entropy.

    This wraps the *only* change MASC makes to the loss.  ``mapping`` is kept as
    a buffer on the target device so relabelling is a single on-device gather.

    Notes
    -----
    The exact reduction, label smoothing, class-conditioning tokens, and (for
    permutation/next-scale models such as RAR/VAR) the ordering of the coarse
    target sequence are properties of the backbone's training objective.  Match
    them to your backbone; the snippet below covers the standard next-token
    decoder-only case used by LlamaGen.
    """

    def __init__(self, mapping: np.ndarray, device: "torch.device" = None):
        m = torch.as_tensor(np.asarray(mapping, dtype=np.int64))
        self.mapping = m.to(device) if device is not None else m
        self.k = int(self.mapping.max().item()) + 1

    def coarse_targets(self, fine_token_ids: "torch.Tensor") -> "torch.Tensor":
        """``z^b = M(z^q)`` on device, preserving shape."""
        return self.mapping[fine_token_ids.long()]

    def loss(self, logits: "torch.Tensor", fine_token_ids: "torch.Tensor") -> "torch.Tensor":
        """Cross-entropy of ``k``-way coarse prediction.

        ``logits``: ``[B, L, k]``;  ``fine_token_ids``: ``[B, L]`` fine targets.
        """
        target = self.coarse_targets(fine_token_ids)
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1)
        )


# ---------------------------------------------------------------------------
# Combined MASC + UPipe surgery (convenience entry point)
# ---------------------------------------------------------------------------

def apply_masc_and_upipe(
    model: "nn.Module",
    adapter: "ARBackbone",
    mapping: "np.ndarray",
    k: int,
    upipe_chunk_heads: Optional[int] = None,
    upipe_use_flash: bool = True,
    attn_submodule_names: tuple = ("attention", "self_attn", "attn"),
    init_from_centroids: bool = True,
) -> "UPipeRegistry":
    """Apply MASC vocabulary surgery and (optionally) UPipe attention patching.

    Call this *once* after the model is on device, before the optimiser is
    created.  The correct application order is:

    1. MASC: resize embedding table + output head to k-way coarse vocab.
       If `init_from_centroids` is True, initialize using cluster-average
       centroids of pretrained weights to enable fast fine-tuning.
    2. UPipe: wrap every attention layer for head-chunked execution.

    Order matters: UPipe must see the already-resized model so it can
    detect head attributes correctly.

    Parameters
    ----------
    model:
        The backbone ``nn.Module`` (must already be on device).
    adapter:
        :class:`ARBackbone` adapter for *model*.
    mapping:
        MASC token mapping array (``MASCMapping.mapping``, shape ``[N]``).
    k:
        Target coarse vocabulary size.
    upipe_chunk_heads:
        Q-heads per UPipe pipeline stage.  Pass ``None`` to skip UPipe
        (MASC-only mode).  Rule of thumb: ``num_heads // 4``.
    upipe_use_flash:
        Use PyTorch SDPA (Flash-Attention 2) per chunk.
    attn_submodule_names:
        Attention submodule names to search; forwarded to
        :func:`~masc.upipe_attention.patch_upipe_attention`.
    init_from_centroids:
        Whether to initialize the resized embedding and head as cluster
        centroids of the pretrained model weights. Default: True.

    Returns
    -------
    dict
        UPipe registry (empty dict when UPipe is disabled).  Pass to
        :func:`~masc.upipe_attention.unpatch_upipe_attention` to undo.
    """
    # Step 1 — MASC: vocabulary surgery
    map_for_init = mapping if init_from_centroids else None
    resize_token_embedding(adapter, k, mapping=map_for_init)
    resize_output_head(adapter, k, mapping=map_for_init)
    print(f"[MASC+UPipe] MASC surgery done: vocab {mapping.shape[0]} → k={k}")

    # Step 2 — UPipe: attention patching (optional)
    registry: dict = {}
    if upipe_chunk_heads is not None:
        from masc.upipe_attention import patch_upipe_attention
        registry = patch_upipe_attention(
            model,
            chunk_heads=upipe_chunk_heads,
            use_flash=upipe_use_flash,
            attn_submodule_names=attn_submodule_names,
            _registry=registry,
        )
        print(f"[MASC+UPipe] UPipe patching done: {len(registry)} layers wrapped.")
    else:
        print("[MASC+UPipe] UPipe disabled (upipe_chunk_heads=None).")

    return registry


# Type alias for the registry dict (for readability in user code)
UPipeRegistry = dict
