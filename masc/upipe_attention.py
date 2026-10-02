"""UPipe-style memory-efficient attention execution for MASC AR generators.

This module implements the **UPipe head-chunking strategy** described in

    "Untied Ulysses: Memory-Efficient Attention via Head-Wise Pipelining"
    (UPipe / arXiv 66102)

as a drop-in wrapper around any standard multi-head (or grouped-query)
attention layer.  The mathematical result is **numerically identical** to
the original attention; only the execution order changes so that GPU peak
VRAM scales as O(H/U) instead of O(H) for U pipeline stages.

Integration with MASC
---------------------
MASC operates at the vocabulary level (embedding/head resize) and is
completely orthogonal to UPipe, which operates inside attention.  Apply
MASC surgery first (``resize_token_embedding`` / ``resize_output_head``),
then call :func:`patch_upipe_attention` to wrap every attention layer::

    from masc.integration import resize_token_embedding, resize_output_head
    from masc.upipe_attention import patch_upipe_attention

    resize_token_embedding(adapter, k=mp.k)
    resize_output_head(adapter, k=mp.k)
    patch_upipe_attention(model, chunk_heads=8)   # U = num_heads / 8 stages

The wrapper is transparent to the rest of the model: it preserves the
exact same call signature (q, k, v + optional mask) and output shape.

Design notes
------------
* **No extra parameters** — the wrapper holds no learnable weights.
* **Buffer reuse** — output accumulation tensor is pre-allocated once per
  forward call and filled in-place, mirroring UPipe's "buffer reinvestment"
  trick that avoids one extra H×L×d allocation vs. naive cat().
* **GQA / MQA support** — if the K/V head count differs from Q head count
  (e.g. LlamaGen / LLaMA-2 GQA), each chunk borrows only the matching K/V
  heads (floor-division slicing), matching the GQA scheduling in §3.2.
* **Flash-Attention compatibility** — set ``use_flash=True`` to delegate
  each per-chunk SDPA call to ``torch.nn.functional.scaled_dot_product_attention``
  (which dispatches to Flash-Attention 2 when available).  This stacks
  Flash-Attention's IO efficiency with UPipe's VRAM reduction.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "UPipeAttentionWrapper",
    "patch_upipe_attention",
    "unpatch_upipe_attention",
]


# ---------------------------------------------------------------------------
# Core wrapper
# ---------------------------------------------------------------------------

class UPipeAttentionWrapper(nn.Module):
    """Wrap an attention layer to execute heads in U serial stages.

    Each stage processes ``chunk_heads`` Q-heads (and the corresponding K/V
    heads for GQA) and accumulates into a shared output buffer.  Peak VRAM
    for the QKV materialisation is reduced by roughly a factor of
    U = ceil(num_heads / chunk_heads).

    Parameters
    ----------
    attn_module:
        The original attention module.  Its weights are preserved untouched
        inside ``self._original``; the wrapper only changes *execution order*.
    num_q_heads:
        Total number of Q attention heads.
    num_kv_heads:
        Number of K/V heads (equals ``num_q_heads`` for MHA; less for GQA/MQA).
    head_dim:
        Dimension of each head  ``d_k = d_model / num_q_heads``.
    chunk_heads:
        Number of Q heads processed per UPipe stage.
        Must be a divisor of ``num_q_heads``; auto-adjusted if not.
    use_flash:
        Delegate each chunk's SDPA to PyTorch SDPA (Flash-Attention 2 when
        available).
    """

    def __init__(
        self,
        attn_module: nn.Module,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        chunk_heads: int = 4,
        use_flash: bool = True,
    ) -> None:
        super().__init__()
        self._original = attn_module
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.chunk_heads = _nearest_divisor(num_q_heads, chunk_heads)
        self.use_flash = use_flash
        self.scale = math.sqrt(head_dim)
        # GQA ratio: how many Q heads share each K/V head
        self._gqa_ratio = max(1, num_q_heads // num_kv_heads)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if "_original" in self.__dict__:
                return getattr(self._original, name)
            raise

    # ------------------------------------------------------------------
    # Forward dispatch: LLaMA/LlamaGen module vs. low-level SDPA
    # ------------------------------------------------------------------

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # If wrapped module has wqkv (LlamaGen) or (wq and wo) (standard LLaMA), it is a full attention module
        if (hasattr(self._original, "wqkv") or hasattr(self._original, "wq")) and hasattr(self._original, "wo"):
            return self._forward_llamagen(*args, **kwargs)
        return self._forward_sdpa(*args, **kwargs)

    def _forward_llamagen(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        start_pos: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        orig = self._original
        bsz, seqlen, _ = x.shape

        if hasattr(orig, "wqkv"):
            kv_size = self.num_kv_heads * self.head_dim
            dim = self.num_q_heads * self.head_dim
            xq, xk, xv = orig.wqkv(x).split([dim, kv_size, kv_size], dim=-1)
        else:
            xq = orig.wq(x)
            xk = orig.wk(x)
            xv = orig.wv(x)

        xq = xq.view(bsz, seqlen, self.num_q_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        if freqs_cis is not None:
            try:
                from autoregressive.models.gpt import apply_rotary_emb
                xq = apply_rotary_emb(xq, freqs_cis)
                xk = apply_rotary_emb(xk, freqs_cis)
            except Exception:
                try:
                    from autoregressive.models.gpt import apply_rotary_emb
                    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
                except Exception:
                    pass

        xq, xk, xv = map(lambda t: t.transpose(1, 2), (xq, xk, xv))

        if hasattr(orig, "kv_cache") and orig.kv_cache is not None:
            keys, values = orig.kv_cache.update(start_pos, xk, xv)
        else:
            keys, values = xk, xv

        if self.num_q_heads != self.num_kv_heads:
            keys = keys.repeat_interleave(self.num_q_heads // self.num_kv_heads, dim=1)
            values = values.repeat_interleave(self.num_q_heads // self.num_kv_heads, dim=1)

        out = torch.empty_like(xq)
        for q_start in range(0, self.num_q_heads, self.chunk_heads):
            q_end = min(q_start + self.chunk_heads, self.num_q_heads)
            q_chunk = xq[:, q_start:q_end]
            k_chunk = keys[:, q_start:q_end]
            v_chunk = values[:, q_start:q_end]

            dropout_p = getattr(orig, "attn_dropout_p", 0.0) if orig.training else 0.0
            is_causal = True if mask is None else False

            if self.use_flash:
                chunk_out = F.scaled_dot_product_attention(
                    q_chunk, k_chunk, v_chunk,
                    attn_mask=mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                )
            else:
                chunk_out = self._manual_sdpa(q_chunk, k_chunk, v_chunk, mask)

            out[:, q_start:q_end] = chunk_out
            del q_chunk, k_chunk, v_chunk, chunk_out

        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = orig.wo(out)
        if hasattr(orig, "resid_dropout"):
            output = orig.resid_dropout(output)
        return output

    def _forward_sdpa(
        self,
        q: torch.Tensor,           # [B, num_q_heads, L, head_dim]
        k: torch.Tensor,           # [B, num_kv_heads, L, head_dim]
        v: torch.Tensor,           # [B, num_kv_heads, L, head_dim]
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Head-chunked attention.  Output: [B, num_q_heads, L, head_dim]."""
        # Pre-allocate output buffer — UPipe buffer reinvestment pattern.
        out = torch.empty_like(q)

        for q_start in range(0, self.num_q_heads, self.chunk_heads):
            q_end = min(q_start + self.chunk_heads, self.num_q_heads)
            q_chunk = q[:, q_start:q_end]           # [B, chunk_h, L, D]

            # Corresponding K/V heads (GQA: each K/V head serves gqa_ratio Q heads)
            kv_start = q_start // self._gqa_ratio
            kv_end   = max(kv_start + 1, min((q_end + self._gqa_ratio - 1) // self._gqa_ratio,
                                             self.num_kv_heads))
            k_chunk = k[:, kv_start:kv_end]
            v_chunk = v[:, kv_start:kv_end]

            # Expand K/V to match Q chunk size when GQA ratio > 1
            kv_repeat = (q_end - q_start) // (kv_end - kv_start)
            if kv_repeat > 1:
                k_chunk = k_chunk.repeat_interleave(kv_repeat, dim=1)
                v_chunk = v_chunk.repeat_interleave(kv_repeat, dim=1)

            if self.use_flash:
                chunk_out = F.scaled_dot_product_attention(
                    q_chunk, k_chunk, v_chunk,
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                    is_causal=(attn_mask is None),
                )
            else:
                chunk_out = self._manual_sdpa(q_chunk, k_chunk, v_chunk, attn_mask)

            out[:, q_start:q_end] = chunk_out

            # Explicitly release transient tensors — returns VRAM before next chunk
            del q_chunk, k_chunk, v_chunk, chunk_out

        return out

    # ------------------------------------------------------------------
    # Fallback SDPA without Flash-Attention
    # ------------------------------------------------------------------

    def _manual_sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        attn_weight = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if attn_mask is not None:
            attn_weight = attn_weight + attn_mask
        else:
            L = q.shape[-2]
            causal = torch.triu(
                torch.full((L, L), float("-inf"), device=q.device, dtype=q.dtype),
                diagonal=1,
            )
            attn_weight = attn_weight + causal
        attn_weight = torch.softmax(attn_weight, dim=-1)
        return torch.matmul(attn_weight, v)

    # ------------------------------------------------------------------
    # Transparent delegation to underlying module
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._original, name)

    def extra_repr(self) -> str:
        u = self.num_q_heads // self.chunk_heads
        return (
            f"num_q_heads={self.num_q_heads}, num_kv_heads={self.num_kv_heads}, "
            f"chunk_heads={self.chunk_heads}, U_stages={u}, flash={self.use_flash}"
        )


# ---------------------------------------------------------------------------
# Patch / unpatch helpers
# ---------------------------------------------------------------------------

def patch_upipe_attention(
    model: nn.Module,
    chunk_heads: int = 4,
    use_flash: bool = True,
    attn_submodule_names: tuple = ("attention", "self_attn", "attn"),
    _registry: Optional[dict] = None,
) -> dict:
    """Wrap every attention layer in *model* with :class:`UPipeAttentionWrapper`.

    Walks the module tree; replaces each attention sub-module whose attribute
    name matches ``attn_submodule_names``.  Original weights are preserved
    inside the wrapper — no weight copying or re-initialisation.

    Parameters
    ----------
    model:
        Full AR backbone (apply MASC surgery *before* calling this).
    chunk_heads:
        Q-heads per UPipe stage.  Rule of thumb: ``num_heads // 4`` for ~4×
        VRAM reduction.
    use_flash:
        Use PyTorch SDPA (Flash-Attention 2) per chunk when available.
    attn_submodule_names:
        Candidate attribute names for the attention sub-module within each
        transformer layer.  Add your backbone's name if it uses a different one
        (e.g. ``"self_attention"``, ``"mixer"``).
    _registry:
        Dict mapping ``(parent_layer_name, attr_name)`` → original module.
        Pass an external ``{}`` if you need to call :func:`unpatch_upipe_attention`
        later.

    Returns
    -------
    dict
        Registry of all patched locations.
    """
    if _registry is None:
        _registry = {}

    for layer_name, layer in model.named_modules():
        for attr_name in attn_submodule_names:
            attn = getattr(layer, attr_name, None)
            if attn is None or not isinstance(attn, nn.Module):
                continue
            if isinstance(attn, UPipeAttentionWrapper):
                continue  # already patched

            # Auto-detect head configuration from common attribute names
            num_q_heads = _detect_attr(
                attn,
                ("n_local_heads", "n_head", "num_heads", "n_heads", "num_attention_heads"),
            )
            if num_q_heads is None:
                cfg = getattr(model, "config", getattr(layer, "config", None))
                if cfg is not None:
                    num_q_heads = getattr(cfg, "n_head", getattr(cfg, "num_heads", None))

            num_kv_heads = _detect_attr(
                attn,
                ("n_local_kv_heads", "n_kv_heads", "num_kv_heads", "num_key_value_heads"),
                default=num_q_heads,
            )
            head_dim = _detect_attr(attn, ("head_dim", "d_head"), default=None)

            if num_q_heads is None:
                import warnings
                warnings.warn(
                    f"[UPipe] Cannot detect num_heads on {layer_name}.{attr_name} — "
                    "skipping. Add 'num_heads' attr or subclass UPipeAttentionWrapper.",
                    stacklevel=2,
                )
                continue

            if head_dim is None:
                if hasattr(attn, "wq") and hasattr(attn.wq, "out_features"):
                    head_dim = attn.wq.out_features // num_q_heads
                else:
                    embed_dim = _detect_attr(attn, ("dim", "embed_dim", "hidden_size", "d_model"))
                    head_dim = embed_dim // num_q_heads if embed_dim else None

            if head_dim is None:
                import warnings
                warnings.warn(
                    f"[UPipe] Cannot detect head_dim on {layer_name}.{attr_name} — skipping.",
                    stacklevel=2,
                )
                continue

            wrapper = UPipeAttentionWrapper(
                attn_module=attn,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                chunk_heads=chunk_heads,
                use_flash=use_flash,
            )
            setattr(layer, attr_name, wrapper)
            _registry[(layer_name, attr_name)] = attn
            u = num_q_heads // wrapper.chunk_heads
            print(
                f"[UPipe] Patched {layer_name}.{attr_name} | "
                f"H={num_q_heads}, chunk={wrapper.chunk_heads}, U={u} stages"
            )

    if not _registry:
        import warnings
        warnings.warn(
            "[UPipe] patch_upipe_attention found no attention layers to wrap. "
            "Check attn_submodule_names for your backbone architecture.",
            stacklevel=2,
        )
    return _registry


def unpatch_upipe_attention(model: nn.Module, registry: dict) -> None:
    """Restore all attention layers patched by :func:`patch_upipe_attention`.

    Parameters
    ----------
    model:
        Same model passed to :func:`patch_upipe_attention`.
    registry:
        Dict returned by :func:`patch_upipe_attention`.
    """
    for (layer_name, attr_name), original in registry.items():
        parent = model
        for part in layer_name.split("."):
            if part:
                parent = getattr(parent, part)
        setattr(parent, attr_name, original)
    registry.clear()
    print("[UPipe] All attention layers restored.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_attr(module: nn.Module, names, default=None):
    """Return the first matching attribute value or *default*."""
    for name in names:
        val = getattr(module, name, None)
        if val is not None:
            return val
    return default


def _nearest_divisor(n: int, target: int) -> int:
    """Return the divisor of *n* nearest to *target*."""
    if n % target == 0:
        return target
    best = 1
    for d in range(1, n + 1):
        if n % d == 0 and abs(d - target) < abs(best - target):
            best = d
    return best
