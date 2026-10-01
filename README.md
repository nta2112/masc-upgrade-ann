# MASC: Manifold-Aligned Semantic Clustering

Reference implementation for the ICML 2026 paper
**“A Flat Vocabulary or a Rich Hierarchy? Re-introducing Intrinsic Structure
Transforms the Autoregressive Image Generation.”**

📄 [Paper](docs/static/MASC_paper.pdf) &nbsp;|&nbsp; 🌐 [Project Page](https://lixuan27.github.io/MASC/) &nbsp;|&nbsp; 💻 [GitHub](https://github.com/nta2112/masc-upgrade-ann)

> **TL;DR.** Discrete autoregressive (AR) image generators predict over a vast,
> *flat* vocabulary of visual tokens, ignoring the fact that the codebook
> embeddings lie on a low-dimensional **semantic manifold**. MASC is a
> one-time, offline preprocessing module that builds a density-driven,
> manifold-aligned **hierarchical semantic tree** over the codebook and uses it
> to turn the hard `N`-way token prediction into a structured `k`-way
> coarse-cluster prediction. It is **plug-and-play**: it accelerates training by
> up to **71%** and improves quality (LlamaGen-XL FID **2.87 → 2.49**) across a
> wide range of AR backbones, with **no architectural changes** to the backbone.

---

## Install

```bash
git clone https://github.com/nta2112/masc-upgrade-ann MASC && cd MASC
python -m pip install -r requirements.txt   # numpy/scipy required; torch for training
```

The MASC **preprocessing core** (clustering / mapping / relabel / decode) depends
only on `numpy`. PyTorch is needed only for the AR-backbone integration helpers
and the training/eval scripts.

## Quickstart — verify the algorithm in 5 seconds

No assets required. Build a MASC tree on a synthetic manifold-structured
codebook and confirm it recovers the latent semantic groups:

```bash
python examples/demo_cluster_codebook.py
#   cluster purity vs latent groups: 1.000  (groups perfectly recovered)
#   figure saved -> examples/masc_demo.png

python scripts/build_masc_tree.py --demo --k 16 --out /tmp/masc.npz --verbose
```

Run the test suite (the clustering is cross-checked against SciPy’s
average-linkage, confirming Algorithm 1 is exact):

```bash
python tests/test_distance.py
python tests/test_clustering.py
python tests/test_relabel_decode.py
# (or: pytest -q tests/)
```

---

## The three-line integration

MASC changes a training pipeline in exactly three places:

```python
from masc import build_masc_tree, load_mapping, relabel_targets, random_sample_decode

# (1) OFFLINE, ONCE: build the coarse vocabulary from the tokenizer codebook.
tree = build_masc_tree(codebook_embeddings, k=8192)          # Algorithm 1
save_mapping("masc_mapping.npz", tree.mapping, tree.k)

# (2) TRAIN: predict the next coarse cluster instead of the next fine token.
coarse_targets = relabel_targets(fine_token_ids, tree.mapping)   # Eq. 5

# (3) SAMPLE: decode a predicted cluster back to a token (default strategy).
fine_tokens = random_sample_decode(coarse_cluster_indices, mapping)
```

Everything else — the transformer, class conditioning, CFG, KV-cache and the
AR sampling loop — is the **unmodified backbone**.

---

## Citation

```bibtex
@inproceedings{he2026masc,
  title     = {A Flat Vocabulary or a Rich Hierarchy? Re-introducing Intrinsic
               Structure Transforms the Autoregressive Image Generation},
  author    = {He, Lixuan and Zheng, Shikang},
  booktitle = {Proceedings of the 43rd International Conference on Machine
               Learning (ICML)},
  year      = {2026}
}
```

---

## MASC + UPipe Integration (this fork)

This repository extends the original MASC with **UPipe** (Untied Ulysses
head-chunked attention execution), enabling training on consumer GPUs such as
Kaggle 2× T4 16 GB — without changing the mathematical result of attention.

### What's new

| Module | Description |
|---|---|
| `masc/upipe_attention.py` | `UPipeAttentionWrapper`, `patch_upipe_attention()`, `unpatch_upipe_attention()` |
| `masc/integration.py` | `apply_masc_and_upipe()` — single entry point for both surgeries |
| `configs/llamagen_l_masc_upipe_kaggle.yaml` | Kaggle-optimised config (T4 16 GB, batch 8, grad accum 16) |
| `scripts/train.py` | New flags: `--upipe-chunk-heads`, `--grad-ckpt`, `--resume` |
| `docs/kaggle_guide.md` | Step-by-step Kaggle notebook setup |

### Quick start (MASC + UPipe combined)

```python
from masc import load_mapping
from masc.integration import apply_masc_and_upipe

mp = load_mapping("masc_mapping_k4096.npz")

# Step 1+2 in one call: MASC vocab surgery → then UPipe attention patching
registry = apply_masc_and_upipe(
    model=model,
    adapter=adapter,          # your ARBackbone adapter
    mapping=mp.mapping,
    k=mp.k,                   # 4096 for Kaggle T4
    upipe_chunk_heads=4,      # LlamaGen-L has 16 heads → 4 stages → ~4× VRAM saving
    upipe_use_flash=True,
)
```

### Train command (Kaggle 2× T4)

```bash
python scripts/train.py \
  --config configs/llamagen_l_masc_upipe_kaggle.yaml \
  --mapping /kaggle/working/masc_mapping_k4096.npz \
  --codes /kaggle/input/imagenet-tokens/codes/ \
  --out /kaggle/working/checkpoints/ \
  --upipe-chunk-heads 4 \
  --grad-ckpt \
  --resume /kaggle/input/llamagen-weights/c2i_L_256.pt
```

See [`docs/kaggle_guide.md`](docs/kaggle_guide.md) for the complete step-by-step
setup including dataset preparation and VRAM budget analysis.
