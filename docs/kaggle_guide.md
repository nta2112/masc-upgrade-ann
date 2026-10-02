# Hướng Dẫn Chạy MASC + UPipe trên Kaggle (2× T4 16 GB)

Tài liệu này hướng dẫn đầy đủ từng bước để chạy MASC + UPipe trên
Kaggle Free Tier với 2× NVIDIA T4 (mỗi GPU 16 GB VRAM).

---

## Tổng quan tài nguyên Kaggle Free Tier

| Tài nguyên | Giá trị |
|---|---|
| GPU | 2× NVIDIA T4 (Turing, 16 GB GDDR6 mỗi cái) |
| VRAM tổng | 32 GB (16 GB × 2) |
| RAM hệ thống | 29 GB |
| Disk /kaggle/working | 20 GB |
| Disk input (read-only) | Không giới hạn (dataset) |
| Session timeout | 12h/session, 30h/week |
| Internet | Tắt khi submit (bật khi interactive) |

---

## Phần 1 — Chuẩn Bị Dữ Liệu Trên Kaggle

### 1.1 Tạo Kaggle Dataset: LlamaGen Weights

Tải c2i_L_256.pt (~1.4 GB) từ
[FoundationVision/LlamaGen releases](https://github.com/FoundationVision/LlamaGen)
rồi upload lên Kaggle Datasets:

`ash
# Trên máy local (Windows PowerShell):
kaggle datasets init -p llamagen-weights
# Đặt c2i_L_256.pt vào thư mục llamagen-weights/
kaggle datasets create -p llamagen-weights --dir-mode zip
`

Hoặc upload thủ công tại https://www.kaggle.com/datasets → "New Dataset".

Sau khi tạo, dataset sẽ mount tại /kaggle/input/llamagen-weights/.

### 1.2 Tạo Kaggle Dataset: ImageNet Token Codes

Kaggle có sẵn ImageNet-1K raw images nhưng nặng 155 GB — **không cần tải ảnh**.
Chỉ cần pre-extracted token codes (~8–12 GB) do LlamaGen cung cấp sẵn.

**Cách 1: Dùng LlamaGen's pre-extracted codes (khuyến nghị)**

`ash
# LlamaGen cung cấp imagenet_code_c2i_flip_ten_crop tại HuggingFace
# Tải về rồi upload lên Kaggle Dataset "imagenet-tokens"
# Layout mong đợi:
# imagenet-tokens/
#   codes/
#     shard_0000.npy   # [B, 256] int16 token ids
#     shard_0001.npy
#     ...
#     labels.npy       # [N_total] int class labels 0-999
`

**Cách 2: Tự extract trong notebook (nếu có ImageNet dataset trên Kaggle)**

`python
# Cell đầu notebook: extract codes từ ảnh
import torch
from pathlib import Path

# Mount VQ-VAE tokenizer
vq_vae = torch.load("/kaggle/input/llamagen-weights/vq_ds16_c2i.pt")
vq_vae.eval().cuda()

# Extract và lưu codes
out_dir = Path("/kaggle/working/codes")
out_dir.mkdir(exist_ok=True)
# ... (xem docs/reproduction.md Step 3)
`

---

## Phần 2 — Tạo Notebook Kaggle

### 2.1 Notebook template

Tạo notebook mới trên Kaggle với cấu hình:
- **Accelerator:** GPU T4 × 2
- **Internet:** ON (để clone repo)
- **Persistence:** Files (lưu /kaggle/working giữa các session)

### 2.2 Cell 1 — Clone repo và install

`python
# Cell 1: Setup environment
import subprocess, os

# Clone repo mới nhất
subprocess.run([
    "git", "clone",
    "https://github.com/nta2112/masc-upgrade-ann",
    "/kaggle/working/MASC"
], check=True)
os.chdir("/kaggle/working/MASC")

# Install dependencies
subprocess.run([
    "pip", "install", "-q",
    "pyyaml>=6.0",
    "accelerate>=0.27",
    "torch-fidelity>=0.3.0",
], check=True)

print("✅ Setup done")
`

### 2.3 Cell 2 — Clone LlamaGen backbone

`python
# Cell 2: Clone LlamaGen (chỉ cần lần đầu)
import os

if not os.path.exists("/kaggle/working/LlamaGen"):
    subprocess.run([
        "git", "clone",
        "https://github.com/FoundationVision/LlamaGen",
        "/kaggle/working/LlamaGen"
    ], check=True)
    print("✅ LlamaGen cloned")
else:
    print("✅ LlamaGen already exists")

# Thêm vào PYTHONPATH để import backbone
import sys
sys.path.insert(0, "/kaggle/working/LlamaGen")
sys.path.insert(0, "/kaggle/working/MASC")
`

### 2.4 Cell 3 — Build MASC mapping (chạy một lần)

`python
# Cell 3: Build MASC tree từ VQ-VAE codebook
import numpy as np, torch
from pathlib import Path
import sys
sys.path.insert(0, "/kaggle/working/MASC")

MAPPING_PATH = Path("/kaggle/working/masc_mapping_k4096.npz")

if MAPPING_PATH.exists():
    print(f"✅ Mapping exists: {MAPPING_PATH}")
else:
    print("⏳ Building MASC tree (k=4096)...")

    # Load VQ-VAE codebook từ LlamaGen pretrained tokenizer
    from tokenizer.tokenizer_image.vq_model import VQ_models
    vq_model = VQ_models["VQ-16"](codebook_size=16384, codebook_embed_dim=8)
    ckpt = torch.load("/kaggle/input/llamagen-weights/vq_ds16_c2i.pt",
                      map_location="cpu")
    state = ckpt.get("model", ckpt.get("ema", ckpt))
    vq_model.load_state_dict(state, strict=False)
    # Extract codebook embeddings [N, d_vq]
    codebook = vq_model.quantize.embedding.weight.detach().cpu().numpy()
    print(f"Codebook shape: {codebook.shape}")  # (16384, 8)

    # Build MASC tree
    from masc import build_masc_tree, save_mapping
    tree = build_masc_tree(codebook, k=4096)
    save_mapping(str(MAPPING_PATH), tree.mapping, tree.k)
    print(f"✅ MASC mapping built: N={tree.n} → k={tree.k}")
    print(f"   Saved to {MAPPING_PATH}")
`

### 2.5 Cell 4 — Build LlamaGen Adapter (chạy một lần)

`python
# Cell 4: Tạo backbones/llamagen_adapter.py
adapter_code = '''
"""LlamaGen → MASC ARBackbone adapter.

LlamaGen-L attribute map:
  Token embedding : model.tok_embeddings   (nn.Embedding)
  Output head     : model.output           (nn.Linear, no bias)
  Attention layers: model.layers[i].attention  (attr name: "attention")
"""
import sys, os
sys.path.insert(0, "/kaggle/working/LlamaGen")

import torch
import torch.nn as nn
from masc.integration import ARBackbone, apply_masc_and_upipe
from masc import load_mapping

# Import LlamaGen model builder
from autoregressive.models.gpt import GPT_models

class LlamaGenAdapter:
    """ARBackbone adapter for LlamaGen GPT models."""
    def __init__(self, model):
        self.m = model

    def get_token_embedding(self) -> nn.Embedding:
        return self.m.tok_embeddings

    def set_token_embedding(self, emb: nn.Embedding) -> None:
        self.m.tok_embeddings = emb

    def get_output_head(self) -> nn.Linear:
        return self.m.output

    def set_output_head(self, head: nn.Linear) -> None:
        self.m.output = head


def build_backbone(cfg: dict):
    """Load LlamaGen-L and return (model, adapter).

    Expected cfg keys:
        name: "llamagen_l"
        pretrained: path to c2i_L_256.pt
        image_size: 256
    """
    model_name_map = {
        "llamagen_b": "GPT-B",
        "llamagen_l": "GPT-L",
        "llamagen_xl": "GPT-XL",
        "llamagen_xxl": "GPT-XXL",
    }
    gpt_key = model_name_map[cfg["name"]]
    model = GPT_models[gpt_key](
        vocab_size=16384,         # fine vocab; will be resized by MASC
        block_size=256,           # 256 tokens per 256x256 image (VQ-16)
        num_classes=1000,
        cls_token_num=1,
        model_type="c2i",
        resid_dropout_p=0.1,
        ffn_dropout_p=0.1,
        drop_path_rate=0.0,
    )

    # Load pretrained checkpoint
    pretrained = cfg.get("pretrained")
    if pretrained and os.path.exists(pretrained):
        ckpt = torch.load(pretrained, map_location="cpu")
        state = ckpt.get("model", ckpt)  # handle both formats
        # Load with strict=False: embedding/head sizes will be resized by MASC
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[build_backbone] Loaded {pretrained}")
        print(f"  Missing keys (will be re-init by MASC): {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")
    else:
        print(f"[build_backbone] WARNING: pretrained not found: {pretrained}")

    adapter = LlamaGenAdapter(model)
    return model, adapter
'''

import os
os.makedirs("/kaggle/working/MASC/backbones", exist_ok=True)
with open("/kaggle/working/MASC/backbones/__init__.py", "w") as f:
    f.write("from .llamagen_adapter import build_backbone\n")
with open("/kaggle/working/MASC/backbones/llamagen_adapter.py", "w") as f:
    f.write(adapter_code)
print("✅ backbones/llamagen_adapter.py created")
`

### 2.6 Cell 5 — Train (MASC + UPipe, 2× T4)

`python
# Cell 5: Launch training với torchrun (2 GPU)
import subprocess

cmd = [
    "torchrun",
    "--nproc_per_node=2",           # 2× T4
    "--master_port=29500",
    "/kaggle/working/MASC/scripts/train.py",
    "--config",   "/kaggle/working/MASC/configs/llamagen_l_masc_upipe_kaggle.yaml",
    "--mapping",  "/kaggle/working/masc_mapping_k4096.npz",
    "--codes",    "/kaggle/input/imagenet-tokens/codes/",
    "--out",      "/kaggle/working/checkpoints/",
    "--upipe-chunk-heads", "4",
    "--grad-ckpt",
    "--resume",   "/kaggle/input/llamagen-weights/c2i_L_256.pt",
]

result = subprocess.run(cmd, cwd="/kaggle/working/MASC",
                        capture_output=False)
`

> **Nếu chỉ có 1 GPU** (hoặc muốn debug đơn giản):
> `python
> cmd = ["python", "/kaggle/working/MASC/scripts/train.py", ...]
> # Bỏ torchrun --nproc_per_node=2
> `

---

## Phần 3 — VRAM Budget Chi Tiết

### 3.1 Phân tích per-GPU (T4 16 GB)

| Hạng mục | fp16 | Ghi chú |
|---|---|---|
| LlamaGen-L weights | ~700 MB | 343M × 2 bytes |
| MASC embedding (k=4096) | ~8 MB | 4096 × 1024 × 2 bytes |
| Output head (k=4096) | ~8 MB | 4096 × 1024 × 2 bytes |
| Optimizer (AdamW) | ~1.4 GB | 2× weights |
| Gradient buffer | ~700 MB | 1× weights |
| **Static total** | **~2.8 GB** | |
| Activation (vanilla, B=8, L=256) | ~9–11 GB | ❌ OOM! |
| Activation + grad_ckpt | ~3–4 GB | Recompute fwd pass |
| Activation + grad_ckpt + UPipe (U=4) | ~1.5–2 GB | /4 từ attn peak |
| **Grand total (MASC+UPipe+ckpt)** | **~5–7 GB** | ✅ Fit T4 |

### 3.2 Sơ đồ VRAM usage per-GPU

`
16 GB ████████████████████████████████
       limit

 7 GB ██████████████                   MASC+UPipe+grad_ckpt ✅✅
10 GB ████████████████████             UPipe only            ✅
14 GB ████████████████████████████     MASC only             ⚠️
~OOM  ████████████████████████████████ Vanilla               ❌
`

---

## Phần 4 — Multi-GPU với torchrun (2× T4)

Kaggle cung cấp 2 GPU trong cùng một máy (single node).
	orchrun --nproc_per_node=2 sẽ spawn 2 processes, mỗi process chiếm 1 GPU.

### 4.1 Effective batch size

`
local_batch = 8            # per GPU, per step
grad_accum  = 16           # accumulate before optimizer.step()
num_gpus    = 2

effective_batch = local_batch × grad_accum × num_gpus
                = 8 × 16 × 2 = 256 samples/step
`

Paper dùng global_batch=256 trên 8× H100 → **cùng effective batch size**!

### 4.2 Tương đương tốc độ

`
Paper:  8× H100 (80 GB), batch=256, ~300 epochs = ~300K steps
Kaggle: 2× T4  (16 GB), batch=256 (via grad_accum), 15K steps ≈ 0.5% của full run
→ Proof-of-concept: đủ để verify loss giảm + so sánh VRAM
`

---

## Phần 5 — Checklist Trước Khi Chạy

- [ ] Dataset llamagen-weights đã có c2i_L_256.pt và q_ds16_c2i.pt
- [ ] Dataset imagenet-tokens đã có codes/*.npy + labels.npy
- [ ] Notebook accelerator = **GPU T4 × 2**
- [ ] Internet = **ON** (để git clone)
- [ ] Cell 1–4 đã chạy thành công (setup + MASC mapping)
- [ ] /kaggle/working/masc_mapping_k4096.npz tồn tại
- [ ] /kaggle/working/MASC/backbones/llamagen_adapter.py tồn tại
- [ ] Bắt đầu training với Cell 5

---

## Phần 6 — Quản Lý Session (12h limit)

### 6.1 Auto-save checkpoint

Config llamagen_l_masc_upipe_kaggle.yaml đã có ckpt_every: 2000.
Ở bước 2000, 4000, ... file ckpt_2000.pt sẽ được lưu vào
/kaggle/working/checkpoints/.

### 6.2 Resume sau khi session hết

`python
# Cell resume — session mới, dùng lại checkpoint cũ
# Giả sử checkpoint đã được lưu ra Kaggle Output Dataset
cmd = [
    "torchrun", "--nproc_per_node=2", "--master_port=29500",
    "/kaggle/working/MASC/scripts/train.py",
    "--config",   "/kaggle/working/MASC/configs/llamagen_l_masc_upipe_kaggle.yaml",
    "--mapping",  "/kaggle/working/masc_mapping_k4096.npz",
    "--codes",    "/kaggle/input/imagenet-tokens/codes/",
    "--out",      "/kaggle/working/checkpoints/",
    "--upipe-chunk-heads", "4",
    "--grad-ckpt",
    "--resume",   "/kaggle/input/my-masc-checkpoints/ckpt_2000.pt",  # ← checkpoint cũ
]
`

> **Tip:** Sau mỗi session, vào Kaggle notebook → **Save & Run All** →
> Output sẽ tự lưu vào Kaggle Output Dataset. Lần sau dùng dataset đó làm input.

---

## Phần 7 — Monitoring Training

`python
# Theo dõi loss realtime trong notebook
import subprocess, threading

log_lines = []
def stream_logs(proc):
    for line in proc.stdout:
        print(line, end="", flush=True)
        log_lines.append(line)

proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True)
t = threading.Thread(target=stream_logs, args=(proc,))
t.start()

# Kiểm tra VRAM usage
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.free",
                "--format=csv"])
`

Output mẫu mong đợi:
`
[MASC+UPipe] MASC surgery done: vocab 16384 → k=4096
[UPipe] Patched layers.0.attention | H=16, chunk=4, U=4 stages
[UPipe] Patched layers.1.attention | H=16, chunk=4, U=4 stages
... (24 layers)
[train] Resumed from step 0 (c2i_L_256.pt)
[train] step 0/15000  loss=8.3241
[train] step 20/15000  loss=7.1832
[train] step 40/15000  loss=6.5421
...
`

---

## Phần 8 — Troubleshooting

| Lỗi | Nguyên nhân | Giải pháp |
|---|---|---|
| CUDA out of memory | VRAM thiếu | Giảm local_batch_size xuống 4; tăng grad_accum lên 32 |
| ModuleNotFoundError: backbones | PYTHONPATH chưa set | Thêm sys.path.insert(0, "/kaggle/working/MASC") |
| KeyError: 'tok_embeddings' | Tên attribute sai | Kiểm tra model.named_modules() hoặc print(model) |
| FileNotFoundError: codes/ | Sai đường dẫn dataset | Kiểm tra /kaggle/input/imagenet-tokens/codes/ tồn tại |
| RuntimeError: Timeout | Session 12h hết | Resume từ checkpoint cuối cùng |
| UPipe warning: 
o layers wrapped | Tên attention khác | Thêm tên vào ttn_submodule_names=("attention", "self_attn", "attn", ...) |
