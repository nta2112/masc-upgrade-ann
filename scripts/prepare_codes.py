#!/usr/bin/env python3
"""Download and convert ImageNet LlamaGen token codes from Hugging Face into NPY shards.

Dataset source: Efficient-Large-Model/imagenet-llamagen-cache (~7.2 GB tar)
Structure inside tar: mar_cache_vq/<synset>/<id>.JPEG.npz
Inside each .npz:
  - indices: int64 (256,) token indices in range [0, 16383]
  - indices_flip: int64 (256,) flipped augmentation
"""

import os
import io
import sys
import time
import tarfile
import argparse
from pathlib import Path
import numpy as np

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from masc.imagenet_synsets import SYNSET_TO_ID
except Exception:
    import urllib.request, json
    u = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"
    d = json.loads(urllib.request.urlopen(u).read().decode())
    SYNSET_TO_ID = {d[str(i)][0]: i for i in range(1000)}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare ImageNet token codes for MASC training.")
    parser.add_argument("--wds-dir", default="/kaggle/working/wds_cache",
                        help="Directory where HF tar file is stored.")
    parser.add_argument("--codes-dir", default="/kaggle/working/codes",
                        help="Directory to save output .npy shards and labels.npy.")
    parser.add_argument("--shard-size", type=int, default=512,
                        help="Number of images per shard file (default: 512).")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit max images converted (None = all ~1.28M images).")
    return parser.parse_args()


def main():
    args = parse_args()
    codes_dir = args.codes_dir
    wds_dir = args.wds_dir
    shard_size = args.shard_size

    os.makedirs(wds_dir, exist_ok=True)
    os.makedirs(codes_dir, exist_ok=True)

    # 1. Check if valid NPY shards already exist
    existing_npy = sorted(Path(codes_dir).glob("shard_*.npy"))
    labels_path = os.path.join(codes_dir, "labels.npy")
    if existing_npy and os.path.exists(labels_path):
        try:
            labels = np.load(labels_path)
            if len(labels) > 0 and len(existing_npy) > 0:
                print(f"[prepare_codes] {len(existing_npy)} NPY shards hop le da co ({len(labels):,} labels). Bo qua convert!")
                return
        except Exception:
            pass

    # 2. Check / download tar from Hugging Face
    existing_tars = sorted(Path(wds_dir).glob("**/*.tar"))
    if existing_tars:
        print(f"[prepare_codes] {len(existing_tars)} tar files da co san tai {wds_dir}, bo qua download.")
    else:
        print("[prepare_codes] Dang tai imagenet-llamagen-cache tu HuggingFace (~7.2 GB)...")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="Efficient-Large-Model/imagenet-llamagen-cache",
            repo_type="dataset",
            local_dir=wds_dir,
            ignore_patterns=["*.json", "*.yaml", "README*"],
        )
        existing_tars = sorted(Path(wds_dir).glob("**/*.tar"))
        if not existing_tars:
            raise FileNotFoundError(f"Khong tim thay file tar nao trong {wds_dir} sau khi tai!")
        print(f"[prepare_codes] Da tai xong {len(existing_tars)} tar files.")

    # 3. Stream tar and convert npz to npy shards
    tar_path = existing_tars[0]
    file_size_gb = os.path.getsize(tar_path) / 1e9
    print(f"[prepare_codes] Bat dau convert: {tar_path.name} ({file_size_gb:.2f} GB)...")

    t0 = time.time()
    shard_codes = []
    all_labels = []
    shard_idx = 0
    total_imgs = 0

    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf:
            if not member.name.endswith(".npz"):
                continue

            parts = member.name.split("/")
            if len(parts) < 2:
                continue

            synset = parts[-2]
            label = SYNSET_TO_ID.get(synset, 0)

            f = tf.extractfile(member)
            if f is None:
                continue

            try:
                npz = np.load(io.BytesIO(f.read()))
                codes = npz["indices"].flatten().astype(np.int32)
            except Exception:
                continue

            if len(codes) != 256:
                continue

            shard_codes.append(codes)
            all_labels.append(label)
            total_imgs += 1

            if len(shard_codes) >= shard_size:
                shard_out = os.path.join(codes_dir, f"shard_{shard_idx:05d}.npy")
                np.save(shard_out, np.array(shard_codes, dtype=np.int32))
                shard_codes = []
                shard_idx += 1
                if shard_idx % 50 == 0:
                    elapsed = (time.time() - t0) / 60
                    speed = total_imgs / max(1, time.time() - t0)
                    print(f"  Shard {shard_idx:4d} | {total_imgs:7,d} anh | {elapsed:.1f} phut | {speed:.0f} img/s", flush=True)

            if args.max_images and total_imgs >= args.max_images:
                print(f"[prepare_codes] Da dat gioi han max_images={args.max_images:,}.")
                break

    # Save leftover images in last shard
    if shard_codes:
        shard_out = os.path.join(codes_dir, f"shard_{shard_idx:05d}.npy")
        np.save(shard_out, np.array(shard_codes, dtype=np.int32))
        shard_idx += 1

    np.save(labels_path, np.array(all_labels, dtype=np.int32))
    elapsed_total = (time.time() - t0) / 60
    print(f"[prepare_codes] Hoan thanh: {total_imgs:,} anh -> {shard_idx} shards ({elapsed_total:.1f} phut).")

    # 4. Verify output files
    npy_files = sorted(Path(codes_dir).glob("shard_*.npy"))
    if not npy_files:
        raise RuntimeError(f"Loi: 0 shard nao duoc tao ra trong {codes_dir}!")

    labels = np.load(labels_path)
    sample = np.load(npy_files[0])
    print(f"\n[prepare_codes] KET QUA XAC THUC:")
    print(f"  - Tong so shards: {len(npy_files)} files")
    print(f"  - Tong so nhan  : {len(labels):,} labels")
    print(f"  - Shape shard mau: {sample.shape} (images x 256 tokens)")
    print(f"  - Token ID range : [{sample.min()}, {sample.max()}] (ky vong: 0 - 16383)")
    print("[prepare_codes] SAN SANG CHO CELL TIEP THEO!")


if __name__ == "__main__":
    main()
