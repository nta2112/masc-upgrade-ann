import os
import sys
import torch
import torch.nn as nn


class LlamaGenAdapter:
    def __init__(self, model):
        self.m = model

    def get_token_embedding(self):
        return self.m.tok_embeddings

    def set_token_embedding(self, e):
        self.m.tok_embeddings = e

    def get_output_head(self):
        return self.m.output

    def set_output_head(self, h):
        self.m.output = h

    def forward_for_masc(self, coarse_in, class_labels):
        out = self.m(coarse_in, class_labels)
        logits = out[0] if isinstance(out, tuple) else out
        if logits.shape[1] > coarse_in.shape[1]:
            logits = logits[:, :coarse_in.shape[1], :]
        return logits


def build_backbone(cfg):
    try:
        from autoregressive.models.gpt import GPT_models
    except ImportError:
        for p in ['/kaggle/working/LlamaGen', './LlamaGen', '../LlamaGen']:
            if os.path.exists(p) and p not in sys.path:
                sys.path.insert(0, p)
        from autoregressive.models.gpt import GPT_models

    name_map = {
        "llamagen_b": "GPT-B",
        "llamagen_l": "GPT-L",
        "llamagen_xl": "GPT-XL",
        "llamagen_xxl": "GPT-XXL",
    }
    model_name = name_map.get(cfg["name"], "GPT-L")
    model = GPT_models[model_name](
        vocab_size=16384,
        block_size=256,
        num_classes=1000,
        cls_token_num=1,
        model_type="c2i",
        resid_dropout_p=0.1,
        ffn_dropout_p=0.1,
        drop_path_rate=0.0,
    )

    pt = cfg.get("pretrained")
    if pt and os.path.exists(pt):
        ckpt = torch.load(pt, map_location="cpu")
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded {pt} | missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"No pretrained found at {pt}, initialized randomly")

    adapter = LlamaGenAdapter(model)
    model.forward_for_masc = adapter.forward_for_masc
    return model, adapter
