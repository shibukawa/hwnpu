#!/usr/bin/env python3
"""Chroma1's T5-XXL encoder and FLUX VAE decoder on the CPU in float32, a part at a time: the
references tools/amd/iron_chroma_parts.py's graphs are checked against (``--ref``).

  t5   a prompt's 512 rows through transformers' T5Block, one block at a time in float32 (the
       whole encoder in float32 is 19 GB), with the tokenizer's attention mask as ChromaPipeline
       passes it: OUT.npz with ids, mask, the embedding rows (x), the relative bias of block 0
       and the encoder's output (y)
  vae  an image encoded by the VAE (its latent mean), to the DiT's latents ((z - shift) * scale),
       decoded again by diffusers' AutoencoderKL in float32: OUT.npz with the latents, every
       stage's output grid (dense, channel last) and the image (3, H, W) in [-1, 1], and each
       GroupNorm's input mean over std per group (how far the group means sit from zero)

  python tools/ref/ref_chroma_parts.py t5 ~/models/Chroma1-Flash OUT.npz --prompt "..."
  python tools/ref/ref_chroma_parts.py vae ~/models/Chroma1-Flash OUT.npz --image IMG.png --size 512
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def t5(a) -> None:
    from safetensors import safe_open
    from transformers import T5Config, T5TokenizerFast
    from transformers.models.t5.modeling_t5 import T5Block, T5LayerNorm

    root = a.root
    cfg = T5Config.from_pretrained(root / "text_encoder")
    cfg._attn_implementation = "eager"
    tok = T5TokenizerFast.from_pretrained(root / "tokenizer")
    t = tok([a.prompt], padding="max_length", max_length=512, truncation=True, return_tensors="pt")
    ids, mask = t.input_ids, t.attention_mask
    index = json.loads((root / "text_encoder" / "model.safetensors.index.json").read_text())["weight_map"]
    files = {f: safe_open(root / "text_encoder" / f, "pt") for f in sorted(set(index.values()))}

    def get(name):
        return files[index[name]].get_tensor(name).float()

    x = get("shared.weight")[ids[0]][None]  # (1, 512, 4096)
    rec = dict(ids=ids[0].numpy().astype(np.int32), mask=mask[0].numpy().astype(np.int32), x=x[0].numpy())
    dt = getattr(torch, a.dtype)
    ext = ((1.0 - mask[:, None, None, :].float()) * torch.finfo(dt).min).to(dt)
    h, bias = x.to(dt), None
    with torch.no_grad():
        for i in range(a.layers or cfg.num_layers):
            b = T5Block(cfg, has_relative_attention_bias=i == 0, layer_idx=i).eval()
            p = f"encoder.block.{i}."
            b.load_state_dict({k: get(p + k) for k in b.state_dict()})
            b.to(dt)
            h, bias, _ = b(h, ext, bias)
            if i == 0:
                rec["bias"] = b.layer[0].SelfAttention.compute_bias(512, 512)[0].float().numpy()
            print(f"t5 block {i}: |h| max {h.abs().max():.1f}", flush=True)
            del b
        n = T5LayerNorm(cfg.d_model, eps=cfg.layer_norm_epsilon)
        n.weight.data = get("encoder.final_layer_norm.weight")
        rec["y"] = n.to(dt)(h)[0].float().numpy()
    np.savez(a.out, **rec)
    print(f"{a.out}: {int(mask.sum())} tokens")


def vae(a) -> None:
    from diffusers import AutoencoderKL
    from PIL import Image

    root = a.root
    cfg = json.loads((root / "vae" / "config.json").read_text()) if (root / "vae" / "config.json").exists() else None
    v = AutoencoderKL.from_pretrained(root / "vae" if cfg else a.vae_config, torch_dtype=torch.float32).eval()
    if cfg is None:
        cfg = dict(v.config)
    img = Image.open(a.image).convert("RGB").resize((a.size, a.size), Image.LANCZOS)
    x = torch.from_numpy(np.asarray(img, np.float32) / 127.5 - 1).permute(2, 0, 1)[None]
    rec, ratios = {}, []

    def gn_hook(mod, args):
        t = args[0][0].double()  # (C, H, W)
        G = mod.num_groups
        g = t.reshape(G, -1)
        m, s = g.mean(1), g.std(1)
        ratios.append(float((m.abs() / s).max()))

    def keep(name):
        def hook(mod, args, out):
            rec[name] = out[0].permute(1, 2, 0).reshape(-1, out.shape[1]).numpy()
        return hook

    with torch.no_grad():
        z = v.encode(x).latent_dist.mean
        lat = (z - v.config.shift_factor) * v.config.scaling_factor
        d = v.decoder
        for m in d.modules():
            if isinstance(m, torch.nn.GroupNorm):
                m.register_forward_pre_hook(gn_hook)
        d.mid_block.register_forward_hook(keep("mid"))
        for i, u in enumerate(d.up_blocks):
            u.register_forward_hook(keep(f"up{i}"))
        y = v.decode(lat / v.config.scaling_factor + v.config.shift_factor).sample
    rec.update(latents=lat[0].numpy(), image=y[0].numpy(), gn_ratio=np.asarray(ratios))
    np.savez(a.out, **rec)
    print(f"{a.out}: latents {tuple(lat.shape)}, group |mean| / std at most {max(ratios):.2f} "
          f"(median of the norms' maxima {np.median(ratios):.2f})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=("t5", "vae"))
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--prompt", default="A red fox sitting in a snowy forest at dawn, soft light, a wooden cabin behind it")
    ap.add_argument("--image", type=Path)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"), help="t5: the blocks' dtype")
    ap.add_argument("--layers", type=int, default=0, help="t5: the first N blocks only, then the final norm (a test)")
    ap.add_argument("--vae-config", type=Path, help="a vae directory with config.json, when ROOT/vae has none")
    a = ap.parse_args()
    torch.set_num_threads(8)
    t5(a) if a.part == "t5" else vae(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
