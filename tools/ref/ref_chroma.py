#!/usr/bin/env python3
"""Chroma1 (lodestones/Chroma1-Flash, Chroma1-HD) on the CPU with diffusers' ChromaPipeline: the
reference hwnpu's NPU pipeline (tools/amd/iron_chroma.py) is checked against.

One part loaded at a time (T5-XXL in float32 is 19 GB, the DiT 36 GB): the prompt's T5 rows
(512, masked past the prompt and one pad token, as the pipeline masks them), then the pipeline on
those embeddings, every step's latents kept through its callback and every DiT call's input and
output through a hook, then the VAE.

Writes OUT.npz (token ids, T5 rows and mask, noise, each DiT call's timestep, input and output,
each step's latents, the final latents, the decoded image) and OUT.png:

  python tools/ref/ref_chroma.py ~/models/Chroma1-Flash OUT --size 512 --steps 8 --heun --prompt "..."
"""

import argparse
import gc
from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--prompt", default="A red fox sitting in a snowy forest at dawn, soft light")
    ap.add_argument("--negative", default="")
    ap.add_argument("--guidance", type=float, default=1.0, help="classifier-free guidance (above 1: the DiT twice)")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--heun", action="store_true", help="FlowMatchHeunDiscreteScheduler (Flash's 8 steps) instead of Euler")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    ap.add_argument("--embeds", type=Path, help="an OUT.npz of an earlier run: take its T5 rows")
    ap.add_argument("--mask", default="keys", choices=("keys", "diffusers"),
                    help="the text's mask: keys (the pads past the prompt and one pad get no weight, as Chroma was "
                         "trained) or diffusers (ChromaPipeline's float 0/1 mask, which SDPA adds to the scores: "
                         "+1 for the prompt's pairs, the pads still attended)")
    a = ap.parse_args()
    from diffusers import AutoencoderKL, ChromaPipeline, ChromaTransformer2DModel, FlowMatchEulerDiscreteScheduler
    from transformers import T5EncoderModel, T5TokenizerFast

    dtype = getattr(torch, a.dtype)

    rec = {}
    tok = T5TokenizerFast.from_pretrained(a.root / "tokenizer")

    def t5_ids(p):
        t = tok([p], padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        return t.input_ids, t.attention_mask

    cfg = a.guidance > 1
    todo = [("", a.prompt)] + ([("neg_", a.negative)] if cfg else [])
    if a.embeds:
        z = np.load(a.embeds)
        for k in ("ids", "t5", "t5_mask", "neg_ids", "neg_t5", "neg_t5_mask"):
            if k in z:
                rec[k] = z[k]
        todo = [(tag, p) for tag, p in todo if tag + "t5" not in rec]
    if todo:
        te = T5EncoderModel.from_pretrained(a.root / "text_encoder", torch_dtype=dtype).eval()
        for tag, p in todo:
            ids, m = t5_ids(p)
            rec[tag + "ids"] = ids[0].numpy().astype(np.int32)
            rec[tag + "t5"] = te(ids, attention_mask=m)[0][0].float().numpy()
            rec[tag + "t5_mask"] = m[0].numpy().astype(np.int32)
        del te
        gc.collect()

    def embeds(tag):
        e = torch.from_numpy(rec[tag + "t5"])[None].to(dtype)
        n = int(rec[tag + "t5_mask"].sum())
        mask = (torch.arange(512) <= n).to(dtype)[None]  # the prompt and one pad token
        return e, mask

    # after T5: its SDPA's float mask holds the relative position bias, which keyed() would drop
    if a.mask == "keys":
        # ChromaPipeline passes its mask as float (0/1), its blocks multiply it into a (query, key)
        # outer product and SDPA adds a float mask to the scores. Make it what it means: each key the
        # prompt's rows and one pad, the rest -inf (a query's own row does not matter)
        import torch.nn.functional as TF

        sdpa = TF.scaled_dot_product_attention

        def keyed(*args, attn_mask=None, **kw):
            q = args[0] if args else kw["query"]
            if attn_mask is not None and attn_mask.dtype != torch.bool:
                keys = attn_mask.amax(dim=-2, keepdim=True) > 0.5
                attn_mask = torch.zeros_like(keys, dtype=q.dtype).masked_fill(~keys, float("-inf"))
            return sdpa(*args, attn_mask=attn_mask, **kw)

        TF.scaled_dot_product_attention = keyed
    tr = ChromaTransformer2DModel.from_pretrained(a.root / "transformer", torch_dtype=dtype).eval()
    vae = AutoencoderKL.from_pretrained(a.root / "vae", torch_dtype=dtype).eval()
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(a.root / "scheduler")
    if a.heun:
        from diffusers import FlowMatchHeunDiscreteScheduler

        sched = FlowMatchHeunDiscreteScheduler.from_config(sched.config)
    pipe = ChromaPipeline(scheduler=sched, vae=vae, text_encoder=None, tokenizer=tok, transformer=tr)
    calls = {"t": [], "in": [], "out": []}

    def hook(mod, args, kwargs, out):
        calls["t"].append(float(kwargs["timestep"][0]))
        calls["in"].append(kwargs["hidden_states"][0].float().numpy())
        calls["out"].append(out[0][0].float().numpy())

    h = tr.register_forward_hook(hook, with_kwargs=True)
    steps = []

    def cb(p, i, t, kw):
        steps.append(kw["latents"][0].float().numpy())
        return kw

    e, m = embeds("")
    kw = dict(prompt_embeds=e, prompt_attention_mask=m)
    if cfg:
        ne, nm = embeds("neg_")
        kw.update(negative_prompt_embeds=ne, negative_prompt_attention_mask=nm)
    g = torch.Generator().manual_seed(a.seed)
    out = pipe(height=a.size, width=a.size, num_inference_steps=a.steps, guidance_scale=a.guidance, generator=g,
               output_type="latent", callback_on_step_end=cb, **kw).images
    h.remove()
    lat = out[0].float()
    rec["dit_t"], rec["dit_in"], rec["dit_out"] = np.array(calls["t"], np.float32), np.stack(calls["in"]), np.stack(calls["out"])
    rec["steps"] = np.stack(steps)
    rec["latents"] = lat.numpy()  # packed (tokens, 64)
    z = pipe._unpack_latents(out, a.size, a.size, pipe.vae_scale_factor)
    z = (z / vae.config.scaling_factor) + vae.config.shift_factor
    img = vae.decode(z.to(dtype), return_dict=False)[0].float()
    rec["decoded"] = img[0].numpy()
    pixels = ((img[0].clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()
    rec["image"] = pixels
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out.with_suffix(".npz"), **rec)
    from PIL import Image

    Image.fromarray(pixels).save(a.out.with_suffix(".png"))
    print(f"{a.out.with_suffix('.png')}: {a.size}x{a.size}, {a.steps} steps ({len(calls['t'])} DiT calls)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
