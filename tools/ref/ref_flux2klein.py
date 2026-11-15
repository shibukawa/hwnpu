#!/usr/bin/env python3
"""FLUX.2 [klein] 4B (black-forest-labs/FLUX.2-klein-4B) on the CPU with diffusers'
Flux2KleinPipeline: the reference an NPU pipeline of it is checked against.

Text to image, and editing with reference images (--image, repeatable): the pipeline itself
runs, with hooks that keep what it passes between its parts:

  1. the prompt in Qwen3-4B's chat template (no thinking), padded to 512 tokens; the hidden
     states after layers 9, 18 and 27, side by side (512 rows of 7680);
  2. each reference through the VAE's encoder, 2 x 2 patches (128 channels at a 16th of the
     picture), normalized by the VAE's batch-norm statistics; its rows follow the picture's in
     the DiT's sequence, with the first position axis 10, 20, ... to tell them apart;
  3. the steps (4, no guidance: the checkpoint is distilled), flow-matching Euler on a linear
     grid shifted by the pipeline's empirical mu; only the picture's rows of the DiT's output
     are used;
  4. the VAE's decoder on the latents un-normalized and un-patched.

Writes OUT.npz (token ids, the attention mask, the text rows, the noise, the references'
latents, each step's sigma, timestep, DiT output and latents, the decoded image) and OUT.png.
Run in the reference virtual environment (torch, transformers 5.x, diffusers with Flux2Klein):

  python tools/ref/ref_flux2klein.py ~/models/FLUX.2-klein-4B OUT --size 512 --prompt "..."
  python tools/ref/ref_flux2klein.py ~/models/FLUX.2-klein-4B OUT --image in.png --prompt "make it night"
  python tools/ref/ref_flux2klein.py ~/models/FLUX.2-klein-4B OUT --encode in.png --size 512   # the VAE's encoder alone
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--prompt", default="A red fox sitting in a snowy forest at dawn, soft light")
    ap.add_argument("--image", type=Path, action="append", help="a reference image to edit or compose from (repeatable)")
    ap.add_argument("--size", type=int, default=0, help="the picture's side (default: 1024, or the first reference's size)")
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    ap.add_argument("--noise", type=Path, help="the noise as float32 .npy, (128, H/16, W/16)")
    ap.add_argument("--encode", type=Path, help="only the VAE's encoder on this picture (scaled to --size): OUT.npz "
                    "holds the picture (H, W, 3) in [-1, 1] and the latents' means (32, H/8, W/8)")
    a = ap.parse_args()

    from PIL import Image

    if a.encode:
        from diffusers import AutoencoderKLFlux2

        size = a.size or 1024
        img = np.asarray(Image.open(a.encode).convert("RGB").resize((a.width or size, a.height or size)), np.float32) / 127.5 - 1
        vae = AutoencoderKLFlux2.from_pretrained(a.root / "vae", torch_dtype=torch.float32)
        z = vae.encode(torch.from_numpy(img.transpose(2, 0, 1))[None]).latent_dist.mode()[0].numpy()
        a.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.out.with_suffix(".npz"), image=img, latents=z)
        print(f"{img.shape[1]}x{img.shape[0]}: latents {z.shape}, |max| {np.abs(z).max():.3f}")
        return 0

    from diffusers import Flux2KleinPipeline

    dtype = getattr(torch, a.dtype)
    pipe = Flux2KleinPipeline.from_pretrained(a.root, torch_dtype=dtype)
    pipe.set_progress_bar_config(disable=True)

    images = [Image.open(p).convert("RGB") for p in (a.image or [])]
    width, height = a.width or a.size, a.height or a.size
    if not images:
        width, height = width or 1024, height or 1024
    elif not width:
        # the pipeline's default: the first reference's size (under a megapixel, a multiple of 16)
        width = height = None

    kept = {"sigma": [], "timestep": [], "dit_out": [], "latents": [], "step_s": []}

    # the text rows and the token ids
    tok = pipe.tokenizer
    orig_tok_call = tok.__class__.__call__

    def tok_call(self, *args, **kw):
        res = orig_tok_call(self, *args, **kw)
        if "ids" not in kept and kw.get("padding") == "max_length":
            kept["ids"] = res["input_ids"][0].numpy().astype(np.int32)
            kept["mask"] = res["attention_mask"][0].numpy().astype(np.int32)
        return res

    tok.__class__.__call__ = tok_call

    # the DiT's input (the first call: the noise and the references' latents), output and timing
    def dit_pre(mod, args, kw):
        kept["t0"] = time.time()
        if "text" not in kept:
            kept["text"] = kw["encoder_hidden_states"][0].float().numpy()
            kept["txt_ids"] = kw["txt_ids"][0].numpy().astype(np.int32)
            kept["img_ids"] = kw["img_ids"][0].numpy().astype(np.int32)
            kept["dit_in"] = kw["hidden_states"][0].float().numpy()
        kept["timestep"].append(float(kw["timestep"][0]))

    def dit_post(mod, args, kw, out):
        kept["step_s"].append(time.time() - kept["t0"])
        kept["dit_out"].append(out[0][0].float().numpy())

    pipe.transformer.register_forward_pre_hook(dit_pre, with_kwargs=True)
    pipe.transformer.register_forward_hook(dit_post, with_kwargs=True)

    def on_step(p, i, t, kw):
        kept["sigma"].append(float(p.scheduler.sigmas[i]))
        kept["latents"].append(kw["latents"][0].float().numpy())
        return {}

    # the VAE's input (the latents un-normalized and un-patched, 32 channels at an 8th), before
    # its post_quant_conv
    def vae_pre(mod, args, kw):
        kept["vae_in"] = args[0][0].float().numpy()

    pipe.vae.post_quant_conv.register_forward_pre_hook(vae_pre, with_kwargs=True)

    latents = None
    if a.noise:
        latents = torch.from_numpy(np.load(a.noise).astype(np.float32))[None].to(dtype)
    t0 = time.time()
    res = pipe(prompt=a.prompt, image=images or None, width=width, height=height,
               num_inference_steps=a.steps, guidance_scale=1.0, latents=latents,
               generator=torch.Generator("cpu").manual_seed(a.seed),
               callback_on_step_end=on_step, callback_on_step_end_tensor_inputs=["latents"],
               output_type="np")
    total = time.time() - t0
    img = res.images[0]  # (H, W, 3) in 0..1
    h, w = img.shape[:2]
    rows = (h // 16) * (w // 16)
    out = {
        "ids": kept["ids"], "mask": kept["mask"], "text": kept["text"],
        "txt_ids": kept["txt_ids"], "img_ids": kept["img_ids"],
        # the noise as the pipeline packs it: (rows, 128); the references' rows follow in dit_in
        "noise": kept["dit_in"][:rows], "refs": kept["dit_in"][rows:],
        "sigma": np.array(kept["sigma"], np.float32), "timestep": np.array(kept["timestep"], np.float32),
        "dit_out": np.stack(kept["dit_out"]), "latents": np.stack(kept["latents"]),
        "vae_in": kept["vae_in"], "image": img.astype(np.float32),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out.with_suffix(".npz"), **out)
    Image.fromarray((img * 255).round().clip(0, 255).astype(np.uint8)).save(a.out.with_suffix(".png"))
    real = int(kept["mask"].sum())
    print(f"{w}x{h}, {rows} picture rows, {len(out['refs'])} reference rows, {real} prompt tokens of {len(kept['ids'])}")
    print("steps: " + ", ".join(f"{s:.1f}s" for s in kept["step_s"]) + f"; total {total:.1f}s")
    print("sigmas: " + ", ".join(f"{s:.4f}" for s in kept["sigma"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
