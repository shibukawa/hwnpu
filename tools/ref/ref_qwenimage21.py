#!/usr/bin/env python3
"""Qwen-Image-2.1 (Qwen/Qwen-Image-2.1, -Turbo) on the CPU with diffusers' modules: the
reference hwnpu's NPU pipeline (cmd/hwnpu-image, tools/amd/iron_qwenimage.py) is checked
against.

Text to image only (no condition images). The loop is QwenImage21Pipeline's, written out so
that every intermediate is kept, one part loaded at a time (the text encoder in float32 is
35 GB, the DiT 28 GB):

  1. the prompt in the pipeline's template, Qwen3-VL-8B's last decoder layer's output (before
     the final RMSNorm, as the pipeline's hook takes it), the system message's rows dropped;
  2. the steps, flow-matching Euler on the checkpoint's sigmas (Turbo's eight) or a linear grid
     shifted as the scheduler does: the DiT with its prefix K/V cache (the text's keys and
     values from the first step), classifier-free guidance only with --negative and
     --guidance > 1, as the pipeline's true_cfg_scale;
  3. the VAE (RGBA) on the latents un-normalized.

Writes OUT.npz (token ids, the text hidden states, the noise, each step's sigma, DiT input and
output, the final latents, the decoded image; with --blocks the hidden states after each DiT
block of the first step and the prefix's keys and values of each layer) and OUT.png. Run in the
reference virtual environment (torch, transformers 5.x, diffusers with QwenImage21):

  python tools/ref/ref_qwenimage21.py ~/models/Qwen-Image-2.1-Turbo OUT --size 512 --prompt "..."
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

SYS_PROMPT = "Comprehend and analyze the provided prompt."
TEMPLATE = (f"<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n"
            "<|im_start|>user\n{}<|im_end|>\n"
            "<|im_start|>assistant\n")


def drop_count(tok) -> int:
    """The pipeline's _drop_idx: the tokens of the system message (its chat template's rendering),
    which it drops."""
    return len(tok("<|im_start|>system\n" + SYS_PROMPT + "<|im_end|>\n").input_ids)


@torch.no_grad()
def encode(root: Path, prompts: list, dtype):
    """Each prompt's token ids (the whole template's), the rows kept (after the system message's)
    and their hidden states (rows, 4096), float32."""
    # the tokenizer alone: Qwen3VLProcessor's video processor wants torchvision, and the text is
    # tokenized the same
    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

    proc = AutoTokenizer.from_pretrained(root / "processor")
    drop = drop_count(proc)
    te = Qwen3VLForConditionalGeneration.from_pretrained(root / "text_encoder", dtype=dtype).eval()
    lm = te.model.language_model
    handle = lm.norm.register_forward_hook(lambda m, args, out: args[0])  # before the final norm
    res = []
    try:
        for p in prompts:
            p = p or " "  # Qwen has no bos: an empty prompt is a space
            t = proc(text=[TEMPLATE.format(p)], return_tensors="pt")
            out = te(input_ids=t.input_ids, attention_mask=t.attention_mask, output_hidden_states=True)
            h = out.hidden_states[-1][0].float()
            res.append((t.input_ids[0].numpy().astype(np.int32), h[drop:].numpy()))
    finally:
        handle.remove()
    del te, lm
    gc.collect()
    return res, drop


def sigmas_for(root: Path, steps: int, seq: int):
    """The pipeline's sigmas: model_index.json's sample_sigmas (Turbo) or a linear grid, through
    the scheduler (its shift, dynamic shift and terminal), with the terminal 0."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift

    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(root / "scheduler")
    mi = json.loads((root / "model_index.json").read_text())
    sig = mi.get("sample_sigmas")
    if steps or sig is None:
        sig = np.linspace(1.0, 1 / (steps or 40), steps or 40)
    cfg = sched.config
    mu = calculate_shift(seq, cfg.get("base_image_seq_len", 256), cfg.get("max_image_seq_len", 4096),
                         cfg.get("base_shift", 0.5), cfg.get("max_shift", 1.15))
    sched.set_timesteps(sigmas=list(sig), mu=mu)
    sched.set_begin_index(0)
    return sched


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--prompt", default="A red fox sitting in a snowy forest at dawn, soft light")
    ap.add_argument("--negative", default=None)
    ap.add_argument("--guidance", type=float, default=1.0, help="true_cfg_scale (CFG only above 1 with --negative)")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0, help="a linear grid of this many steps instead of the checkpoint's sigmas")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    ap.add_argument("--noise", type=Path, help="the noise as float32 .npy, (64, H/16, W/16)")
    ap.add_argument("--embeds", type=Path, help="an OUT.npz of an earlier run: take its text rows (skip the text encoder)")
    ap.add_argument("--blocks", action="store_true", help="keep each block's output of the first step and the prefix K/V")
    ap.add_argument("--stop", type=int, default=0, help="run only this many steps, then decode")
    ap.add_argument("--text-only", action="store_true", help="only the text rows (OUT.npz), for --embeds later")
    a = ap.parse_args()
    from diffusers import AutoencoderKLQwenImage21, QwenImage21Transformer2DModel
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

    dtype = getattr(torch, a.dtype)
    W, H = a.width or a.size, a.height or a.size
    W, H = W // 32 * 32, H // 32 * 32
    gh, gw = H // 16, W // 16
    n = gh * gw
    cfg = a.negative is not None and a.guidance > 1
    rec = {}
    if a.embeds:
        z = np.load(a.embeds)
        rec["ids"], rec["text"] = z["ids"], z["text"]
        if cfg:
            rec["neg_ids"], rec["neg_text"] = z["neg_ids"], z["neg_text"]
        rec["drop"] = z["drop"]
    else:
        enc, drop = encode(a.root, [a.prompt] + ([a.negative] if cfg else []), dtype)
        rec["ids"], rec["text"] = enc[0]
        if cfg:
            rec["neg_ids"], rec["neg_text"] = enc[1]
        rec["drop"] = np.int32(drop)
    print(f"text: {len(rec['ids'])} tokens, {len(rec['text'])} rows kept", flush=True)
    if a.text_only:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.out.with_suffix(".npz"), **rec)
        return 0

    tr = QwenImage21Transformer2DModel.from_pretrained(a.root / "transformer", torch_dtype=dtype).eval()
    g = torch.Generator().manual_seed(a.seed)
    noise = torch.from_numpy(np.load(a.noise)).float() if a.noise else torch.randn((64, gh, gw), generator=g)
    rec["noise"] = noise.numpy()
    x = noise.reshape(64, n).T[None].to(dtype)  # (1, n, 64): the pipeline's pack

    def conds(text):
        T = text.shape[0]
        emb = torch.from_numpy(text)[None].to(dtype)
        mask = torch.cat([torch.zeros(1, T, dtype=torch.bool), torch.ones(1, n // 4, dtype=torch.bool)], 1)
        return emb, mask

    c_emb, c_mask = conds(rec["text"])
    if cfg:
        u_emb, u_mask = conds(rec["neg_text"])
    shapes = [[(1, gh, gw)]]
    sched = sigmas_for(a.root, a.steps, n)
    rec["sigmas"] = sched.sigmas.numpy().astype(np.float32)
    caches = [QwenImage21KVCache(len(tr.transformer_blocks)) for _ in range(2 if cfg else 1)]

    if a.blocks:  # each block's output of the first step, and the K/V the cache keeps
        outs = []
        hooks = [b.register_forward_hook(lambda m, i, o: outs.append(o[0].float().numpy())) for b in tr.transformer_blocks]
    ins, vs = [], []
    steps = a.stop or len(sched.timesteps)
    for i, t in enumerate(sched.timesteps[:steps]):
        mode = "extract" if i == 0 else "cached"
        ts = (t / 1000).expand(1).to(dtype)
        v = tr(hidden_states=x, encoder_hidden_states=c_emb, timestep=ts, img_shapes=shapes, img_mask=c_mask,
               kv_cache=caches[0], kv_cache_mode=mode, return_dict=False)[0][:, -n:].float()
        if a.blocks and i == 0:
            for h in hooks:
                h.remove()
            rec["blocks"] = np.stack(outs)
            rec["prefix_k"] = np.stack([c.k[0].float().numpy() for c in caches[0].layer_caches])
            rec["prefix_v"] = np.stack([c.v[0].float().numpy() for c in caches[0].layer_caches])
        if cfg:
            u = tr(hidden_states=x, encoder_hidden_states=u_emb, timestep=ts, img_shapes=shapes, img_mask=u_mask,
                   kv_cache=caches[1], kv_cache_mode=mode, return_dict=False)[0][:, -n:].float()
            v = u + a.guidance * (v - u)
        ins.append(x[0].float().numpy())
        vs.append(v[0].numpy())
        x = sched.step(v.to(dtype), t, x, return_dict=False)[0]
        print(f"step {i}: sigma {float(sched.sigmas[i]):.4f}", flush=True)
    rec["dit_in"], rec["dit_out"] = np.stack(ins), np.stack(vs)
    lat = x[0].float().T.reshape(64, gh, gw)
    rec["latents"] = lat.numpy()
    del tr, caches
    gc.collect()

    vae = AutoencoderKLQwenImage21.from_pretrained(a.root / "vae", torch_dtype=dtype).eval()
    mean = torch.tensor(vae.config.latents_mean).view(64, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(64, 1, 1)
    z = (lat * std + mean)[None, :, None].to(dtype)
    img = vae.decode(z, return_dict=False)[0][:, :, 0].float()  # (1, 4, H, W), RGBA in [-1, 1]
    rec["decoded"] = img[0].numpy()
    pixels = ((img[0].clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()
    rec["image"] = pixels
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out.with_suffix(".npz"), **rec)
    from PIL import Image

    Image.fromarray(pixels, mode="RGBA").save(a.out.with_suffix(".png"))
    print(f"{a.out.with_suffix('.png')}: {W}x{H}, {steps} steps, {len(rec['ids'])} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
