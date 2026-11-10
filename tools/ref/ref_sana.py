#!/usr/bin/env python3
"""Sana (NVlabs) on the CPU with diffusers' modules: the reference hwnpu's NPU
pipeline (cmd/hwnpu-image, tools/amd/iron_sana.py) is checked against.

Reads a diffusers repository of Efficient-Large-Model (Sana_600M_1024px_diffusers,
SANA1.5_1.6B_1024px_diffusers, ...): transformer/, vae/, scheduler/, and text_encoder/ and tokenizer/
(the same Gemma 2 2B in every one of them; --text names another repository's when this one has
none). The loop is the pipeline's, written out so that every intermediate is kept: flow
DPM-Solver++ (2nd order, the repository's scheduler) with classifier-free guidance, the DiT on
the prompt and on the empty negative prompt each step. Sana-Sprint is not supported (as in
hwnpu, design.md 16.148).

Writes OUT.npz (token ids, Gemma 2's selected hidden states and mask, the noises, each step's
timestep, DiT input and output, the final latents, the decoded image) and OUT.png. Run in the
reference virtual environment (torch, transformers, diffusers 0.42+):

  python tools/ref/ref_sana.py ~/models/Sana_600M_1024px_diffusers OUT --prompt "..." --steps 20
"""

import argparse
from pathlib import Path

import numpy as np
import torch

# diffusers' default (SanaPipeline)
CHI = [
    "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
    "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
    "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
    "User Prompt: ",
]


def encode(tok, gemma, prompt: str, instruction: bool, max_len: int = 300):
    """diffusers' encode_prompt: the prompt lowered and stripped, the instruction before it,
    padded to the instruction's tokens + max_len - 2, Gemma 2's last hidden state at the first
    token and the last max_len - 1. Returns (ids, hidden (max_len, 2304), mask (max_len,))."""
    prompt = prompt.lower().strip()
    if instruction:
        chi = "\n".join(CHI)
        n_chi = len(tok.encode(chi))
        prompt = chi + prompt
        total = n_chi + max_len - 2
    else:
        total = max_len
    t = tok([prompt], padding="max_length", max_length=total, truncation=True, add_special_tokens=True, return_tensors="pt")
    h = gemma(t.input_ids, attention_mask=t.attention_mask)[0]
    sel = [0] + list(range(-max_len + 1, 0))
    n = int(t.attention_mask.sum())
    return t.input_ids[0, :n].numpy().astype(np.int32), h[0, sel].float(), t.attention_mask[0, sel]


def dpm_timesteps(sched, steps: int):
    sched.set_timesteps(steps)
    return sched.timesteps, sched.sigmas


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--text", type=Path, help="the repository whose text_encoder/ and tokenizer/ to use (default: root)")
    ap.add_argument("--prompt", default="a cyberpunk cat with a neon sign that says \"Sana\"")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0, help="steps (default 20)")
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--negative", default="", help="Sana's negative prompt (default empty)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    ap.add_argument("--no-softcap", action="store_true", help="Gemma 2 without its attention logit softcapping (50)")
    ap.add_argument("--noise", type=Path, help="the noise as float32 .npy, (1, 32, H/32, W/32)")
    ap.add_argument("--stop", type=int, default=0, help="run only this many steps, then decode")
    a = ap.parse_args()
    from diffusers import AutoencoderDC, DPMSolverMultistepScheduler, SanaTransformer2DModel
    from transformers import AutoTokenizer, Gemma2Config, Gemma2Model

    dtype = getattr(torch, a.dtype)
    text = a.text or a.root
    tok = AutoTokenizer.from_pretrained(text / "tokenizer")
    tok.padding_side = "right"
    gc = Gemma2Config.from_pretrained(text / "text_encoder")
    if a.no_softcap:
        gc.attn_logit_softcapping = None
    gemma = Gemma2Model.from_pretrained(text / "text_encoder", config=gc, torch_dtype=dtype, attn_implementation="eager").eval()
    tr = SanaTransformer2DModel.from_pretrained(a.root / "transformer", torch_dtype=dtype).eval()
    vae = AutoencoderDC.from_pretrained(a.root / "vae", torch_dtype=dtype).eval()
    if tr.config.guidance_embeds:
        raise SystemExit("Sana-Sprint (guidance_embeds) is not supported")
    steps = a.steps or 20
    rec = {}
    ids, emb, mask = encode(tok, gemma, a.prompt, True)
    rec["ids"], rec["emb"], rec["mask"] = ids, emb.numpy(), mask.numpy().astype(np.int32)
    nids, nemb, nmask = encode(tok, gemma, a.negative, False)
    rec["neg_ids"], rec["neg_emb"], rec["neg_mask"] = nids, nemb.numpy(), nmask.numpy().astype(np.int32)
    del gemma
    W, H = a.width or a.size, a.height or a.size
    h, w = H // 32, W // 32
    g = torch.Generator().manual_seed(a.seed)
    nnoise = 1
    if a.noise is not None:
        noises = torch.from_numpy(np.load(a.noise)).float()
    else:
        noises = torch.stack([torch.randn((32, h, w), generator=g) for _ in range(nnoise)])
    rec["noise"] = noises.numpy()
    ctx = emb[None].to(dtype)
    cmask = mask[None]
    ins, outs, ts = [], [], []
    sched = DPMSolverMultistepScheduler.from_pretrained(a.root / "scheduler")
    timesteps, sig = dpm_timesteps(sched, steps)
    rec["timesteps"], rec["sigmas"] = timesteps.numpy().astype(np.float32), sig.numpy().astype(np.float32)
    nctx = torch.from_numpy(rec["neg_emb"])[None].to(dtype)
    nmask_t = torch.from_numpy(rec["neg_mask"])[None]
    x = noises[0][None]
    n = a.stop or steps
    for i, t in enumerate(timesteps[:n]):
        tt = t.expand(2).to(dtype)
        inp = torch.cat([x, x]).to(dtype)
        out = tr(inp, encoder_hidden_states=torch.cat([nctx, ctx]), encoder_attention_mask=torch.cat([nmask_t, cmask]),
                 timestep=tt, return_dict=False)[0].float()
        ins.append(x[0].numpy())
        outs.append(out.numpy())  # (2, 32, h, w): uncond, cond
        ts.append(float(t))
        v = out[0:1] + a.guidance * (out[1:2] - out[0:1])
        x = sched.step(v, t, x, return_dict=False)[0]
        print(f"step {i}: t {float(t):.2f}", flush=True)
    latents = x
    rec["dit_in"], rec["dit_out"], rec["t"] = np.stack(ins), np.stack(outs), np.array(ts, np.float32)
    rec["latents"] = latents[0].numpy()
    img = vae.decode((latents / vae.config.scaling_factor).to(dtype), return_dict=False)[0].float()
    rec["decoded"] = img[0].numpy()
    pixels = ((img[0].clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()
    rec["image"] = pixels
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out.with_suffix(".npz"), **rec)
    from PIL import Image

    Image.fromarray(pixels).save(a.out.with_suffix(".png"))
    print(f"{a.out.with_suffix('.png')}: {W}x{H}, {steps} steps, {len(ids)} tokens ({int(mask.sum())} kept)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
