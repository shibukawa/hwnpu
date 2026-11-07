#!/usr/bin/env python3
"""GPTQ for Anima's DiT (tools/amd/iron_anima.py's q4nx and q8 block projections): the codes
picked against the DiT's own activations, for iron_anima.py --q4-codes.

Rounding each weight to its nearest code (iron_q4.quantize_fmt) keeps the weight's error small,
not the projection's output error; the 8-step distilled DiT carries q4nx's rounding into the
image (latent cosine 0.78 against q8's 0.96, design.md 16.81). GPTQ (q4_gptq.gptq, the same
as the language models') chooses the codes with H = E[x x^T] of each projection's input, in
passes of a few blocks, the earlier blocks already quantized. The inputs are the DiT's at each
step of the reference (fp32) trajectories of a few prompts: the calibration is teacher forced,
so a pass sees the noisy latents the sampler gives the full-precision model.

Measures, on prompts and seeds of their own, the velocity's error against fp32 at the
reference's latents (relative RMS over every step) and the final latents' cosine of a free
run (as iron_anima's DitFormats docstring reports). Run in the reference virtual environment
(torch, diffusers 0.42+), on the CPU:

  python tools/amd/anima_gptq.py ~/models/Anima OUT --mlp q4 --attn q4
  python tools/amd/iron_anima.py ~/models/Anima EXPORT --dit-mlp q4 --dit-attn q4 --q4-codes OUT/codes

OUT/codes holds iron_q4.load_codes' layout (index.json and KEY.npy, keyed by
iron_q4.weight_key of the checkpoint's weight); OUT/result.json the measurements.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ref"))
import iron_q4  # noqa: E402
import q4_gptq  # noqa: E402

CALIB = [
    "masterpiece, best quality, 1girl, solo, smile, long hair, blue eyes, cherry blossoms, outdoors",
    "best quality, 1boy, short hair, school uniform, classroom, window, sunlight, looking at viewer",
    "scenery, no humans, mountain, lake, reflection, sunset, clouds, wide shot",
    "1girl, witch hat, holding staff, night, full moon, forest, glowing, magic circle",
    "still life, apples on a wooden table, kitchen, morning light, detailed",
    "2girls, cafe, coffee cups, chatting, indoors, warm colors, upper body",
    "city street, rain, neon signs, night, umbrella, reflections on wet asphalt",
    "1other, robot, mecha, sky, clouds, dynamic pose, from below, sword",
]
EVAL = [
    "masterpiece, best quality, score_7, safe, 1girl, solo, smile, long hair, blue eyes, cherry blossoms, outdoors",
    "a cat sleeping on a windowsill, potted plants, afternoon, soft light",
    "1boy, armor, castle, banner, standing, holding shield, overcast sky",
]


def targets(tr, attn: str, mlp: str):
    """The DiT blocks' projections iron_anima quantizes, (block, name, Linear, fmt), by block.
    The adaLN modulations, embeddings and final layer stay bf16 there and here."""
    out = []
    for b, blk in enumerate(tr.transformer_blocks):
        mine = []
        for n in ("attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
                  "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0"):
            if attn != "bf16":
                mine.append((f"transformer_blocks.{b}.{n}", blk.get_submodule(n), attn))
        for n in ("ff.net.0.proj", "ff.net.2"):
            if mlp != "bf16":
                mine.append((f"transformer_blocks.{b}.{n}", blk.get_submodule(n), mlp))
        out.append(mine)
    return out


def main(argv=None) -> int:
    import ref_anima
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="the Anima directory (ref_anima.py's layout)")
    ap.add_argument("out", type=Path, help="writes OUT/codes (with --method gptq) and OUT/result.json")
    ap.add_argument("--dit", default="anima-turbo-v1.1.safetensors")
    ap.add_argument("--method", choices=("gptq", "rtn", "codes"), default="gptq",
                    help="codes: evaluate the codes in --codes (an earlier run's OUT/codes)")
    ap.add_argument("--codes", type=Path)
    ap.add_argument("--attn", choices=("q4", "q8", "bf16"), default="q4")
    ap.add_argument("--mlp", choices=("q4", "q8", "bf16"), default="q4")
    ap.add_argument("--size", type=int, default=512, help="the calibration and evaluation images' width and height")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--calib-prompts", type=int, default=len(CALIB))
    ap.add_argument("--blocks-a-pass", type=int, default=7)
    ap.add_argument("--damp", type=float, default=0.1, help="GPTQ's damping, times H's mean diagonal")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    t0 = time.time()
    tok, t5, qwen, tc, tr, _vae = ref_anima.load(a.root, a.dit, torch.float32)
    del _vae
    W = H = a.size
    sig = ref_anima.sigmas(a.steps, a.shift)
    pad = torch.zeros(1, 1, H, W)

    def cond_of(prompt):
        q = tok([prompt], return_tensors="pt")
        hidden = qwen(input_ids=q.input_ids, attention_mask=q.attention_mask).last_hidden_state
        t = t5([prompt], return_tensors="pt")
        return tc(source_hidden_states=hidden, target_input_ids=t.input_ids,
                  target_attention_mask=t.attention_mask, source_attention_mask=q.attention_mask)

    def velocity(x, i, cond):
        return tr(hidden_states=x, timestep=torch.tensor([sig[i]], dtype=torch.float32),
                  encoder_hidden_states=cond, padding_mask=pad, return_dict=False)[0].float()

    def trajectory(cond, noise):
        """The latents each step reads, the velocities and the final latents."""
        x, xs, vs = noise.clone(), [], []
        for i in range(a.steps):
            v = velocity(x, i, cond)
            xs.append(x)
            vs.append(v)
            x = x + (sig[i + 1] - sig[i]) * v
        return xs, vs, x

    def noise_of(seed):
        g = torch.Generator().manual_seed(seed)
        return torch.randn((1, 16, 1, H // 8, W // 8), generator=g, dtype=torch.float32)

    conds = [cond_of(p) for p in CALIB[: a.calib_prompts]]
    econds = [cond_of(p) for p in EVAL]
    del qwen, tc
    refs = [trajectory(c, noise_of(1000 + j)) for j, c in enumerate(econds)]
    calib = []
    if a.method == "gptq":
        for j, c in enumerate(conds):
            xs, _, _ = trajectory(c, noise_of(j))
            calib += [(c, i, x) for i, x in enumerate(xs)]
    print(f"references: {len(refs)} eval and {len(conds)} calibration trajectories at {W}x{H}, "
          f"{time.time() - t0:.0f} s", flush=True)

    blocks = targets(tr, a.attn, a.mlp)
    codes_dir = a.out / "codes"
    index = {}

    def put(name, lin, fmt, q):
        codes_dir.mkdir(parents=True, exist_ok=True)
        k = q4_gptq.key(lin.weight.detach().numpy())
        np.save(codes_dir / f"{k}.npy", q)
        index[k] = dict(name=name, fmt=fmt, shape=list(lin.weight.shape))

    if a.method == "codes":
        n = iron_q4.load_codes(a.codes)
        print(f"{a.codes}: {n} weights", flush=True)
    if a.method in ("rtn", "codes"):
        for mine in blocks:
            for _, lin, fmt in mine:
                w = lin.weight.detach().numpy()
                q4_gptq.set_weight(lin, iron_q4.dequantize_fmt(iron_q4.quantize_fmt(w, fmt), *w.shape, fmt))
        if a.method == "codes":
            print(f"codes used: {iron_q4.CODES_USED}", flush=True)
    else:
        groups = [list(range(i, min(i + a.blocks_a_pass, len(blocks)))) for i in range(0, len(blocks), a.blocks_a_pass)]
        for grp in groups:
            lins = [x for b in grp for x in blocks[b]]
            Hs = {id(lin): [None, 0] for _, lin, _ in lins}
            same = {}

            def hook(mod, inp, _out):
                x = inp[0]
                if id(x) not in same:  # q, k and v read one tensor; held, so its id stays its own
                    xs = x.detach().reshape(-1, x.shape[-1]).double()
                    same[id(x)] = (x, xs.T @ xs, xs.shape[0])
                _, xtx, n = same[id(x)]
                h = Hs[id(mod)]
                h[0] = xtx.clone() if h[0] is None else h[0].add_(xtx)
                h[1] += n

            hooks = [lin.register_forward_hook(hook) for _, lin, _ in lins]
            t = time.time()
            for c, i, x in calib:
                same.clear()
                velocity(x, i, c)
            same.clear()
            for h in hooks:
                h.remove()
            tcal = time.time() - t
            t = time.time()
            for name, lin, fmt in lins:
                Hm, n = Hs[id(lin)]
                w = lin.weight.detach().numpy().copy()
                c, s, mn = q4_gptq.gptq(w, (Hm / n).numpy(), fmt, damp=a.damp)
                q = q4_gptq.blocks_of(c, s, mn, fmt)
                put(name, lin, fmt, q)
                q4_gptq.set_weight(lin, iron_q4.dequantize_fmt(q, *w.shape, fmt))
            print(f"blocks {grp}: {len(lins)} projections, calibration {tcal:.0f} s, GPTQ {time.time() - t:.0f} s",
                  flush=True)
        if index:
            (codes_dir / "index.json").write_text(json.dumps(index, indent=1))

    vel, cos = [], []
    for c, (xs, vs, lat) in zip(econds, refs):
        num = den = 0.0
        for i, (x, v) in enumerate(zip(xs, vs)):
            d = velocity(x, i, c) - v
            num += float((d * d).sum())
            den += float((v * v).sum())
        vel.append((num / den) ** 0.5)
        _, _, mine = trajectory(c, xs[0])
        cos.append(float(torch.nn.functional.cosine_similarity(mine.reshape(-1), lat.reshape(-1), dim=0)))
    out = dict(method=a.method, attn=a.attn, mlp=a.mlp, size=a.size, steps=a.steps, damp=a.damp,
               calib_prompts=len(conds), velocity_rel_rms=vel, latent_cosine=cos)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "result.json").write_text(json.dumps(out, indent=1))
    for p, v, c in zip(EVAL, vel, cos):
        print(f"{p[:40]}...: velocity rel RMS {v:.4f}, latent cosine {c:.4f}")
    print(f"mean: velocity rel RMS {np.mean(vel):.4f}, latent cosine {np.mean(cos):.4f}, {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
