#!/usr/bin/env python3
"""The weight formats of Qwen-Image-2.1's DiT on the CPU (new-model-checklist 5.1-5.3): which of
iron_q4's q8, q6 and q4nx the blocks' attention projections (to_q, to_k, to_v, to_out) and MLP
(gate_layer, proj, out) can take, and what GPTQ does for the 4-bit codes.

The DiT is written out here in torch (float32) from diffusers' transformer_qwenimage21.py and
run one block at a time, each block's weights read from the safetensors with pread (not mapped:
mapped pages count as resident) and held in bf16, the next block read ahead in a thread (a
block is 216M weights), so the process stays under 2.5 GB anonymous memory: the whole DiT in
float32 is 28 GB. Run it with MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_ARENA_MAX=2 (glibc kept
8 GB of a 4 GB peak without them); decode at 1024 x 1024 peaks at 14 GB (the VAE). The prompt's rows are modulated at t = 0 (causal_condition), so they run
through the blocks once a prompt (causal, as the pipeline's "extract" step) and each step runs
the image's rows only, attending to [the text's keys and values, its own] ("cached" steps,
iron_qwenimage.py's dit_text and dit_blocks). The text's keys and values are recomputed with
the quantized weights, as the NPU does.

A weight is rounded as the NPU holds it: iron_q4.quantize_fmt's blocks, dequantize_fmt, then
bf16 (QDequant writes bf16; the checkpoint's own weights are bf16). --bfp16 also rounds both
inputs of each projection to bfp16 (blocks of 8 along K with a shared exponent, 8-bit
mantissas: IRON's Precision.MEDIUM), as tools/ref/flux_quant.py. The rest (embeddings,
modulation, norms, attention, the residual stream) stays float32; the NPU's bf16 activations
and bfp16 QK^T are not emulated.

The measures (checklist 5.1, images), against the float32 reference runs of
tools/ref/ref_qwenimage21.py (REF.npz: text rows, noise, each step's DiT input and output, the
final latents):

  velocity  each step's relative RMS error of the DiT's output on the reference's own input
            x_i (teacher forced), and over all steps
  cosine    of the final latents of a free run: the 8 Euler steps with the quantized DiT from
            the reference's noise
  blocks    the relative error of the image rows after each block at step 0 (the reference's
            --blocks), where the error grows

Subcommands (run in the reference virtual environment, on the CPU):

  eval    --attn SRC --mlp SRC [--bfp16]: the measures; SRC is float, q8, q6, q4 (rounding,
          the blocks cached under OUT/w/rtn-FMT) or the name of a gptq run (OUT/w/NAME)
  traj    a float32 free run of a reference's prompt (or --text's, a ref_qwenimage21.py
          --text-only npz) from another seed (OUT/calib/NAME_sSEED.npz): more references for
          the free runs, and the step inputs GPTQ calibrates on
  gptq    --attn FMT --mlp FMT --gptq attn,mlp --name NAME: GPTQ (q4_gptq.gptq, damping 0.1 x
          the mean of H's diagonal) block by block, each projection's H = E[x x^T] over the
          calibration steps' image rows with every earlier projection already quantized
          (the earlier blocks and, within a block, q/k/v before to_out before gate/proj before
          out); the parts not in --gptq are rounded. Writes OUT/w/NAME (iron_q4's blocks) and
          OUT/w/NAME/codes (iron_q4.load_codes' layout, for an export)
  decode  the VAE of the free runs' final latents (OUT/latents): OUT/png and the PSNR against
          the reference's picture, into the results
  table   the results (OUT/results/*.json) as a table

eval keeps the final latents (OUT/latents/RUN.REF.npy, --no-save not) and writes
OUT/results/RUN.json; --free-only runs the free run again into an earlier run's json.

  PY=~/.cache/hwnpu/dev/ref-venv/bin/python
  $PY tools/ref/qwenimage_quant.py eval --attn float --mlp float          # the check: ~1e-6
  $PY tools/ref/qwenimage_quant.py eval --attn q8 --mlp q8 [--bfp16]
  $PY tools/ref/qwenimage_quant.py eval --attn q8 --mlp q4
  $PY tools/ref/qwenimage_quant.py eval --attn q8 --mlp q6 --free-only --refs p1_512_s3 p2_512_s1
  $PY tools/ref/qwenimage_quant.py traj --ref p2_512 --seed 1
  $PY tools/ref/ref_qwenimage21.py ~/models/Qwen-Image-2.1-Turbo OUT/text/c1 --text-only \
      --dtype bfloat16 --prompt "..."                                     # 17 GB, once
  $PY tools/ref/qwenimage_quant.py traj --ref p1_512 --text OUT/text/c1.npz --seed 10
  $PY tools/ref/qwenimage_quant.py gptq --attn q8 --mlp q4 --gptq mlp --name gptq-mlp4c \
      --calib c1_512_s10 c2_512_s10 c3_512_s10 c4_512_s10
  $PY tools/ref/qwenimage_quant.py eval --attn q8 --mlp gptq-mlp4c
  $PY tools/ref/qwenimage_quant.py decode
  $PY tools/ref/qwenimage_quant.py table

At 512 x 512 an eval is about 450 TFLOP in float32 (8 teacher-forced and 8 free steps of two
prompts), 20-30 minutes on a Ryzen AI 9 365; a GPTQ run of the MLP 35 minutes, of every
projection 60 (32K calibration rows).

Results (2026-11-12, OUT = ~/models/qwen21-test/quant): q8 matches the NPU (latent cosine
against the NPU's q8 run 0.9997 / 0.9996); the MLP at q6 keeps the pictures (velocity 0.010 /
0.008 against q8's 0.005 / 0.004, 1.2 GB less); q4nx changes details (MLP 0.031 / 0.027, both
parts 0.039 / 0.035); GPTQ lowers the teacher-forced velocity error by 13-15 % on prompts it
was not calibrated on but not the free runs' (worse on 2 of 3 extra seeds): not adopted.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "amd"))
import iron_q4  # noqa: E402  (numpy and ml_dtypes only)

HOME = Path.home()
ROOT = HOME / "models" / "Qwen-Image-2.1-Turbo"
REFS = HOME / "models" / "qwen21-test" / "ref"
OUT = HOME / "models" / "qwen21-test" / "quant"
NPU = HOME / "models" / "qwen21-test"  # npu_REF.lat: the NPU's final latents (q8, MEDIUM)

E, HEADS, D, M, LAYERS = 4096, 32, 128, 12288, 32
EPS = 1e-6
AXES = (16, 56, 56)
ATTN = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0")
MLP = ("img_mlp.gate_layer", "img_mlp.proj", "img_mlp.out")
SHAPES = {"attn.to_q": (E, E), "attn.to_k": (E, E), "attn.to_v": (E, E), "attn.to_out.0": (E, E),
          "img_mlp.gate_layer": (M, E), "img_mlp.proj": (M, E), "img_mlp.out": (E, M)}
FLOAT_SRC = ("float", "bf16")


def part_of(name: str) -> str:
    return "attn" if name in ATTN else "mlp"


def fmt_bytes(fmt: str, n: int, k: int) -> int:
    if fmt in FLOAT_SRC:
        return 2 * n * k
    return n * k // (iron_q4.M_TILE * iron_q4.K_TILE) * iron_q4.FORMATS[fmt]


# --- the checkpoint, one tensor at a time -------------------------------------------------------


class Checkpoint:
    """The DiT's tensors by name from the sharded safetensors, each read with pread (not
    memory-mapped: mapped pages count in the process's resident memory)."""

    def __init__(self, root: Path = ROOT):
        d = root / "transformer"
        index = json.loads((d / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
        self.fds, self.header = {}, {}
        for f in sorted(set(index.values())):
            fd = os.open(d / f, os.O_RDONLY)
            n = int.from_bytes(os.pread(fd, 8, 0), "little")
            h = json.loads(os.pread(fd, n, 8))
            h.pop("__metadata__", None)
            self.fds[f] = (fd, 8 + n)
            self.header.update({k: (f, v) for k, v in h.items()})

    def bf16(self, name: str) -> torch.Tensor:
        f, e = self.header[name]
        fd, base = self.fds[f]
        b0, b1 = e["data_offsets"]
        if e["dtype"] != "BF16":
            raise ValueError(f"{name}: {e['dtype']}")
        buf = bytearray(b1 - b0)
        view = memoryview(buf)
        got = 0
        while got < len(buf):
            got += os.preadv(fd, [view[got:]], base + b0 + got)
        return torch.frombuffer(buf, dtype=torch.bfloat16).reshape(e["shape"])

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.bf16(name).float()


def rss_anon_gb() -> float:
    for line in open("/proc/self/status"):
        if line.startswith("RssAnon:"):
            return int(line.split()[1]) / 2**20
    return 0.0


# --- the weights' sources: float, rounding (cached blocks) or GPTQ's blocks ----------------------


def bfp16(x: torch.Tensor) -> torch.Tensor:
    """x rounded as AIE2P's bfp16 (bfp16ebs8): blocks of 8 along the last axis share an 8-bit
    exponent, each value an 8-bit signed mantissa, rounded to nearest even (flux_quant.py)."""
    shape = x.shape
    b = x.reshape(-1, 8)
    m = b.abs().amax(-1, keepdim=True)
    e = torch.floor(torch.log2(torch.where(m > 0, m, torch.ones_like(m))))
    step = torch.exp2(e - 6)
    return (torch.clamp(torch.round(b / step), -128, 127) * step).reshape(shape)


class Source:
    """Where a part's weights come from: "float" (the checkpoint's bf16), a format rounded by
    iron_q4 (its blocks cached under OUT/w/rtn-FMT), or a gptq run's directory OUT/w/NAME."""

    def __init__(self, spec: str, part: str):
        self.spec = spec
        if spec in FLOAT_SRC:
            self.fmt, self.dir = "float", None
        elif spec in iron_q4.FORMATS:
            self.fmt, self.dir = spec, OUT / "w" / f"rtn-{spec}"
        else:
            self.dir = OUT / "w" / spec
            meta = json.loads((self.dir / "meta.json").read_text())
            self.fmt = meta[part]
        if self.dir is not None:
            self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, ck: Checkpoint, b: int, name: str) -> torch.Tensor:
        key = f"transformer_blocks.{b}.{name}.weight"
        if self.fmt == "float":
            return ck.bf16(key)
        n, k = SHAPES[name]
        p = self.dir / f"b{b:02d}.{name}.npy"
        if not p.exists():
            if self.spec not in iron_q4.FORMATS:
                raise FileNotFoundError(p)
            q = iron_q4.quantize_fmt(ck[key].numpy(), self.fmt)
            tmp = p.with_suffix(f".{os.getpid()}.tmp.npy")
            np.save(tmp, q)
            os.replace(tmp, p)
        else:
            q = np.load(p)
        w = torch.from_numpy(iron_q4.dequantize_fmt(q, n, k, self.fmt))
        return w.to(torch.bfloat16)  # QDequant's bf16


class Weights:
    """The DiT's weights: the blocks' through the sources (one block at a time, the next read
    ahead in a thread), the rest float (the checkpoint's bf16)."""

    def __init__(self, attn: str, mlp: str, bfp: bool):
        self.ck = Checkpoint()
        self.src = {"attn": Source(attn, "attn"), "mlp": Source(mlp, "mlp")}
        self.bfp = bfp
        g = lambda n: self.ck[n]  # noqa: E731
        self.img_in = g("img_in.weight")
        self.t1 = g("time_text_embed.timestep_embedder.linear_1.weight")
        self.t2 = g("time_text_embed.timestep_embedder.linear_2.weight")
        self.mod = g("modulation.1.weight")
        self.txt_norm = g("txt_in.text_norm.weight")
        self.txt_a = g("txt_in.in_layer.weight")
        self.txt_b = g("txt_in.out_layer.weight")
        self.no = g("norm_out.linear.weight")
        self.proj_out = g("proj_out.weight")
        self.pool = ThreadPoolExecutor(1)
        self.pending = {}

    def _load(self, b: int) -> dict:
        w = {}
        for name in ATTN + MLP:
            x = self.src[part_of(name)].get(self.ck, b, name)
            w[name] = bfp16(x.float()).to(torch.bfloat16) if self.bfp else x  # bfp16 values are bf16
        w["norm_q"] = self.ck[f"transformer_blocks.{b}.attn.norm_q.weight"]
        w["norm_k"] = self.ck[f"transformer_blocks.{b}.attn.norm_k.weight"]
        return w

    def block(self, b: int) -> dict:
        """Block b's weights; starts reading block b + 1 (b + 1 = 32 reads block 0)."""
        f = self.pending.pop(b, None)
        w = f.result() if f is not None else self._load(b)
        nxt = (b + 1) % LAYERS
        if nxt not in self.pending:
            self.pending[nxt] = self.pool.submit(self._load, nxt)
        return w

    def drop_ahead(self):
        for f in self.pending.values():
            f.result()
        self.pending.clear()


# --- the DiT ------------------------------------------------------------------------------------


def silu(x):
    return F.silu(x)


def temb_of(W: Weights, sigma: float) -> torch.Tensor:
    """The timestep's embedding of sigma (QwenImage21TimestepProjEmbeddings; the pipeline passes
    t / 1000 with t = 1000 sigma, time_proj multiplies by 1000)."""
    t = (torch.tensor([sigma], dtype=torch.float32) * 1000) / 1000
    half = 128
    freqs = torch.exp(-np.log(10000) * torch.arange(half, dtype=torch.float32) / half)
    args = (1000 * t)[:, None] * freqs[None]
    s = torch.cat([torch.cos(args), torch.sin(args)], -1)
    return (silu(s @ W.t1.T) @ W.t2.T)[0]


def mod_of(W: Weights, sigma: float):
    """(1 + scale1, tanh gate1, 1 + scale2, tanh gate2) and norm_out's 1 + scale of a timestep."""
    te = silu(temb_of(W, sigma))
    m = te @ W.mod.T
    s1, g1, s2, g2 = m.split(E)
    return (1 + s1, torch.tanh(g1), 1 + s2, torch.tanh(g2)), 1 + te @ W.no.T


def rope_of(pos: np.ndarray):
    """cos and sin (rows, 64) of positions (rows, 3) as QwenImage21Rope's float32 tables."""
    parts = []
    for a, d in enumerate(AXES):
        inv = 1.0 / torch.pow(10000, torch.arange(0, d, 2).to(torch.float32).div(d))
        parts.append(torch.outer(torch.from_numpy(pos[:, a]).float(), inv))
    ang = torch.cat(parts, 1)
    return torch.cos(ang), torch.sin(ang)


def head_perm() -> np.ndarray:
    """iron_qwenimage.head_perm: the out-feature order that puts the RoPE pair (2i, 2i + 1) at (i, i + 64)."""
    within = np.concatenate([np.arange(0, D, 2), np.arange(1, D, 2)])
    return (np.arange(HEADS)[:, None] * D + within[None, :]).reshape(-1)


def text_positions(n: int) -> np.ndarray:
    p = np.arange(n)
    return np.stack([p, p, p], 1)


def image_positions(gh: int, gw: int, text_len: int) -> np.ndarray:
    hi = np.arange(-(gh - gh // 2), gh // 2)
    wi = np.arange(-(gw - gw // 2), gw // 2)
    return np.stack([np.full(gh * gw, text_len), np.repeat(hi, gw), np.tile(wi, gh)], 1)


def rope(x, cs):
    """x (n, H, D) turned by pairs (2i, 2i + 1) (apply_rotary_emb_qwen, use_real=False)."""
    cos, sin = cs[0][:, None], cs[1][:, None]
    xr, xi = x[..., 0::2], x[..., 1::2]
    return torch.stack([xr * cos - xi * sin, xr * sin + xi * cos], -1).flatten(2)


def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def ln(x):
    return F.layer_norm(x, (x.shape[-1],), eps=EPS)


class Seg:
    """Rows run through the blocks together: a prompt's text (causal, its keys and values
    kept) or an image's at a step (attending to its prompt's text keys and values)."""

    def __init__(self, x, mod, cs, text: bool, prompt: int, calib: bool = False):
        self.x, self.mod, self.cs, self.text, self.prompt, self.calib = x, mod, cs, text, prompt, calib


def plain_linear(bfp: bool):
    def lin(w, name, x, segs=None, rows=None):
        if bfp:
            x = bfp16(x)
        return x @ w[name].float().T
    return lin


def run_block(w: dict, lin, segs: list, kv: dict, chunk: int = 4096):
    """One block over segs (their x updated in place). kv[prompt] = (k, v) of the text at this
    block: written by a text seg, read by the image segs (a text seg comes first)."""
    sizes = [s.x.shape[0] for s in segs]
    x = torch.cat([s.x for s in segs])
    m1 = torch.cat([s.mod[0].expand(n, E) for s, n in zip(segs, sizes)]) if len(segs) > 1 else segs[0].mod[0]
    h = ln(x) * m1
    del m1
    q, k, v = (lin(w, n, h, segs, sizes) for n in ATTN[:3])
    del h
    outs = []
    r = 0
    for s, n in zip(segs, sizes):
        qs = rope(rms(q[r : r + n].view(n, HEADS, D), w["norm_q"]), s.cs)
        ks = rope(rms(k[r : r + n].view(n, HEADS, D), w["norm_k"]), s.cs)
        vs = v[r : r + n].view(n, HEADS, D)
        r += n
        if s.text:
            kv[s.prompt] = (ks, vs.clone())  # not a view of all the rows' v
            o = F.scaled_dot_product_attention(qs.transpose(0, 1), ks.transpose(0, 1), vs.transpose(0, 1),
                                               is_causal=True)
        else:
            tk, tv = kv[s.prompt]
            o = F.scaled_dot_product_attention(qs.transpose(0, 1), torch.cat([tk, ks]).transpose(0, 1),
                                               torch.cat([tv, vs]).transpose(0, 1))
        outs.append(o.transpose(0, 1).reshape(n, E))
    del q, k, v
    o = torch.cat(outs)
    del outs
    a = lin(w, "attn.to_out.0", o, segs, sizes)
    del o
    r = 0
    for s, n in zip(segs, sizes):
        s.x = s.x + s.mod[1] * a[r : r + n]
        r += n
    del a
    x = torch.cat([s.x for s in segs])
    m2 = torch.cat([s.mod[2].expand(n, E) for s, n in zip(segs, sizes)]) if len(segs) > 1 else segs[0].mod[2]
    h = ln(x) * m2
    del x, m2
    if getattr(lin, "whole", False):  # GPTQ: gate and proj on H of h, then out on H of f, a chunk at a time
        H = lin.hessian(lambda c0, c1: h[c0:c1], segs, sizes, ("img_mlp.gate_layer", "img_mlp.proj"))
        lin.prepare("img_mlp.gate_layer", H)
        lin.prepare("img_mlp.proj", H)
        del H

        def f_rows(c0, c1):
            return silu(lin(w, "img_mlp.gate_layer", h[c0:c1])) * lin(w, "img_mlp.proj", h[c0:c1])

        H = lin.hessian(f_rows, segs, sizes, ("img_mlp.out",))
        lin.prepare("img_mlp.out", H)
        del H
    y = torch.empty_like(h)
    for c in range(0, h.shape[0], chunk):
        hc = h[c : c + chunk]
        f = silu(lin(w, "img_mlp.gate_layer", hc)) * lin(w, "img_mlp.proj", hc)
        y[c : c + chunk] = lin(w, "img_mlp.out", f)
        del f
    del h
    r = 0
    for s, n in zip(segs, sizes):
        s.x = s.x + s.mod[3] * y[r : r + n]
        r += n


class Prompt:
    """A reference run's prompt: its text rows after txt_in, its RoPE, and the image's."""

    def __init__(self, W: Weights, ref: dict, idx: int):
        text = torch.from_numpy(ref["text"]).float()
        h = rms(text, W.txt_norm + 1)  # QwenImage21ZeroCenterRMSNorm
        self.txt = F.gelu(h @ W.txt_a.T, approximate="tanh") @ W.txt_b.T
        T = text.shape[0]
        _, gh, gw = ref["noise"].shape
        self.gh, self.gw, self.n = gh, gw, gh * gw
        self.cs_text = rope_of(text_positions(T))
        self.cs_img = rope_of(image_positions(gh, gw, T))
        self.mod0, _ = mod_of(W, 0.0)
        self.idx = idx


def dit_in(W: Weights, lat_rows: torch.Tensor) -> torch.Tensor:
    return lat_rows @ W.img_in.T


def dit_out(W: Weights, x: torch.Tensor, out_scale: torch.Tensor) -> torch.Tensor:
    return (ln(x) * out_scale) @ W.proj_out.T


def teacher(W: Weights, lin, prompts: list, refs: list, kvs: dict, block_err: dict):
    """Each step's velocity on the reference's own inputs, one prompt at a time (all its steps
    together); fills kvs[prompt][block] with the text's keys and values. Returns per prompt the
    (steps, n, 64) velocities."""
    res = []
    for P, ref in zip(prompts, refs):
        sig = ref["sigmas"]
        steps = len(ref["dit_in"])
        segs = [Seg(P.txt.clone(), P.mod0, P.cs_text, True, P.idx)]
        outs = []
        for i in range(steps):
            mod, osc = mod_of(W, float(sig[i]))
            outs.append(osc)
            segs.append(Seg(dit_in(W, torch.from_numpy(ref["dit_in"][i]).float()), mod, P.cs_img, False, P.idx))
        kvs[P.idx] = []
        errs = []
        t = time.time()
        for b in range(LAYERS):
            w = W.block(b)
            kv = {}
            run_block(w, lin, segs, kv)
            kvs[P.idx].append(kv[P.idx])
            if b % 8 == 7:
                print(f"    block {b}: {rss_anon_gb():.1f} GB anon", flush=True)
            if "blocks" in ref:
                r = torch.from_numpy(np.array(ref["blocks"][b][-P.n :]))
                errs.append(float((segs[1].x - r).norm() / r.norm()))
        block_err[P.idx] = errs
        v = torch.stack([dit_out(W, s.x, o) for s, o in zip(segs[1:], outs)])
        res.append(v)
        print(f"  teacher p{P.idx}: {time.time() - t:.0f} s, {rss_anon_gb():.1f} GB anon", flush=True)
    return res


def free_run(W: Weights, lin, prompts: list, noises: list, sigmas: list, kvs: dict, keep: bool = False):
    """The Euler steps from the noises with the DiT, all prompts together each step; the text's
    keys and values from kvs. Returns the final latent rows (n, 64) per prompt, and with keep
    each step's inputs and velocities."""
    xs = [n.clone() for n in noises]
    hist = [([], []) for _ in prompts]
    steps = len(sigmas[0]) - 1
    t = time.time()
    for i in range(steps):
        segs, outs = [], []
        for P, x, sig in zip(prompts, xs, sigmas):
            mod, osc = mod_of(W, float(sig[i]))
            segs.append(Seg(dit_in(W, x), mod, P.cs_img, False, P.idx))
            outs.append(osc)
        for b in range(LAYERS):
            w = W.block(b)
            run_block(w, lin, segs, {P.idx: kvs[P.idx][b] for P in prompts})
        for j, (s, o, sig) in enumerate(zip(segs, outs, sigmas)):
            v = dit_out(W, s.x, o)
            if keep:
                hist[j][0].append(xs[j].numpy().copy())
                hist[j][1].append(v.numpy().copy())
            xs[j] = xs[j] + (float(sig[i + 1]) - float(sig[i])) * v
        print(f"  free step {i}: {time.time() - t:.0f} s, {rss_anon_gb():.1f} GB anon", flush=True)
    return xs, hist


def text_kv(W: Weights, lin, prompts: list, kvs: dict):
    """The text's keys and values of every block (the text rows alone)."""
    segs = [Seg(P.txt.clone(), P.mod0, P.cs_text, True, P.idx) for P in prompts]
    for P in prompts:
        kvs[P.idx] = []
    for b in range(LAYERS):
        w = W.block(b)
        for s in segs:
            kv = {}
            run_block(w, lin, [s], kv)
            kvs[s.prompt].append(kv[s.prompt])


def load_ref(name: str) -> dict:
    p = REFS / f"{name}.npz" if not (OUT / "calib" / f"{name}.npz").exists() else OUT / "calib" / f"{name}.npz"
    z = np.load(p)
    keys = ["text", "noise", "sigmas", "dit_in", "dit_out", "latents"]
    ref = {k: z[k] for k in keys}
    if "blocks" in z.files:  # (32, rows, 4096): read from a copy memory-mapped, 0.55 GB
        bp = OUT / "refcache" / f"{name}.blocks.npy"
        if not bp.exists():
            bp.parent.mkdir(parents=True, exist_ok=True)
            np.save(bp.with_suffix(".tmp.npy"), z["blocks"])
            os.replace(bp.with_suffix(".tmp.npy"), bp)
        ref["blocks"] = np.load(bp, mmap_mode="r")
    return ref


def cos(a, b):
    a, b = np.ravel(a).astype(np.float64), np.ravel(b).astype(np.float64)
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def bytes_of(attn_fmt: str, mlp_fmt: str) -> dict:
    a = sum(fmt_bytes(attn_fmt, *SHAPES[n]) for n in ATTN) * LAYERS
    m = sum(fmt_bytes(mlp_fmt, *SHAPES[n]) for n in MLP) * LAYERS
    return {"attn": a, "mlp": m, "total": a + m}


def cmd_eval(a) -> int:
    torch.set_grad_enabled(False)
    t0 = time.time()
    W = Weights(a.attn, a.mlp, a.bfp16)
    lin = plain_linear(a.bfp16)
    refs = [load_ref(r) for r in a.refs]
    prompts = [Prompt(W, r, i) for i, r in enumerate(refs)]
    kvs, block_err = {}, {}
    path = OUT / "results" / f"{run_name(a)}.json"
    if a.free_only:  # the free run alone (into an earlier run's result if there is one)
        res = json.loads(path.read_text()) if path.exists() else {
            "attn": a.attn, "mlp": a.mlp, "attn_fmt": W.src["attn"].fmt, "mlp_fmt": W.src["mlp"].fmt,
            "bfp16": a.bfp16, "bytes": bytes_of(W.src["attn"].fmt, W.src["mlp"].fmt),
            "refs": {name: {} for name in a.refs}}
        text_kv(W, lin, prompts, kvs)
    else:
        res = None
        vels = teacher(W, lin, prompts, refs, kvs, block_err)
    if res is None:
        res = {"attn": a.attn, "mlp": a.mlp, "attn_fmt": W.src["attn"].fmt, "mlp_fmt": W.src["mlp"].fmt,
               "bfp16": a.bfp16, "bytes": bytes_of(W.src["attn"].fmt, W.src["mlp"].fmt), "refs": {}}
        for name, ref, v in zip(a.refs, refs, vels):
            r = torch.from_numpy(ref["dit_out"])
            d = v - r
            per = [float(d[i].norm() / r[i].norm()) for i in range(len(r))]
            res["refs"][name] = {"velocity": per, "velocity_all": float(d.norm() / r.norm())}
        for P, name in zip(prompts, a.refs):
            res["refs"][name]["blocks_step0"] = block_err[P.idx]
    if not a.no_free:
        noises = [torch.from_numpy(r["noise"]).reshape(64, -1).T.contiguous() for r in refs]
        xs, _ = free_run(W, lin, prompts, noises, [r["sigmas"] for r in refs], kvs)
        for P, name, ref, x in zip(prompts, a.refs, refs, xs):
            lat = x.T.reshape(64, P.gh, P.gw).numpy()
            res["refs"][name]["latent_cosine"] = cos(lat, ref["latents"])
            npu = NPU / f"npu_{name}.lat"
            if npu.exists():
                res["refs"][name]["cosine_vs_npu_q8"] = cos(lat, np.fromfile(npu, np.float32))
            if not a.no_save:  # the final latents, for the VAE (decode)
                (OUT / "latents").mkdir(parents=True, exist_ok=True)
                np.save(OUT / "latents" / f"{run_name(a)}.{name}.npy", lat)
    W.drop_ahead()
    res["seconds_free_only" if a.free_only else "seconds"] = time.time() - t0
    (OUT / "results").mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1))
    print_result(path.stem, res)
    return 0


def run_name(a) -> str:
    refs = "+".join(a.refs)
    return f"attn-{a.attn}.mlp-{a.mlp}{'.bfp16' if a.bfp16 else ''}.{refs}"


def print_result(name: str, res: dict):
    print(f"{name}: attn {res['attn_fmt']} mlp {res['mlp_fmt']} bfp16 {res['bfp16']}, "
          f"{res['bytes']['total'] / 1e9:.2f} GB")
    for r, e in res["refs"].items():
        steps = " ".join(f"{x:.4f}" for x in e.get("velocity", []))
        lc = e.get("latent_cosine")
        line = f"  {r}: velocity all {e['velocity_all']:.4f} [{steps}]" if "velocity" in e else f"  {r}: free run"
        if lc is not None:
            line += f", latent cosine {lc:.4f}"
        if "psnr" in e:
            line += f", PSNR {e['psnr']:.1f} dB"
        if "cosine_vs_npu_q8" in e:
            line += f" (vs NPU {e['cosine_vs_npu_q8']:.4f})"
        if e.get("blocks_step0"):
            line += f", block 31 at step 0 {e['blocks_step0'][-1]:.4f}"
        print(line)


def cmd_traj(a) -> int:
    """A float32 free run of ref's prompt (or --text's, an OUT.npz of ref_qwenimage21.py --text-only,
    at ref's size and sigmas) from seed's noise (torch.randn, as ref_qwenimage21.py)."""
    torch.set_grad_enabled(False)
    W = Weights("float", "float", False)
    lin = plain_linear(False)
    ref = load_ref(a.ref)
    name = a.ref
    if a.text:
        ref["text"] = np.load(a.text)["text"].astype(np.float32)
        name = f"{Path(a.text).stem}_{a.ref.rsplit('_', 1)[-1]}"
    P = Prompt(W, ref, 0)
    kvs = {}
    text_kv(W, lin, [P], kvs)
    g = torch.Generator().manual_seed(a.seed)
    noise = torch.randn((64, P.gh, P.gw), generator=g)
    xs, hist = free_run(W, lin, [P], [noise.reshape(64, -1).T.contiguous()], [ref["sigmas"]], kvs, keep=True)
    W.drop_ahead()
    (OUT / "calib").mkdir(parents=True, exist_ok=True)
    out = OUT / "calib" / f"{name}_s{a.seed}.npz"
    np.savez(out, text=ref["text"], noise=noise.numpy(), sigmas=ref["sigmas"], dit_in=np.stack(hist[0][0]),
             dit_out=np.stack(hist[0][1]), latents=xs[0].T.reshape(64, P.gh, P.gw).numpy())
    print(out)
    return 0


# --- GPTQ ---------------------------------------------------------------------------------------


def cmd_gptq(a) -> int:
    import q4_gptq

    torch.set_grad_enabled(False)
    gparts = set(a.gptq.split(","))
    fmts = {"attn": a.attn, "mlp": a.mlp}
    d = OUT / "w" / a.name
    (d / "codes").mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps({"attn": a.attn, "mlp": a.mlp, "gptq": sorted(gparts),
                                              "damp": a.damp, "calib": a.calib}))
    W = Weights("float", "float", False)
    ck = W.ck
    refs = [load_ref(r) for r in a.calib]
    prompts = [Prompt(W, r, i) for i, r in enumerate(refs)]
    segs = []
    for P, ref in zip(prompts, refs):
        segs.append(Seg(P.txt.clone(), P.mod0, P.cs_text, True, P.idx))
        for i in range(len(ref["dit_in"])):
            mod, _ = mod_of(W, float(ref["sigmas"][i]))
            segs.append(Seg(dit_in(W, torch.from_numpy(ref["dit_in"][i]).float()), mod, P.cs_img, False, P.idx,
                            calib=True))
    print(f"calibration: {sum(s.x.shape[0] for s in segs if s.calib)} image rows of {len(a.calib)} runs", flush=True)
    index = {}
    if (d / "codes" / "index.json").exists():
        index = json.loads((d / "codes" / "index.json").read_text())
    t0 = time.time()
    for b in range(LAYERS):
        tb = time.time()
        wq = {"norm_q": ck[f"transformer_blocks.{b}.attn.norm_q.weight"],
              "norm_k": ck[f"transformer_blocks.{b}.attn.norm_k.weight"]}
        Hcache = {}

        def hessian(x, segs_, sizes):
            """E[x x^T] over the calibration's image rows of x, or of rows(c0, c1) if x is that."""
            rows = x if callable(x) else (lambda c0, c1: x[c0:c1])
            H = None
            count, r0 = 0, 0
            for s0, sg in zip(sizes, segs_):
                if sg.calib:
                    for c in range(r0, r0 + s0, 2048):
                        xc = rows(c, min(c + 2048, r0 + s0))
                        if H is None:
                            H = torch.zeros(xc.shape[1], xc.shape[1], dtype=torch.float64)
                        H += (xc.T @ xc).double()
                    count += s0
                r0 += s0
            return (H / count).numpy()

        def prepare(name, H=None):
            """name's quantized weight into wq: an earlier run's blocks, GPTQ's on H, or rounding."""
            fmt = fmts[part_of(name)]
            p = d / f"b{b:02d}.{name}.npy"
            key = f"transformer_blocks.{b}.{name}.weight"
            wf = ck[key].numpy()
            n, k = wf.shape
            if p.exists():
                q = np.load(p)
            elif H is not None:
                c_, s_, mn_ = q4_gptq.gptq(wf, H, fmt, damp=a.damp)
                q = q4_gptq.blocks_of(c_, s_, mn_, fmt)
                ky = iron_q4.weight_key(wf)
                np.save(d / "codes" / f"{ky}.npy", q)
                index[ky] = {"name": key, "fmt": fmt, "shape": [n, k]}
                if name in ("attn.to_q", "attn.to_k"):
                    # iron_qwenimage.py quantizes these with their rows in IRON's RoPE order
                    # (head_perm): GPTQ's rows are independent, so the same codes, rows permuted
                    pm = head_perm()
                    kp = iron_q4.weight_key(wf[pm])
                    np.save(d / "codes" / f"{kp}.npy", q4_gptq.blocks_of(c_[pm], s_[pm], mn_[pm], fmt))
                    index[kp] = {"name": key + " (head_perm)", "fmt": fmt, "shape": [n, k]}
            else:
                q = iron_q4.quantize_fmt(wf, fmt)
            if not p.exists():
                np.save(p.with_suffix(".tmp.npy"), q)
                os.replace(p.with_suffix(".tmp.npy"), p)
            wq[name] = torch.from_numpy(iron_q4.dequantize_fmt(q, n, k, fmt)).to(torch.bfloat16).float()

        def lin(w, name, x, segs_=None, sizes=None):
            if name not in wq:
                H = None
                if part_of(name) in gparts and not (d / f"b{b:02d}.{name}.npy").exists():
                    if id(x) not in Hcache:  # q, k and v (gate and proj) read one x; held, so its id stays
                        Hcache.clear()
                        Hcache[id(x)] = (x, hessian(x, segs_, sizes))
                    H = Hcache[id(x)][1]
                prepare(name, H)
            return x @ wq[name].T

        def lin_hessian(x, segs_, sizes, names):
            """H for names (None: they are rounded, or chosen by an earlier run)."""
            if "mlp" not in gparts or all((d / f"b{b:02d}.{n}.npy").exists() for n in names):
                return None
            return hessian(x, segs_, sizes)

        lin.hessian = lin_hessian
        lin.prepare = lambda name, H: (Hcache.clear(), prepare(name, H) if name not in wq else None)
        lin.whole = True
        run_block(wq, lin, segs, {})
        Hcache.clear()
        (d / "codes" / "index.json").write_text(json.dumps(index, indent=1))
        print(f"block {b}: {time.time() - tb:.0f} s (all {time.time() - t0:.0f} s)", flush=True)
    return 0


# --- the pictures -------------------------------------------------------------------------------


def cmd_decode(a) -> int:
    """The VAE (float32, diffusers) of each saved run's final latents (OUT/latents/RUN.REF.npy):
    OUT/png/RUN.REF.png and the PSNR of its RGB against the reference's picture, into the run's
    result json (refs.REF.psnr)."""
    from diffusers import AutoencoderKLQwenImage21
    from PIL import Image

    torch.set_grad_enabled(False)
    vae = AutoencoderKLQwenImage21.from_pretrained(ROOT / "vae", torch_dtype=torch.float32).eval()
    mean = torch.tensor(vae.config.latents_mean).view(64, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(64, 1, 1)
    (OUT / "png").mkdir(parents=True, exist_ok=True)

    def picture(lat):
        img = vae.decode((torch.from_numpy(lat) * std + mean)[None, :, None], return_dict=False)[0][:, :, 0].float()
        return ((img[0].clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()

    refs = {}

    def ref_picture(ref):
        """The reference's picture: its run's, or (traj's runs) its latents through the VAE."""
        if ref not in refs:
            p = REFS / f"{ref}.npz"
            if p.exists():
                refs[ref] = np.load(p)["image"]
            else:
                refs[ref] = picture(np.load(OUT / "calib" / f"{ref}.npz")["latents"])
                Image.fromarray(refs[ref], mode="RGBA").save(OUT / "png" / f"ref.{ref}.png")
        return refs[ref]

    for f in sorted((OUT / "latents").glob("*.npy")):
        run, ref = f.stem.rsplit(".", 1)
        res_p = OUT / "results" / f"{run}.json"
        png = OUT / "png" / f"{f.stem}.png"
        if not res_p.exists():
            continue
        res = json.loads(res_p.read_text())
        if "psnr" in res["refs"].get(ref, {}) and png.exists() and not a.force:
            continue
        pix = picture(np.load(f))
        Image.fromarray(pix, mode="RGBA").save(png)
        refimg = ref_picture(ref)
        mse = ((pix[..., :3].astype(np.float64) - refimg[..., :3]) ** 2).mean()
        res["refs"][ref]["psnr"] = float(10 * np.log10(255**2 / max(mse, 1e-12)))
        res_p.write_text(json.dumps(res, indent=1))
        print(f"{f.stem}: PSNR {res['refs'][ref]['psnr']:.1f} dB", flush=True)
    return 0


# --- the table ----------------------------------------------------------------------------------


def cmd_table(a) -> int:
    rows = []
    for p in sorted((OUT / "results").glob("*.json")):
        res = json.loads(p.read_text())
        rows.append((p.stem, res))
    for name, res in rows:
        print_result(name, res)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=0)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--attn", default="q8", help="float, q8, q6, q4 or a gptq run's name")
    e.add_argument("--mlp", default="q8")
    e.add_argument("--bfp16", action="store_true", help="the projections' inputs in bfp16 (Precision.MEDIUM)")
    e.add_argument("--refs", nargs="+", default=["p1_512", "p2_512"])
    e.add_argument("--no-free", action="store_true", help="the teacher-forced velocities only")
    e.add_argument("--free-only", action="store_true", help="the free run only, into an earlier run's result")
    e.add_argument("--no-save", action="store_true", help="do not keep the final latents (OUT/latents)")
    t = sub.add_parser("traj")
    t.add_argument("--ref", default="p1_512")
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--text", type=Path, help="another prompt's text rows (ref_qwenimage21.py --text-only)")
    g = sub.add_parser("gptq")
    g.add_argument("--attn", default="q8", choices=tuple(iron_q4.FORMATS))
    g.add_argument("--mlp", default="q4", choices=tuple(iron_q4.FORMATS))
    g.add_argument("--gptq", default="mlp", help="the parts GPTQ chooses the codes of: attn, mlp or attn,mlp")
    g.add_argument("--name", required=True)
    g.add_argument("--calib", nargs="+", default=["p1_512_s1", "p1_512_s2"])
    g.add_argument("--damp", type=float, default=0.1)
    dc = sub.add_parser("decode")
    dc.add_argument("--force", action="store_true")
    sub.add_parser("table")
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    return {"eval": cmd_eval, "traj": cmd_traj, "gptq": cmd_gptq, "decode": cmd_decode, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
