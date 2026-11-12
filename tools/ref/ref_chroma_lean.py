#!/usr/bin/env python3
"""Chroma1 (lodestones/Chroma1-Flash, Chroma1-HD) on the CPU in float32 within a 16 GB Mac: the
reference of hwnpu's Core ML pipeline (internal/imagegen/chroma/mil).

tools/ref/ref_chroma.py loads the whole DiT (36 GB in float32) and T5-XXL (19 GB); this one reads
a block at a time from the checkpoint's shards: T5's blocks through transformers' T5Block (as
ref_chroma_parts.py), the DiT's through diffusers' ChromaTransformerBlock and
ChromaSingleTransformerBlock, built on the meta device and given their weights per call, with
ChromaTransformer2DModel.forward's order. The text's mask is the keys' (the prompt's rows and one
pad, as Chroma was trained, ref_chroma.py --mask keys); the rows are [image, text] as hwnpu orders
them (attention does not depend on the order).

The sampler is the pipelines' (FlowMatchEulerDiscreteScheduler's linear sigmas through the shift,
Heun as FlowMatchHeunDiscreteScheduler, DPM-Solver++ 2M on the data prediction as
internal/imagegen/anima.DPM2M), the guidance v = u + s (c - u) at the sigmas --cfg-sigmas picks.

Writes OUT.npz: the prompts' ids, T5 rows and key counts, the noise (also OUT.noise.f32, for
hwnpu-image generate --noise), each DiT call's sigma, input and output, the final latents, the
decoded image; with --dump the first DiT call's residual streams before every block and the
modulation rows (for checking the programs one at a time); with --stats the largest magnitudes
in every block (fp16's range). OUT.png is the picture.

  python tools/ref/ref_chroma_lean.py ROOT OUT --size 512 --steps 8 --sampler heun --prompt "..."
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch


class Shards:
    """The tensors of a directory's safetensors shards (model.safetensors.index.json) or file."""

    def __init__(self, d: Path, base: str):
        from safetensors import safe_open

        idx = d / f"{base}.safetensors.index.json"
        if idx.exists():
            wm = json.loads(idx.read_text())["weight_map"]
        else:
            f = safe_open(d / f"{base}.safetensors", "pt")
            wm = {k: f"{base}.safetensors" for k in f.keys()}
        self.map = wm
        self.files = {n: safe_open(d / n, "pt") for n in sorted(set(wm.values()))}

    def get(self, name: str) -> torch.Tensor:
        return self.files[self.map[name]].get_tensor(name).float()

    def prefixed(self, p: str) -> dict:
        return {k[len(p):]: self.get(k) for k in self.map if k.startswith(p)}


def t5_rows(root: Path, prompts, layers: int = 0, stats=None):
    """T5-XXL's rows of each prompt (512, 4096), its ids and its token count (</s> included); with
    stats (a dict) the largest magnitudes of each layer's parts over the prompt's rows."""
    from transformers import T5Config, T5TokenizerFast
    from transformers.models.t5.modeling_t5 import T5Block, T5LayerNorm

    cfg = T5Config.from_pretrained(root / "text_encoder")
    cfg._attn_implementation = "eager"
    tok = T5TokenizerFast.from_pretrained(root / "tokenizer")
    w = Shards(root / "text_encoder", "model")
    emb = w.get("shared.weight")
    hs, exts, out = [], [], []
    for p in prompts:
        t = tok([p], padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        ids, mask = t.input_ids, t.attention_mask
        hs.append(emb[ids[0]][None])
        exts.append((1.0 - mask[:, None, None, :].float()) * torch.finfo(torch.float32).min)
        out.append(dict(ids=ids[0].numpy().astype(np.int32), n=int(mask.sum())))
    del emb
    biases = [None] * len(prompts)
    t0 = time.time()
    for i in range(layers or cfg.num_layers):
        b = T5Block(cfg, has_relative_attention_bias=i == 0, layer_idx=i).eval()
        b.load_state_dict(w.prefixed(f"encoder.block.{i}."))
        hooks = []
        if stats is not None:
            def note(name):
                def hook(mod, args, out):
                    v = (out[0] if isinstance(out, tuple) else out)[:, :cur[0]]
                    stats[f"t5.{i}.{name}"] = max(stats.get(f"t5.{i}.{name}", 0.0), float(v.abs().max()))
                    if name == "wo":
                        a = args[0][:, :cur[0]]
                        stats[f"t5.{i}.act"] = max(stats.get(f"t5.{i}.act", 0.0), float(a.abs().max()))
                return hook
            ff = b.layer[1].DenseReluDense
            for name, m in (("o", b.layer[0].SelfAttention.o), ("wi0", ff.wi_0), ("wi1", ff.wi_1), ("wo", ff.wo)):
                hooks.append(m.register_forward_hook(note(name)))
        for k in range(len(prompts)):
            cur = [out[k]["n"]]
            hs[k], biases[k], _ = b(hs[k], exts[k], biases[k])
            if stats is not None:
                stats[f"t5.{i}.res"] = max(stats.get(f"t5.{i}.res", 0.0), float(hs[k][:, :cur[0]].abs().max()))
        for h in hooks:
            h.remove()
        del b
    n = T5LayerNorm(cfg.d_model, eps=cfg.layer_norm_epsilon)
    n.weight.data = w.get("encoder.final_layer_norm.weight")
    for k in range(len(prompts)):
        out[k]["t5"] = n(hs[k])[0].numpy()
    print(f"T5-XXL: {time.time() - t0:.0f} s, {[o['n'] for o in out]} tokens", flush=True)
    return out


class Dit:
    """ChromaTransformer2DModel a block at a time."""

    def __init__(self, root: Path, size: int, stats: bool):
        from diffusers.models.embeddings import FluxPosEmbed
        from diffusers.models.transformers.transformer_chroma import (
            ChromaApproximator, ChromaCombinedTimestepTextProjEmbeddings, ChromaSingleTransformerBlock,
            ChromaTransformerBlock)

        self.cfg = json.loads((root / "transformer" / "config.json").read_text())
        c = self.cfg
        self.w = Shards(root / "transformer", "diffusion_pytorch_model")
        E = c["num_attention_heads"] * c["attention_head_dim"]
        self.E, self.D, self.S = E, c["num_layers"], c["num_single_layers"]
        with torch.device("meta"):
            self.double = ChromaTransformerBlock(E, c["num_attention_heads"], c["attention_head_dim"])
            self.single = ChromaSingleTransformerBlock(E, c["num_attention_heads"], c["attention_head_dim"])
        self.tte = ChromaCombinedTimestepTextProjEmbeddings(c["approximator_num_channels"] // 4, 3 * self.S + 12 * self.D + 2)
        self.approx = ChromaApproximator(c["approximator_num_channels"], E, c["approximator_hidden_dim"], c["approximator_layers"])
        self.approx.load_state_dict(self.w.prefixed("distilled_guidance_layer."))
        self.approx.eval()
        lin = lambda n: (self.w.get(n + ".weight"), self.w.get(n + ".bias"))  # noqa: E731
        self.x_emb, self.ctx_emb, self.proj = lin("x_embedder"), lin("context_embedder"), lin("proj_out")
        gh = gw = size // 16
        self.N = gh * gw
        img = torch.zeros(gh, gw, 3)
        img[..., 1] = torch.arange(gh)[:, None]
        img[..., 2] = torch.arange(gw)[None, :]
        self.img_ids = img.reshape(-1, 3)
        self.pos = FluxPosEmbed(theta=10000, axes_dim=c["axes_dims_rope"])
        self.stats = {} if stats else None
        self.dump = None

    def mods(self, sigma: float) -> torch.Tensor:
        t = torch.tensor([sigma * 1000], dtype=torch.float32)
        return self.approx(self.tte(t))  # (1, 344, E)

    def note(self, key, x):
        if self.stats is not None:
            self.stats[key] = max(self.stats.get(key, 0.0), float(x.abs().max()))

    def __call__(self, x: np.ndarray, sigma: float, ctx: torch.Tensor, keys: int) -> np.ndarray:
        """The velocity of packed latents x (tokens, 64) at sigma, the text's rows ctx (T5's,
        (512, 4096)) with keys of them attended to."""
        E, N = self.E, self.N
        T = ctx.shape[0]
        hs = torch.nn.functional.linear(torch.from_numpy(x)[None], *self.x_emb)
        enc = torch.nn.functional.linear(ctx[None], *self.ctx_emb)
        pm = self.mods(sigma)
        txt_ids = torch.zeros(T, 3)
        rope = self.pos(torch.cat([txt_ids, self.img_ids], 0))
        mask = torch.zeros(1, T + N)
        mask[0, : keys] = 1  # the text's first keys rows, then every image row
        mask[0, T:] = 1
        dump = self.dump
        if dump is not None:
            dump["mods"] = pm[0].numpy()
            dump["x_emb"] = hs[0].numpy()
            dump["ctx_emb"] = enc[0].numpy()
        for i in range(self.D):
            p = f"transformer_blocks.{i}."
            self.double.load_state_dict(self.w.prefixed(p), assign=True)
            io, to = 3 * self.S + 6 * i, 3 * self.S + 6 * self.D + 6 * i
            temb = torch.cat([pm[:, io: io + 6], pm[:, to: to + 6]], 1)
            enc, hs = self.double(hs, enc, temb, rope, mask)
            self.note(f"d{i}.img", hs)
            self.note(f"d{i}.txt", enc[:, :keys])
            if dump is not None:
                dump[f"d{i}"] = torch.cat([hs, enc], 1)[0].numpy()  # [image, text] after block i
        h = torch.cat([enc, hs], 1)
        for i in range(self.S):
            self.single.load_state_dict(self.w.prefixed(f"single_transformer_blocks.{i}."), assign=True)
            h = self.single(h, pm[:, 3 * i: 3 * i + 3], rope, mask)
            self.note(f"s{i}.img", h[:, T:])
            self.note(f"s{i}.txt", h[:, :keys])
            if dump is not None:
                dump[f"s{i}"] = torch.cat([h[:, T:], h[:, :T]], 1)[0].numpy()
        h = h[:, T:]
        shift, scale = pm[:, -2], pm[:, -1]
        h = torch.nn.functional.layer_norm(h, (E,), eps=1e-6) * (1 + scale[:, None]) + shift[:, None]
        return torch.nn.functional.linear(h, *self.proj)[0].numpy()


def sigmas(steps: int, shift: float) -> list:
    s = []
    for i in range(steps):
        t = 1 + (1 / steps - 1) * i / max(1, steps - 1)
        if shift not in (0, 1):
            t = shift * t / (1 + (shift - 1) * t)
        s.append(t)
    return s + [0.0]


def dpm2m(x, v, prev, s, s_next, h_last):
    """internal/imagegen/anima.DPM2M: DPM-Solver++ 2M on the data prediction x0 = x - s v."""
    x0 = x - s * v
    if s_next == 0:
        return x0, (x0, math.inf)
    lam = lambda a: -math.inf if a >= 1 else math.log((1 - a) / a)  # noqa: E731
    h = lam(s_next) - lam(s)
    eh = s_next * (1 - s) / ((1 - s_next) * s)  # e^-h
    d = x0
    if prev is not None and not math.isinf(h_last) and not math.isinf(h):
        k = h / (2 * h_last)
        d = (1 + k) * x0 - k * prev[0]
    return (s_next / s) * x + (1 - s_next) * (1 - eh) * d, (x0, h)


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--prompt", default="A red fox sitting in a snowy forest at dawn, soft light")
    ap.add_argument("--negative", default="")
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--cfg-sigmas", default="", help="LO,HI: guide only at these sigmas")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--shift", type=float, default=1.0)
    ap.add_argument("--sampler", default="heun", choices=("heun", "euler", "dpm2m"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=Path, help="float32 packed noise (tokens x 64) instead of the seed's")
    ap.add_argument("--calls", type=int, default=0, help="stop after N DiT calls (a check of the first)")
    ap.add_argument("--dump", action="store_true", help="the first call's residual streams before every block")
    ap.add_argument("--dump-call", type=int, default=0, help="--dump of this DiT call (0: the first)")
    ap.add_argument("--stats", action="store_true", help="the largest magnitudes in every block")
    ap.add_argument("--no-vae", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = a.guidance > 1
    rec = {}
    t5stats = {} if a.stats else None
    texts = t5_rows(a.root, [a.prompt] + ([a.negative] if cfg else []), stats=t5stats)
    if t5stats:
        print("T5 largest magnitudes:", ", ".join(f"{k} {v:.3g}" for k, v in t5stats.items()), flush=True)
    conds = []
    for tag, t in zip(("", "neg_"), texts):
        rec[tag + "ids"], rec[tag + "t5"], rec[tag + "keys"] = t["ids"], t["t5"], np.int32(min(t["n"] + 1, 512))
        conds.append((torch.from_numpy(t["t5"]), min(t["n"] + 1, 512)))
    # after T5 (its SDPA's float mask holds the relative bias): the blocks pass the text's 0/1 mask
    # as a float outer product, which SDPA would add to the scores; make it the keys' mask
    import torch.nn.functional as TF

    sdpa = TF.scaled_dot_product_attention

    def keyed(*args, attn_mask=None, **kw):
        q = args[0] if args else kw["query"]
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            keys = attn_mask.amax(dim=-2, keepdim=True) > 0.5
            attn_mask = torch.zeros_like(keys, dtype=q.dtype).masked_fill(~keys, float("-inf"))
        return sdpa(*args, attn_mask=attn_mask, **kw)

    TF.scaled_dot_product_attention = keyed
    dit = Dit(a.root, a.size, a.stats)
    N = dit.N
    if a.noise:
        x = np.fromfile(a.noise, np.float32).reshape(N, 64)
    else:
        g = torch.Generator().manual_seed(a.seed)
        x = torch.randn(N, 64, generator=g).numpy()
    rec["noise"] = x.copy()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    x.astype(np.float32).tofile(a.out.with_suffix(".noise.f32"))
    lo, hi = (float(v) for v in a.cfg_sigmas.split(",")) if a.cfg_sigmas else (0.0, 0.0)
    calls = {"sigma": [], "in": [], "out": []}
    dumped = {}

    def vel(x, s):
        if len(calls["sigma"]) == a.dump_call and a.dump:
            dit.dump = dumped
        vs = []
        guided = cfg and (hi <= 0 or lo <= s <= hi)
        for c, k in conds[: 2 if guided else 1]:
            t0 = time.time()
            vs.append(dit(x, s, c, k))
            if dit.dump is not None:
                dumped["dump_call"] = np.int32(len(calls["sigma"]))
            dit.dump = None
            calls["sigma"].append(s)
            calls["in"].append(x.copy())
            calls["out"].append(vs[-1])
            print(f"DiT call {len(calls['sigma'])}: sigma {s:.4f}, {time.time() - t0:.0f} s", flush=True)
            if a.calls and len(calls["sigma"]) >= a.calls:
                raise StopIteration
        v = vs[0]
        if len(vs) == 2:
            v = vs[1] + a.guidance * (vs[0] - vs[1])
        return v

    sg = sigmas(a.steps, a.shift)
    prev, h_last = None, 0.0
    try:
        for i in range(a.steps):
            s, sn = sg[i], sg[i + 1]
            v = vel(x, s)
            if a.sampler == "dpm2m":
                x, (x0, h) = dpm2m(x, v, prev, s, sn, h_last)
                prev, h_last = (x0,), h
            elif a.sampler == "heun" and sn != 0:
                v2 = vel(x + (sn - s) * v, sn)
                x = x + (sn - s) * (v + v2) / 2
            else:
                x = x + (sn - s) * v
    except StopIteration:
        a.no_vae = True
    rec["sigma"] = np.array(calls["sigma"], np.float32)
    rec["dit_in"], rec["dit_out"] = np.stack(calls["in"]), np.stack(calls["out"])
    rec["latents"] = x.astype(np.float32)
    rec.update(dumped)
    if dit.stats is not None:
        keys = sorted(dit.stats, key=lambda k: (k[0], int(k[1:].split(".")[0]), k))
        rec["stats_names"] = np.array(keys)
        rec["stats"] = np.array([dit.stats[k] for k in keys], np.float32)
        print("largest magnitudes:", ", ".join(f"{k} {dit.stats[k]:.3g}" for k in keys))
    if not a.no_vae:
        from diffusers import AutoencoderKL

        vae = AutoencoderKL.from_pretrained(a.root / "vae", torch_dtype=torch.float32).eval()
        gh = a.size // 16
        z = torch.from_numpy(x).reshape(gh, gh, 16, 2, 2).permute(2, 0, 3, 1, 4).reshape(1, 16, 2 * gh, 2 * gh)
        z = z / vae.config.scaling_factor + vae.config.shift_factor
        img = vae.decode(z, return_dict=False)[0]
        rec["decoded"] = img[0].numpy()
        pixels = ((img[0].clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()
        rec["image"] = pixels
        from PIL import Image

        Image.fromarray(pixels).save(a.out.with_suffix(".png"))
    np.savez(a.out.with_suffix(".npz"), **rec)
    print(f"{a.out.with_suffix('.npz')}: {a.size}x{a.size}, {len(calls['sigma'])} DiT calls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
