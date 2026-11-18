#!/usr/bin/env python3
"""FLUX.3 [klein] (black-forest-labs/flux-3-image-klein) with diffusers' Flux3 modules: the reference
an NPU pipeline of it is checked against, and the check of the DiT's weight formats.

Written from diffusers 0.43.0's modular_pipelines/flux3 (the step-distilled blocks) without the modular
pipeline, so that the parts are loaded one after another: a 16 GB Mac cannot hold the text encoder
(Qwen3-VL, 4.4B, 8.9 GB in bf16) and the DiT (5.5B, 11.1 GB) at once. The prompt upsampler (a Qwen3.5
LLM) is not run: the prompt is encoded as given.

  1. the prompt in Qwen3-VL's chat template, padded on the right to a multiple of 80 tokens (the
     padding's position ids 1, as in transformers 4.57); the hidden states after layers 4, 8, ..., 32
     side by side (rows of 8 x 2560 = 20480);
  2. a reference (--image, an edit) through the VAE's encoder (its mode), 2 x 2 patches (128 channels
     at a 16th of the picture), normalized by the VAE's batch-norm statistics; position ids (0, h/2,
     w/2, -1), the picture's (0, h/2, w/2, 0), the text's (0, 0, 0, i);
  3. 16 flow-map steps without guidance: sigmas linspace(1, 1/16) shifted by 6.93 (flow-matching
     Euler); each step gives the DiT its timestep and the next one (target_timestep, 0 at the last);
  4. the VAE's decoder on the latents un-normalized and un-patched.

The DiT runs in bf16 on PyTorch's MPS (the float32 DiT, 22 GB, does not fit), the VAE in float32.

  python tools/ref/ref_flux3klein.py OUT --size 512 --prompt "..."                 # text to image
  python tools/ref/ref_flux3klein.py OUT --size 512 --image in.png --prompt "..."  # an edit

writes OUT.npz (text rows, noise, the reference's latents, each step's sigma, input and DiT output, the
final latents) and OUT.png. The weight formats (new-model-checklist 5.1-5.3):

  python tools/ref/flux3_quant.py int8 ~/models/flux3-q8          # the DiT's linear weights
  python tools/ref/ref_flux3klein.py OUT --from REF --quant int8 --qdir ~/models/flux3-q8

reruns REF.npz's prompt and reference (its text rows and latents: no text encoder, no VAE encoder) with
the DiT's linear weights held in int8 or lut8 (expanded to bf16 for each call; the DiT takes about
half of its bf16 memory) and prints each step's relative RMS error of the DiT's output on REF's own
input (teacher forced), the final latents' cosine of a free run from REF's noise, and the PSNR of the
pictures. Run in the reference virtual environment (torch with MPS, transformers 5.x, diffusers 0.43).
"""

import argparse
import gc
import math
import os
import time

import numpy as np
import torch
from PIL import Image

MODEL = os.path.expanduser("~/models/flux-3-image-klein")
DEV = torch.device("mps")
DT = torch.bfloat16
SHIFT = 6.93  # the constant timestep shift ("alpha") of the FLUX.3 recipe
STEPS = 16
LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)  # Qwen3-VL's hidden states stacked along the channels
PAD = 80
MAXLEN = 8192
IMAGE_ROPE_SCALE = 0.5
GROUP = 16  # rows of a lut8 table


def free():
    gc.collect()
    torch.mps.empty_cache()


def encode_prompt(prompt):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    tok = AutoProcessor.from_pretrained(f"{MODEL}/text_encoder")
    te = Qwen3VLForConditionalGeneration.from_pretrained(f"{MODEL}/text_encoder", dtype=DT).to(DEV).eval()
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    real = tok.tokenizer(text, return_tensors="pt", truncation=True, max_length=MAXLEN)["input_ids"].shape[1]
    padded = min(math.ceil(real / PAD) * PAD, MAXLEN)
    inp = tok.tokenizer(text, return_tensors="pt", padding="max_length", padding_side="right",
                        truncation=True, max_length=padded)
    ids, mask = inp["input_ids"].to(DEV), inp["attention_mask"].to(DEV)
    pos = torch.arange(padded, device=DEV)[None].masked_fill(mask == 0, 1)
    with torch.no_grad():
        out = te(input_ids=ids, attention_mask=mask, position_ids=pos, output_hidden_states=True, use_cache=False)
    st = torch.stack([out.hidden_states[k] for k in LAYERS], dim=1).to(DT)  # (1, 8, L, 2560)
    b, c, n, h = st.shape
    rows = st.permute(0, 2, 1, 3).reshape(b, n, c * h)  # (1, L, 20480)
    print(f"prompt: {real} tokens, window {padded}", flush=True)
    del te, out, st
    free()
    return rows


def txt_ids(n):
    return torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(n)).float()


def img_ids(h, w, l_coord=0):
    ids = torch.cartesian_prod(torch.arange(1), torch.arange(h), torch.arange(w), torch.tensor([l_coord])).float()
    ids[:, 1:3] = ids[:, 1:3] * IMAGE_ROPE_SCALE
    return ids


def patchify(lat):
    b, c, h, w = lat.shape
    lat = lat.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4)
    return lat.reshape(b, c * 4, h // 2, w // 2)


def unpatchify(lat):
    b, c, h, w = lat.shape
    lat = lat.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
    return lat.reshape(b, c // 4, h * 2, w * 2)


def load_vae():
    from diffusers import AutoencoderKLFlux2

    return AutoencoderKLFlux2.from_pretrained(f"{MODEL}/vae", dtype=torch.float32).to(DEV).eval()


def bn_stats(vae):
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(DEV, torch.float32)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(DEV, torch.float32)
    return mean, std


def encode_reference(vae, path, size):
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    x = torch.from_numpy(np.asarray(img)).float().permute(2, 0, 1)[None] / 127.5 - 1.0
    mean, std = bn_stats(vae)
    with torch.no_grad():
        z = vae.encode(x.to(DEV)).latent_dist.mode()
    return (patchify(z) - mean) / std  # (1, 128, size/16, size/16) float32


def decode(vae, lat, lh, lw):
    z = lat.transpose(1, 2).reshape(1, 128, lh, lw).float()
    mean, std = bn_stats(vae)
    with torch.no_grad():
        x = vae.decode(unpatchify(z * std + mean), return_dict=False)[0]
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()


class QLinear(torch.nn.Module):
    """A linear layer whose weight stays in 8 bits, expanded to bf16 for each call: int8 with one fp16
    scale per row (aux of shape (rows,)), or lut8 with one 256-entry fp16 table per 16 rows (aux of
    shape (rows/16, 256)); tools/ref/flux3_quant.py."""

    def __init__(self, q, aux, bias):
        super().__init__()
        self.fmt = "int8" if aux.ndim == 1 else "lut8"
        self.register_buffer("q", q)
        self.register_buffer("aux", aux)
        self.bias = None if bias is None else torch.nn.Parameter(bias, requires_grad=False)

    def weight(self):
        if self.fmt == "int8":
            return self.q.to(DT) * self.aux.to(DT).unsqueeze(1)
        rows, cols = self.q.shape
        return torch.gather(self.aux.to(DT), 1, self.q.view(rows // GROUP, GROUP * cols).long()).view(rows, cols)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight(), self.bias)


def load_transformer(qfile, overrides):
    """The DiT in bf16 (qfile None), or with its linear weights in 8 bits from qfile, the layers a
    pattern of overrides (REGEX=FILE, the first that matches) names from that file instead; FILE
    "bf16" keeps those layers' checkpoint weights (fp16 layers on the Neural Engine, roughly)."""
    import re

    from diffusers.models import Flux3Transformer2DModel

    if qfile is None:
        return Flux3Transformer2DModel.from_pretrained(f"{MODEL}/transformer", dtype=DT).to(DEV).eval()
    from accelerate import init_empty_weights
    from safetensors import safe_open

    with init_empty_weights():
        tr = Flux3Transformer2DModel.from_config(Flux3Transformer2DModel.load_config(f"{MODEL}/transformer"))
    rules = []
    for o in overrides:
        pat, _, path = o.rpartition("=")
        rules.append((re.compile(pat), path if path == "bf16" else os.path.expanduser(path)))
    files = {}
    wmap = None

    def original(name):
        nonlocal wmap
        if wmap is None:
            import json

            wmap = json.load(open(f"{MODEL}/transformer/diffusion_pytorch_model.safetensors.index.json"))["weight_map"]
        return opened(f"{MODEL}/transformer/{wmap[name]}").get_tensor(name)

    def opened(path):
        if path not in files:
            files[path] = safe_open(path, framework="pt", device="cpu")
        return files[path]

    base = opened(os.path.expanduser(qfile))
    keys = set(base.keys())
    qnames = sorted(k[: -len(".q")] for k in keys if k.endswith(".weight.q"))
    done, kinds, assigned = set(), {}, set()
    for wname in qnames:  # each 8-bit layer, moved to the GPU a tensor at a time
        mod = wname[: -len(".weight")]
        src = base
        for rx, path in rules:
            if rx.search(mod):
                src = "bf16" if path == "bf16" else opened(path)
                break
        if src == "bf16":
            parent_name, _, child = mod.rpartition(".")
            parent = tr.get_submodule(parent_name) if parent_name else tr
            w = original(wname)
            bias = base.get_tensor(mod + ".bias") if (mod + ".bias") in keys else None
            lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=bias is not None, device="meta")
            lin.weight = torch.nn.Parameter(w.to(DEV, DT), requires_grad=False)
            if bias is not None:
                lin.bias = torch.nn.Parameter(bias.to(DEV, DT), requires_grad=False)
            setattr(parent, child, lin)
            kinds["bf16"] = kinds.get("bf16", 0) + 1
            done |= {wname + ".q", wname + ".s", wname + ".lut", mod + ".bias"}
            assigned |= {mod + ".weight", mod + ".bias"}
            continue
        aux_key = wname + ".s" if (wname + ".s") in set(src.keys()) else wname + ".lut"
        parent_name, _, child = mod.rpartition(".")
        parent = tr.get_submodule(parent_name) if parent_name else tr
        old = getattr(parent, child)
        q = src.get_tensor(wname + ".q").to(DEV)
        if tuple(old.weight.shape) != tuple(q.shape):
            raise RuntimeError(f"{wname}: {tuple(q.shape)} for {tuple(old.weight.shape)}")
        aux = src.get_tensor(aux_key).to(DEV)
        bias = base.get_tensor(mod + ".bias").to(DEV, DT) if (mod + ".bias") in keys else None
        layer = QLinear(q, aux, bias)
        setattr(parent, child, layer)
        kinds[layer.fmt] = kinds.get(layer.fmt, 0) + 1
        done |= {wname + ".q", wname + ".s", wname + ".lut", mod + ".bias"}
        assigned |= {mod + ".q", mod + ".aux", mod + ".bias"}
    rest = {k: base.get_tensor(k).to(DEV, DT) for k in keys if k not in done}
    missing, unexpected = tr.load_state_dict(rest, strict=False, assign=True)
    missing = [m for m in missing if m not in assigned]
    if missing or unexpected:
        raise RuntimeError(f"missing {missing[:5]}, unexpected {unexpected[:5]}")
    print(f"transformer: {len(qnames)} linear layers {kinds}", flush=True)
    return tr.eval()


class X2Stats:
    """E[x^2] of each linear layer's input, per in-feature (column), summed over the tokens of every
    call: the column weights of flux3_quant.py --fit wkmeans."""

    def __init__(self, tr, path):
        self.path, self.acc, self.hooks = path, {}, []
        if os.path.exists(path):
            self.acc = torch.load(path)
        for name, m in tr.named_modules():
            if isinstance(m, (torch.nn.Linear, QLinear)):
                self.hooks.append(m.register_forward_pre_hook(self._hook(name)))

    def _hook(self, name):
        def f(_m, args):
            x = args[0].detach().float().reshape(-1, args[0].shape[-1])
            e = self.acc.setdefault(name, {"sum": torch.zeros(x.shape[-1]), "count": 0})
            e["sum"] += (x * x).sum(dim=0).cpu()
            e["count"] += x.shape[0]

        return f

    def save(self):
        for h in self.hooks:
            h.remove()
        torch.save(self.acc, self.path)
        print(f"E[x^2] of {len(self.acc)} layers -> {self.path}", flush=True)


def schedule():
    from diffusers import FlowMatchEulerDiscreteScheduler

    sch = FlowMatchEulerDiscreteScheduler(shift=SHIFT)
    sch.set_timesteps(sigmas=np.linspace(1.0, 1 / STEPS, STEPS), device="cpu")
    return sch


def dit_call(tr, x, rows, cond, t, nxt, ids):
    kw = dict(hidden_states=x.to(DEV, DT), encoder_hidden_states=rows, timestep=(t / 1000).expand(1).to(DEV, DT),
              target_timestep=(nxt / 1000).expand(1).to(DEV, DT), img_ids=ids["img"], txt_ids=ids["txt"],
              return_dict=False)
    if cond is not None:
        kw.update(image_cond_hidden_states=cond, image_cond_ids=ids["cond"])
    with torch.no_grad():
        return tr(**kw)[0].float()


def rel_rms(a, b):
    return float(torch.sqrt(((a - b) ** 2).mean()) / torch.sqrt((b**2).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--prompt")
    ap.add_argument("--image", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--from", dest="ref", default=None, help="REF.npz: rerun its prompt and reference")
    ap.add_argument("--quant", default="none", choices=["none", "int8", "lut8"],
                    help="the DiT's linear weights: QDIR/transformer_QUANT.safetensors (--qfile overrides)")
    ap.add_argument("--qdir", default="~/models/flux3-q8")
    ap.add_argument("--qfile", default=None, help="a flux3_quant.py file for the DiT's linear weights")
    ap.add_argument("--override", action="append", default=[], metavar="REGEX=FILE",
                    help="the layers whose module name matches REGEX from FILE instead (repeatable)")
    ap.add_argument("--teacher-only", action="store_true", help="with --from: the teacher-forced errors alone")
    ap.add_argument("--stats", default=None, help="accumulate E[x^2] of each linear layer's input here")
    a = ap.parse_args()
    t0 = time.time()

    ref = np.load(a.ref if a.ref.endswith(".npz") else a.ref + ".npz") if a.ref else None
    if ref is not None:
        size = int(ref["size"])
        rows = torch.from_numpy(ref["text_rows"]).to(DEV, DT)
        cond = torch.from_numpy(ref["cond"]).to(DEV, DT) if ref["cond"].size else None
        noise = torch.from_numpy(ref["noise"]).to(DEV)
    else:
        if not a.prompt:
            ap.error("--prompt is required without --from")
        size = a.size
        rows = encode_prompt(a.prompt)
        noise = None
        cond = None
    lh = lw = size // 16
    vae = load_vae()
    if ref is None and a.image:
        c = encode_reference(vae, a.image, size)
        cond = c.flatten(2).transpose(1, 2).to(DT)  # (1, tokens, 128)
        print(f"reference: {tuple(cond.shape)}", flush=True)
    if noise is None:
        g = torch.Generator("cpu").manual_seed(a.seed)
        noise = torch.randn((1, 128, lh, lw), generator=g, dtype=torch.float32).flatten(2).transpose(1, 2).to(DEV)

    qfile = a.qfile or (None if a.quant == "none" else os.path.join(os.path.expanduser(a.qdir), f"transformer_{a.quant}.safetensors"))
    tr = load_transformer(qfile, a.override)
    label = os.path.basename(qfile) if qfile else "bf16"
    if a.override:
        label += " + " + ",".join(a.override)
    stats = X2Stats(tr, os.path.expanduser(a.stats)) if a.stats else None
    print(f"loaded in {time.time() - t0:.1f} s", flush=True)
    ids = {"img": img_ids(lh, lw).to(DEV), "txt": txt_ids(rows.shape[1]).to(DEV),
           "cond": img_ids(lh, lw, -1).to(DEV) if cond is not None else None}
    sch = schedule()
    ts = sch.timesteps

    # teacher forced: the DiT's output on REF's own input of each step
    if ref is not None:
        errs = []
        for i, t in enumerate(ts):
            nxt = ts[i + 1] if i + 1 < len(ts) else torch.zeros_like(t)
            v = dit_call(tr, torch.from_numpy(ref["step_in"][i]).to(DEV), rows, cond, t, nxt, ids)
            errs.append(rel_rms(v, torch.from_numpy(ref["step_out"][i]).to(DEV)))
        print("velocity rel RMS by step: " + " ".join(f"{e:.4f}" for e in errs), flush=True)
        print(f"velocity rel RMS over all steps: {float(np.sqrt(np.mean(np.square(errs)))):.4f} [{label}]", flush=True)
        if a.teacher_only:
            print(f"total {time.time() - t0:.1f} s", flush=True)
            return

    # the free run
    lat = noise.clone()
    sigmas, step_in, step_out = [], [], []
    t1 = time.time()
    for i, t in enumerate(ts):
        nxt = ts[i + 1] if i + 1 < len(ts) else torch.zeros_like(t)
        v = dit_call(tr, lat, rows, cond, t, nxt, ids)
        step_in.append(lat.cpu().numpy())
        step_out.append(v.cpu().numpy())
        sigmas.append(float(t) / 1000)
        lat = sch.step(v, t, lat, return_dict=False)[0]
        print(f"step {i + 1}/{STEPS} t={float(t):.2f} -> {float(nxt):.2f} ({time.time() - t1:.1f} s)", flush=True)
    if stats:
        stats.save()
    del tr
    free()

    img = decode(vae, lat, lh, lw)
    Image.fromarray(img).save(a.out + ".png")
    np.savez(a.out + ".npz", size=size, text_rows=rows.float().cpu().numpy(),
             cond=cond.float().cpu().numpy() if cond is not None else np.zeros(0, np.float32),
             noise=noise.cpu().numpy(), sigmas=np.array(sigmas), step_in=np.stack(step_in),
             step_out=np.stack(step_out), latents=lat.cpu().numpy(), quant=label)
    if ref is not None:
        z, x = torch.from_numpy(ref["latents"]).double().ravel(), lat.cpu().double().ravel()
        cos = float(z @ x / z.norm() / x.norm())
        rpng = (a.ref[: -len(".npz")] if a.ref.endswith(".npz") else a.ref) + ".png"
        r = np.asarray(Image.open(rpng).convert("RGB"), np.float64)
        mse = float(((r - img.astype(np.float64)) ** 2).mean())
        psnr = 10 * math.log10(255**2 / mse) if mse else float("inf")
        print(f"final latents cosine {cos:.5f}, PSNR {psnr:.2f} dB against {rpng} [{label}]", flush=True)
    print(f"wrote {a.out}.png and .npz ({size} x {size}); steps {time.time() - t1:.1f} s, total {time.time() - t0:.1f} s",
          flush=True)


if __name__ == "__main__":
    main()
