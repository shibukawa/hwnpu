#!/usr/bin/env python3
"""MOSS-SoundEffect v2.0 (OpenMOSS-Team/MOSS-SoundEffect-v2.0) on the CPU in fp32: the reference
the sound effects' NPU programs (internal/sfx, hwnpu-music generate -m sfx) are checked against.

The modules are those of OpenMOSS/MOSS-TTS moss_soundeffect_v2 (Apache-2.0): the DAC VAE's
(diffsynth/models/dac_vae.py, without audiotools), the Wan DiT's (wan_video_dit.py,
wan_audio_dit.py), the text encoder's (Qwen3 of transformers, its last hidden state) and the
flow-match scheduler's (schedulers/flow_match.py), copied here so that the reference needs only
torch, transformers and numpy.

  vae     decode latents (raw float32 (128, T)) or a WAV's own latents (the encoder's mean) into
          OUT.wav, writing OUT.latents.bin (when from a WAV), OUT.stageN.bin (each stage's output,
          float32 (C, T)), and OUT.samples.bin
  text    the prompt's token ids (OUT.ids.bin, int32) and the text encoder's output
          (OUT.context.bin, float32 (512, 2048), zeros past the prompt)
  run     the whole pipeline: OUT.noise.bin (float32 (128, 1500)) or --seed, every step's
          velocity of the conditional and the unconditional passes for the first --dump steps,
          the final latents and OUT.wav

  ~/.cache/hwnpu/dev/ref-venv/bin/python tools/ref/ref_moss_sfx.py vae MOSS OUT --wav in.wav --seconds 4
"""

import argparse
import json
import math
import os
import struct
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


# ---------------------------------------------------------------- DAC (dac_vae.py)

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    return x.reshape(shape)


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


class ResidualUnit(nn.Module):
    def __init__(self, dim=16, dilation=1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim=16, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(dim // 2, dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, d_model=64, strides=(2, 4, 8, 8), d_latent=64):
        super().__init__()
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]
        self.block += [Snake1d(d_model), WNConv1d(d_model, d_latent, kernel_size=3, padding=1)]
        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim=16, output_dim=8, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(input_dim, output_dim, kernel_size=2 * stride, stride=stride,
                              padding=math.ceil(stride / 2), output_padding=stride % 2),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(self, input_channel, channels, rates, d_out=1):
        super().__init__()
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]
        for i, stride in enumerate(rates):
            layers += [DecoderBlock(channels // 2 ** i, channels // 2 ** (i + 1), stride)]
        output_dim = channels // 2 ** len(rates)
        layers += [Snake1d(output_dim), WNConv1d(output_dim, d_out, kernel_size=7, padding=3), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DAC(nn.Module):
    def __init__(self, encoder_dim, encoder_rates, latent_dim, decoder_dim, decoder_rates, **_):
        super().__init__()
        self.hop_length = int(np.prod(encoder_rates))
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)
        self.quant_conv = nn.Conv1d(latent_dim, 2 * latent_dim, 1)
        self.post_quant_conv = nn.Conv1d(latent_dim, latent_dim, 1)
        self.decoder = Decoder(latent_dim, decoder_dim, decoder_rates)

    def encode_mean(self, audio):
        z = self.quant_conv(self.encoder(audio))
        return z[:, : z.shape[1] // 2]

    def decode(self, z):
        return self.decoder(self.post_quant_conv(z))


def load_dac(root):
    ck = torch.load(Path(root) / "vae" / "vae_128d_48k.pth", map_location="cpu", weights_only=False)
    m = DAC(**ck["metadata"]["kwargs"])
    missing, unexpected = m.load_state_dict(ck["state_dict"], strict=False)
    unexpected = [k for k in unexpected if not k.startswith("quantizer")]
    if missing or unexpected:
        raise SystemExit(f"DAC: missing {missing}, unexpected {unexpected}")
    return m.eval()



# ---------------------------------------------------------------- Wan DiT (wan_video_dit.py, wan_audio_dit.py)

def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis(dim, end=16384, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    pos = torch.arange(end, dtype=torch.float64)
    return torch.polar(torch.ones_like(torch.outer(pos, freqs)), torch.outer(pos, freqs))


def rope_apply(x, freqs, num_heads):
    b, s, nd = x.shape
    x = x.float().reshape(b, s, num_heads, -1, 2)
    xe, xo = x[..., 0], x[..., 1]
    cos, sin = freqs.real.float(), freqs.imag.float()
    return torch.stack((xe * cos - xo * sin, xe * sin + xo * cos), dim=-1).flatten(2)


def attention(q, k, v, n):
    b, s, _ = q.shape
    q, k, v = (t.reshape(b, -1, n, t.shape[-1] // n).transpose(1, 2) for t in (q, k, v))
    x = F.scaled_dot_product_attention(q, k, v)
    return x.transpose(1, 2).reshape(b, s, -1)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))
        self.norm_q = nn.RMSNorm(dim, eps=eps)
        self.norm_k = nn.RMSNorm(dim, eps=eps)

    def forward(self, x, freqs):
        q = rope_apply(self.norm_q(self.q(x)), freqs, self.num_heads)
        k = rope_apply(self.norm_k(self.k(x)), freqs, self.num_heads)
        return self.o(attention(q, k, self.v(x), self.num_heads))


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))
        self.norm_q = nn.RMSNorm(dim, eps=eps)
        self.norm_k = nn.RMSNorm(dim, eps=eps)

    def forward(self, x, y):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(y))
        return self.o(attention(q, k, self.v(y), self.num_heads))


class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim, eps=1e-6):
        super().__init__()
        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    def forward(self, x, context, t_mod, freqs, stats=None):
        sm, cm, gm, sf, cf, gf = (self.modulation + t_mod).chunk(6, dim=1)
        a = self.self_attn(self.norm1(x) * (1 + cm) + sm, freqs)
        x = x + gm * a
        c = self.cross_attn(self.norm3(x), context)
        x = x + c
        h = self.norm2(x) * (1 + cf) + sf
        f1 = self.ffn[1](self.ffn[0](h))
        f = self.ffn[2](f1)
        x = x + gf * f
        if stats is not None:
            stats.append({"self": a.abs().max().item(), "gate_self": (gm * a).abs().max().item(), "cross": c.abs().max().item(),
                          "ffn_hidden": f1.abs().max().item(), "ffn": f.abs().max().item(), "gate_ffn": (gf * f).abs().max().item(),
                          "x": x.abs().max().item(), "x_rms": x.pow(2).mean().sqrt().item()})
        return x


class Head(nn.Module):
    def __init__(self, dim, out_dim, eps):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x, t):
        shift, scale = (self.modulation + t.unsqueeze(1)).chunk(2, dim=1)
        return self.head(self.norm(x) * (1 + scale) + shift)


class WanAudio(nn.Module):
    def __init__(self, dim, in_dim, ffn_dim, out_dim, text_dim, freq_dim, eps, num_heads, num_layers, **_):
        super().__init__()
        self.dim, self.freq_dim, self.num_heads = dim, freq_dim, num_heads
        self.patch_embedding = nn.Conv1d(in_dim, dim, kernel_size=1, stride=1)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([DiTBlock(dim, num_heads, ffn_dim, eps) for _ in range(num_layers)])
        self.head = Head(dim, out_dim, eps)
        self.freqs = precompute_freqs_cis(dim // num_heads)

    def forward(self, x, timestep, context, stats=None):
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        x = self.patch_embedding(x).transpose(1, 2)
        f = x.shape[1]
        freqs = self.freqs[:f].view(f, 1, -1)
        for block in self.blocks:
            x = block(x, context, t_mod, freqs, stats)
        x = self.head(x, t)
        return x.transpose(1, 2)


# the exported checkpoint's diffusers-style names to WanAudioModel's (wan_audio.py's renames)
DIT_BLOCK_RENAME = {
    "attn1.": "self_attn.", "attn2.": "cross_attn.", "to_q.": "q.", "to_k.": "k.", "to_v.": "v.", "to_out.0.": "o.",
    "ffn.net.0.proj.": "ffn.0.", "ffn.net.2.": "ffn.2.", "norm2.": "norm3.", "scale_shift_table": "modulation",
}
DIT_GLOBAL_RENAME = {
    "condition_embedder.text_embedder.linear_1.": "text_embedding.0.", "condition_embedder.text_embedder.linear_2.": "text_embedding.2.",
    "condition_embedder.time_embedder.linear_1.": "time_embedding.0.", "condition_embedder.time_embedder.linear_2.": "time_embedding.2.",
    "condition_embedder.time_proj.": "time_projection.1.", "scale_shift_table": "head.modulation", "proj_out.": "head.head.",
}


def rename_dit(k):
    if k.startswith("blocks."):
        _, i, rest = k.split(".", 2)
        for a, b in DIT_BLOCK_RENAME.items():
            rest = rest.replace(a, b)
        return f"blocks.{i}.{rest}"
    for a, b in DIT_GLOBAL_RENAME.items():
        if k.startswith(a):
            return b + k[len(a):]
    return k


def load_dit(root):
    from safetensors.torch import load_file
    cfg = json.loads((Path(root) / "transformer" / "config.json").read_text())
    m = WanAudio(**cfg)
    sd = load_file(str(Path(root) / "transformer" / "diffusion_pytorch_model.safetensors"))
    sd = {rename_dit(k): v.float() for k, v in sd.items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"DiT: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return m.eval()


# ---------------------------------------------------------------- scheduler (flow_match.py: shift 5, sigma_min 0, extra_one_step)

def sigmas(steps, shift=5.0):
    s = torch.linspace(1.0, 0.0, steps + 1)[:-1]
    return shift * s / (1 + (shift - 1) * s)


# ---------------------------------------------------------------- text (wan_prompter.py, qwen3_text_encoder.py)

TEXT_LEN = 512


def clean(text):
    import html
    import re
    text = html.unescape(html.unescape(text)).strip()  # ftfy.fix_text is left out: plain text is its own fix
    return re.sub(r"\s+", " ", text).strip()


def encode_text(root, prompt):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(Path(root) / "tokenizer"))
    enc = tok([clean(prompt)], return_tensors="pt", padding="max_length", truncation=True, max_length=TEXT_LEN,
              add_special_tokens=True)
    ids, mask = enc.input_ids, enc.attention_mask
    n = int(mask.sum())
    model = AutoModelForCausalLM.from_pretrained(str(Path(root) / "text_encoder"), torch_dtype=torch.float32)
    # hidden_states[-1] of transformers 4.51 (MOSS's): the last layer's output after the final norm
    h = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    h[:, n:] = 0
    return ids[0, :n].numpy().astype(np.int32), h[0].float()

# ---------------------------------------------------------------- WAV

def read_wav(path, seconds):
    with wave.open(str(path)) as w:
        ch, rate, width = w.getnchannels(), w.getframerate(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise SystemExit("16-bit WAV only")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
    x = x.reshape(-1, ch).mean(axis=1)
    if rate != 48000:
        raise SystemExit("48 kHz WAV only")
    if seconds:
        x = x[: int(seconds * 48000)]
    return x


def write_wav(path, x, rate=48000):
    x = np.clip(np.asarray(x, dtype=np.float64), -1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((x * 32767).round().astype("<i2").tobytes())


def save(path, a, dtype=np.float32):
    np.ascontiguousarray(a, dtype=dtype).tofile(path)


# ---------------------------------------------------------------- commands

def cmd_vae(a):
    dac = load_dac(a.root)
    out = Path(a.out)
    with torch.no_grad():
        if a.wav:
            x = read_wav(a.wav, a.seconds)
            pad = (-len(x)) % dac.hop_length
            x = np.pad(x, (0, pad))
            z = dac.encode_mean(torch.from_numpy(x)[None, None])
            save(f"{out}.latents.bin", z[0].numpy())
        else:
            z = torch.from_numpy(np.fromfile(a.latents, dtype=np.float32).reshape(128, -1))[None]
        h = dac.post_quant_conv(z)
        layers = list(dac.decoder.model)
        h = layers[0](h)
        for i in range(5):
            h = layers[1 + i](h)
            if i == 4:
                h = layers[8](layers[7](layers[6](h)))
            save(f"{out}.stage{i}.bin", h[0].numpy())
            print(f"stage {i}: {tuple(h.shape)}, max |x| {h.abs().max().item():.4g}, rms {h.pow(2).mean().sqrt().item():.4g}")
        y = h[0, 0].numpy()
        save(f"{out}.samples.bin", y)
        write_wav(f"{out}.wav", y)


def cmd_text(a):
    ids, ctx = encode_text(a.root, a.prompt + (f" duration: {round(a.seconds, 1):.1f}s" if a.seconds else ""))
    print("ids", ids.tolist())
    save(f"{a.out}.ids.bin", ids, np.int32)
    save(f"{a.out}.context.bin", ctx.numpy())


def cmd_run(a):
    out = Path(a.out)
    seconds = round(a.seconds, 1)
    if a.context:
        ctx = torch.from_numpy(np.fromfile(a.context, dtype=np.float32).reshape(TEXT_LEN, -1))
    else:
        ids, ctx = encode_text(a.root, f"{a.prompt.strip()} duration: {seconds:.1f}s")
        save(f"{out}.ids.bin", ids, np.int32)
        save(f"{out}.context.bin", ctx.numpy())
    frames = 30 * 50
    if a.noise:
        x = torch.from_numpy(np.fromfile(a.noise, dtype=np.float32).reshape(1, 128, frames))
    else:
        g = torch.Generator().manual_seed(a.seed)
        x = torch.randn(1, 128, frames, generator=g)
        save(f"{out}.noise.bin", x[0].numpy())
    dit = load_dit(a.root)
    pos, neg = ctx[None], torch.zeros_like(ctx)[None]
    sg = sigmas(a.steps, a.shift)
    for i, sigma in enumerate(sg):
        t = (sigma * 1000).reshape(1)
        stats = [] if i == 0 else None
        vp = dit(x, t, pos, stats)
        vn = dit(x, t, neg)
        v = vn + a.cfg * (vp - vn)
        if stats is not None:
            for j, st in enumerate(stats):
                print(j, " ".join(f"{k} {v_:.4g}" for k, v_ in st.items()))
        if i < a.dump:
            save(f"{out}.step{i}.pos.bin", vp[0].numpy())
            save(f"{out}.step{i}.neg.bin", vn[0].numpy())
            save(f"{out}.step{i}.x.bin", x[0].numpy())
        nxt = sg[i + 1] if i + 1 < len(sg) else 0.0
        x = x + v * (nxt - sigma)
        print(f"step {i + 1}/{len(sg)}: sigma {sigma:.4f}", flush=True)
    save(f"{out}.final.bin", x[0].numpy())
    del dit
    dac = load_dac(a.root)
    y = dac.decode(x)[0, 0, : int(48000 * seconds)].numpy()
    save(f"{out}.samples.bin", y)
    write_wav(f"{out}.wav", y)


def cmd_hidden(a):
    """One pass of the DiT at the latents a.x (float32 (128, 1500)) and a.sigma, for the
    context: OUT.ctx.bin (the text embedding, (512, 1536)), OUT.h0.bin (the patch embedding) and
    OUT.hN.bin (the residual after block N - 1), each (1500, 1536), and OUT.v.bin (the velocity)."""
    ctx = torch.from_numpy(np.fromfile(a.context, dtype=np.float32).reshape(TEXT_LEN, -1))[None] if a.context else torch.zeros(1, TEXT_LEN, 2048)
    x = torch.from_numpy(np.fromfile(a.x, dtype=np.float32).reshape(1, 128, -1))
    dit = load_dit(a.root)
    t = torch.tensor([a.sigma * 1000], dtype=torch.float32)
    temb = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, t))
    tmod = dit.time_projection(temb).unflatten(1, (6, dit.dim))
    c = dit.text_embedding(ctx)
    save(f"{a.out}.ctx.bin", c[0].numpy())
    h = dit.patch_embedding(x).transpose(1, 2)
    f = h.shape[1]
    freqs = dit.freqs[:f].view(f, 1, -1)
    save(f"{a.out}.h0.bin", h[0].numpy())
    for i, b in enumerate(dit.blocks):
        h = b(h, c, tmod, freqs)
        save(f"{a.out}.h{i + 1}.bin", h[0].numpy())
    v = dit.head(h, temb).transpose(1, 2)
    save(f"{a.out}.v.bin", v[0].numpy())


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("vae")
    v.add_argument("root")
    v.add_argument("out")
    v.add_argument("--wav")
    v.add_argument("--latents")
    v.add_argument("--seconds", type=float, default=0)
    t = sub.add_parser("text")
    t.add_argument("root")
    t.add_argument("out")
    t.add_argument("--prompt", required=True)
    t.add_argument("--seconds", type=float, default=0, help="the duration suffix the pipeline adds (0: none)")
    r = sub.add_parser("run")
    r.add_argument("root")
    r.add_argument("out")
    r.add_argument("--prompt", default="")
    r.add_argument("--context", help="the text encoder's output (OUT.context.bin of text) in place of --prompt's")
    r.add_argument("--seconds", type=float, default=10)
    r.add_argument("--steps", type=int, default=100)
    r.add_argument("--cfg", type=float, default=4.0)
    r.add_argument("--shift", type=float, default=5.0)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--noise")
    r.add_argument("--dump", type=int, default=2)
    h = sub.add_parser("hidden")
    h.add_argument("root")
    h.add_argument("out")
    h.add_argument("--x", required=True)
    h.add_argument("--sigma", type=float, required=True)
    h.add_argument("--context")
    a = p.parse_args()
    torch.set_grad_enabled(False)
    {"vae": cmd_vae, "text": cmd_text, "run": cmd_run, "hidden": cmd_hidden}[a.cmd](a)


if __name__ == "__main__":
    main()
