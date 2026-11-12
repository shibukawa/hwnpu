# Qwen-Image-2.1's programs as Core AI programs (.aimodel), for the Core AI experiment: the same
# graphs as internal/imagegen/qwenimage/mil (residual streams divided by DitScale and TextScale,
# every norm of a row first divided by its largest magnitude), written in torch and converted with
# coreai-torch. The matrices are hwnpu's own lookup tables (tools/coreai/qlut), the norms' weights
# read from the bf16 checkpoints.
#
#   python qwenimage_dit.py dit     ROOT LUTS OUT.aimodel --first 0 --count 4 [--final] [--check]
#   python qwenimage_dit.py dittext ROOT LUTS OUT.aimodel --first 0 --count 4 [--check]
#   python qwenimage_dit.py text    ROOT LUTS OUT.aimodel --text TEXTROOT --first 0 --count 4 [--final] [--check]
#   python qwenimage_dit.py vae     ROOT - OUT.aimodel --first STAGE [--check]
#
# dit is mil.Blocks (the image's rows), dittext mil.TextBlocks (the text's rows at t = 0, giving
# each block's keys and values), text mil.TextGroup (Qwen3-VL-8B's layers, the last group with the
# DiT's txt_in), vae mil.VaePart (stage --first of the decoder, fp16, 512 x 512). --check runs the torch float32 graph (the tables expanded) and the program on the
# Neural Engine on the same random inputs and prints their cosine and relative RMS error.
import argparse, json, math, time, warnings
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

warnings.filterwarnings("ignore")

DIT_SCALE = 8
TEXT_SCALE = 4
TEXT_ROWS = 512


def head_perm(D):
    return list(range(0, D, 2)) + list(range(1, D, 2))


def lut_dense(idx, lut):
    # idx (rows, cols) uint8, lut (rows/g, 1, K, 1): each block of g rows its own table
    g = idx.shape[0] // lut.shape[0]
    tab = lut[:, 0, :, 0].float().repeat_interleave(g, 0)  # (rows, K)
    return torch.gather(tab, 1, idx.long())


class Lin(torch.nn.Module):
    def __init__(s, luts, name, dense):
        super().__init__()
        idx, lut = luts.get_tensor(name + ".idx"), luts.get_tensor(name + ".lut")
        if dense:
            s.register_buffer("w", lut_dense(idx, lut))
            s.p = None
        else:
            from coreai_torch._compression.custom_layers import PalettizeModule
            s.p = PalettizeModule(idx, lut.half())

    def forward(s, x):
        return torch.nn.functional.linear(x, s.w if s.p is None else s.p())


def max_scaled(x):
    m = torch.maximum(x.abs().amax(-1, keepdim=True), torch.tensor(1e-3, dtype=x.dtype))
    return x / m


def scaled_layer_norm(x, eps):
    x = max_scaled(x)
    d = x - x.mean(-1, keepdim=True)
    return d * torch.rsqrt((d * d).mean(-1, keepdim=True) + eps)


def rms_norm(x, w, eps):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * w


def rope(x, cos, sin):
    h = x.shape[-1] // 2
    return x * cos + torch.cat([-x[..., h:], x[..., :h]], -1) * sin


def silu(x):
    return x / (1 + torch.exp(-x))


def tanh_gelu(x):
    return x / (1 + torch.exp(-1.5957691 * (x + 0.044715 * x * x * x)))


def softmax(x):
    # The softmax over the last axis, as exp times the reciprocal of its sum. Core AI (macOS 27.2)
    # turns matmul, softmax, matmul (and exp(x - max) / sum) into one attention operation that it
    # does not place on the Neural Engine: the whole program then runs on the GPU. This form it
    # leaves alone.
    e = torch.exp(x - x.amax(-1, keepdim=True))
    return e * torch.reciprocal(e.sum(-1, keepdim=True))


def causal_mask(n):
    return torch.triu(torch.full((n, n), -1e4), 1).reshape(1, 1, n, n)


def dit_rope(cfg, n):
    # mil.RopeTables of mil.TextPositions(n): position i on all three axes
    freqs = []
    for d in cfg["axes_dims_rope"]:
        freqs += [1 / 10000 ** (k / d) for k in range(0, d, 2)]
    a = torch.arange(n, dtype=torch.float64)[:, None] * torch.tensor(freqs, dtype=torch.float64)[None]
    cos, sin = torch.cat([a.cos()] * 2, -1), torch.cat([a.sin()] * 2, -1)
    D = cfg["attention_head_dim"]
    return cos.float().reshape(1, n, 1, D), sin.float().reshape(1, n, 1, D)


class DitBlock(torch.nn.Module):
    def __init__(s, ckpt, luts, b, dense):
        super().__init__()
        p = f"dit.transformer_blocks.{b}."
        for n in ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "img_mlp.gate_layer", "img_mlp.proj", "img_mlp.out"]:
            setattr(s, n.replace(".", "_"), Lin(luts, p + n + ".weight", dense))
        D = ckpt(p + "attn.norm_q.weight").shape[0]
        s.register_buffer("nq", ckpt(p + "attn.norm_q.weight")[head_perm(D)].float())
        s.register_buffer("nk", ckpt(p + "attn.norm_k.weight")[head_perm(D)].float())

    def qkv(s, x, m1, cos, sin, H, D, eps):
        n = x.shape[1]
        h = scaled_layer_norm(x, eps) * m1
        q = s.attn_to_q(h).reshape(1, n, H, D)
        k = s.attn_to_k(h).reshape(1, n, H, D)
        v = s.attn_to_v(h).reshape(1, n, H, D)
        q = rope(rms_norm(max_scaled(q), s.nq.to(q.dtype), eps), cos, sin)
        k = rope(rms_norm(max_scaled(k), s.nk.to(k.dtype), eps), cos, sin)
        return q, k, v

    def rest(s, x, o, g1, m2, g2, eps):
        x = x + s.attn_to_out_0(o) * g1
        h = scaled_layer_norm(x, eps) * m2
        return x + s.img_mlp_out(silu(s.img_mlp_gate_layer(h)) * s.img_mlp_proj(h)) * g2


def attend(q, k, v, bias):
    # q (1, n, H, D), k and v (1, m, H, D), bias added to the scores (1, H or 1, n or 1, m)
    D = q.shape[-1]
    qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    sc = torch.matmul(qh, kh.transpose(-1, -2)) * (1 / math.sqrt(D)) + bias
    o = torch.matmul(softmax(sc), vh).transpose(1, 2)
    return o.reshape(q.shape[0], q.shape[1], -1)


class Blocks(torch.nn.Module):
    """mil.Blocks: inputs x, mod, cos, sin, tk, tv, kbias (and scale for the last group)."""

    def __init__(s, cfg, ckpt, luts, first, count, final, dense):
        super().__init__()
        s.E, s.H, s.D = cfg["num_attention_heads"] * cfg["attention_head_dim"], cfg["num_attention_heads"], cfg["attention_head_dim"]
        s.eps = cfg["eps"]
        s.img_in = Lin(luts, "dit.img_in.weight", dense) if first == 0 else None
        s.proj_out = Lin(luts, "dit.proj_out.weight", dense) if final else None
        s.blocks = torch.nn.ModuleList([DitBlock(ckpt, luts, b, dense) for b in range(first, first + count)])

    def forward(s, x, mod, cos, sin, tk, tv, kbias, scale=None):
        E, H, D, eps = s.E, s.H, s.D, s.eps
        if s.img_in is not None:
            x = s.img_in(x)
        m1, g1, m2, g2 = (mod[..., i * E:(i + 1) * E] for i in range(4))
        for j, b in enumerate(s.blocks):
            q, k, v = b.qkv(x, m1, cos, sin, H, D, eps)
            tkj = tk[:, j * TEXT_ROWS:(j + 1) * TEXT_ROWS].reshape(1, TEXT_ROWS, H, D)
            tvj = tv[:, j * TEXT_ROWS:(j + 1) * TEXT_ROWS].reshape(1, TEXT_ROWS, H, D)
            o = attend(q, torch.cat([k, tkj], 1), torch.cat([v, tvj], 1), kbias)
            x = b.rest(x, o, g1, m2, g2, eps)
        if s.proj_out is not None:
            x = s.proj_out(scaled_layer_norm(x, eps) * scale)
        return x


class TextBlocks(torch.nn.Module):
    """mil.TextBlocks: inputs x, mod; output the rows after the blocks, then each block's k and v."""

    def __init__(s, cfg, ckpt, luts, first, count, dense):
        super().__init__()
        s.E, s.H, s.D = cfg["num_attention_heads"] * cfg["attention_head_dim"], cfg["num_attention_heads"], cfg["attention_head_dim"]
        s.eps = cfg["eps"]
        s.blocks = torch.nn.ModuleList([DitBlock(ckpt, luts, b, dense) for b in range(first, first + count)])
        cos, sin = dit_rope(cfg, TEXT_ROWS)
        s.register_buffer("cos", cos)
        s.register_buffer("sin", sin)
        s.register_buffer("causal", causal_mask(TEXT_ROWS))

    def forward(s, x, mod):
        E, H, D, eps = s.E, s.H, s.D, s.eps
        cos, sin, causal = s.cos.to(x.dtype), s.sin.to(x.dtype), s.causal.to(x.dtype)
        m1, g1, m2, g2 = (mod[..., i * E:(i + 1) * E] for i in range(4))
        outs = []
        for b in s.blocks:
            q, k, v = b.qkv(x, m1, cos, sin, H, D, eps)
            outs += [k.reshape(1, TEXT_ROWS, E), v.reshape(1, TEXT_ROWS, E)]
            x = b.rest(x, attend(q, k, v, causal), g1, m2, g2, eps)
        return torch.cat([x] + outs, 1)


class TextGroup(torch.nn.Module):
    """mil.TextGroup: Qwen3-VL-8B's layers over TEXT_ROWS positions, causal; the last with txt_in."""

    def __init__(s, tcfg, dcfg, ckpt, luts, first, count, final, dense):
        super().__init__()
        s.H, s.KV, s.D, s.eps = tcfg["num_attention_heads"], tcfg["num_key_value_heads"], tcfg["head_dim"], tcfg["rms_norm_eps"]
        s.deps = dcfg["eps"]
        s.layers = torch.nn.ModuleList()
        for i in range(first, first + count):
            p = f"text.layers.{i}."
            m = torch.nn.Module()
            for n in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
                setattr(m, n.replace(".", "_"), Lin(luts, p + n + ".weight", dense))
            for n in ["input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"]:
                m.register_buffer(n.replace(".", "_"), ckpt(p + n + ".weight").float())
            s.layers.append(m)
        s.txt_in = None
        if final:
            s.txt_in = torch.nn.Module()
            s.txt_in.in_layer = Lin(luts, "dit.txt_in.in_layer.weight", dense)
            s.txt_in.out_layer = Lin(luts, "dit.txt_in.out_layer.weight", dense)
            s.txt_in.register_buffer("norm", ckpt("dit.txt_in.text_norm.weight").float() + 1)
        inv = 1 / tcfg["rope_theta"] ** (torch.arange(0, s.D, 2, dtype=torch.float64) / s.D)
        a = torch.arange(TEXT_ROWS, dtype=torch.float64)[:, None] * inv[None]
        s.register_buffer("cos", torch.cat([a.cos()] * 2, -1).float().reshape(1, TEXT_ROWS, 1, s.D))
        s.register_buffer("sin", torch.cat([a.sin()] * 2, -1).float().reshape(1, TEXT_ROWS, 1, s.D))
        s.register_buffer("causal", causal_mask(TEXT_ROWS))

    def forward(s, x):
        H, KV, D, eps, L = s.H, s.KV, s.D, s.eps, TEXT_ROWS
        cos, sin, causal = s.cos.to(x.dtype), s.sin.to(x.dtype), s.causal.to(x.dtype)
        rep = H // KV
        for m in s.layers:
            h = rms_norm(max_scaled(x), m.input_layernorm.to(x.dtype), eps)
            q = rope(rms_norm(max_scaled(m.self_attn_q_proj(h).reshape(1, L, H, D)), m.self_attn_q_norm.to(x.dtype), eps), cos, sin)
            k = rope(rms_norm(max_scaled(m.self_attn_k_proj(h).reshape(1, L, KV, D)), m.self_attn_k_norm.to(x.dtype), eps), cos, sin)
            v = m.self_attn_v_proj(h).reshape(1, L, KV, D)
            # blocks.SDPAGrouped: the query heads sharing a kv head stacked along the rows
            qg = q.transpose(1, 2).reshape(1, KV, rep * L, D)
            sc = torch.matmul(qg, k.transpose(1, 2).transpose(-1, -2)) * (1 / math.sqrt(D))
            sc = (sc.reshape(1, H, L, L) + causal).reshape(1, KV, rep * L, L)
            o = torch.matmul(softmax(sc), v.transpose(1, 2)).reshape(1, H, L, D)
            x = x + m.self_attn_o_proj(o.transpose(1, 2).reshape(1, L, H * D))
            h = rms_norm(max_scaled(x), m.post_attention_layernorm.to(x.dtype), eps)
            x = x + m.mlp_down_proj(silu(m.mlp_gate_proj(h)) * m.mlp_up_proj(h))
        if s.txt_in is not None:
            h = rms_norm(max_scaled(x), s.txt_in.norm.to(x.dtype), s.deps)
            x = s.txt_in.out_layer(tanh_gelu(s.txt_in.in_layer(h)))
        return x


def vae_dims(v):
    m = [v["dim_mult"][-1]] + v["dim_mult"][::-1]
    return [v["decoder_base_dim"] * u for u in m]


def vae_stages(v, w, h):
    # mil.VaeStages: the middle, each up block, the last split after its first residual block
    dims = vae_dims(v)
    tup = v["temperal_downsample"][::-1]
    gh, gw = h // 16, w // 16
    d = "vae.decoder."
    res = lambda path, i, o: ("res", d + path, i, o, 0)
    s = [([("in", "", 0, dims[0], 0), res("mid_block.resnets.0", dims[0], dims[0]), ("attn", d + "mid_block.attentions.0", 0, 0, 0),
           res("mid_block.resnets.1", dims[0], dims[0])], v["z_dim"], gh, gw)]
    for i in range(len(dims) - 1):
        cin, cout = dims[i], dims[i + 1]
        p = f"up_blocks.{i}"
        ops = [res(f"{p}.resnets.{j}", cin if j == 0 else cout, cout) for j in range(v["num_res_blocks"] + 1)]
        if i < len(dims) - 2:
            ops.append(("up", d + p, cin, cout, 2 if tup[i] else 1))
            s.append((ops, cin, gh, gw))
            gh, gw = 2 * gh, 2 * gw
            continue
        s.append((ops[:1], cin, gh, gw))
        s.append((ops[1:] + [("head", "", cout, v["out_channels"], 0)], cout, gh, gw))
    return s


def dup_source(cin, cout, ft):
    rep = cout * ft * 4 // cin
    return [(((co * ft + (ft - 1)) * 2 + i) * 2 + j) // rep for co in range(cout) for i in range(2) for j in range(2)]


def chan_norm(x, gamma):
    x = x / torch.maximum(x.abs().amax(1, keepdim=True), torch.tensor(1e-3, dtype=x.dtype))
    return x * torch.rsqrt((x * x).mean(1, keepdim=True) + 1e-12) * gamma


class Conv(torch.nn.Module):
    def __init__(s, w, b):
        super().__init__()
        s.register_buffer("w", w.float()); s.register_buffer("b", None if b is None else b.float())
        s.pad = w.shape[-1] // 2

    def forward(s, x):
        return torch.nn.functional.conv2d(x, s.w.to(x.dtype), None if s.b is None else s.b.to(x.dtype), padding=s.pad)


def up2(x):
    return x.repeat_interleave(2, 2).repeat_interleave(2, 3)


class VaeStage(torch.nn.Module):
    """mil.VaePart: stage i of the decoder (fp16), input x (1, C, H, W)."""

    def __init__(s, v, ckpt, stage, side=512):
        super().__init__()
        ops, s.C, s.Hh, s.Ww = vae_stages(v, side, side)[stage]
        s.ops = [op[0] for op in ops]
        s.mods = torch.nn.ModuleList()
        conv = lambda path: Conv(ckpt(path + ".weight"), ckpt(path + ".bias"))
        g = lambda path: ckpt(path).float().reshape(1, -1, 1, 1)
        for kind, path, cin, cout, ft in ops:
            m = torch.nn.Module()
            if kind == "in":
                Z = v["z_dim"]
                pq, pb = ckpt("vae.post_quant_conv.weight").float().reshape(Z, Z), ckpt("vae.post_quant_conv.bias").float()
                std, mean = torch.tensor(v["latents_std"]), torch.tensor(v["latents_mean"])
                m.pre = Conv((pq * std[None]).reshape(Z, Z, 1, 1), pb + pq @ mean)
                m.conv = conv("vae.decoder.conv_in")
            elif kind == "res":
                m.register_buffer("n1", g(path + ".norm1.gamma")); m.register_buffer("n2", g(path + ".norm2.gamma"))
                m.c1, m.c2 = conv(path + ".conv1"), conv(path + ".conv2")
                m.sc = conv(path + ".conv_shortcut") if cin != cout else None
            elif kind == "attn":
                C = ckpt(path + ".proj.bias").shape[0]
                m.register_buffer("n", g(path + ".norm.gamma"))
                w, b = ckpt(path + ".to_qkv.weight").float().reshape(3 * C, C, 1, 1), ckpt(path + ".to_qkv.bias").float()
                m.q, m.k, m.v = (Conv(w[j * C:(j + 1) * C], b[j * C:(j + 1) * C]) for j in range(3))
                m.proj = conv(path + ".proj")
            elif kind == "up":
                m.conv = conv(path + ".upsampler.resample.1")
                src = dup_source(cin, cout, ft)
                same = all(src[co * 4 + p] == src[co * 4] for co in range(cout) for p in range(1, 4))
                ident = cin == cout and all(src[co * 4] == co for co in range(cout))
                m.same, m.ident = same, ident
                if same and not ident:
                    sel = torch.zeros(cout, cin); sel[torch.arange(cout), torch.tensor(src[::4])] = 1
                    m.dup = Conv(sel.reshape(cout, cin, 1, 1), None)
                elif not same:
                    sel = torch.zeros(4 * cout, cin); sel[torch.arange(4 * cout), torch.tensor(src)] = 1
                    m.dup = Conv(sel.reshape(4 * cout, cin, 1, 1), None)
            elif kind == "head":
                m.register_buffer("n", g("vae.decoder.norm_out.gamma"))
                m.conv = conv("vae.decoder.conv_out")
            s.mods.append(m)

    def forward(s, x):
        inp = x
        for kind, m in zip(s.ops, s.mods):
            if kind == "in":
                x = m.conv(m.pre(x))
            elif kind == "res":
                h = x if m.sc is None else m.sc(x)
                a = m.c1(torch.nn.functional.silu(chan_norm(x, m.n1.to(x.dtype))))
                a = m.c2(torch.nn.functional.silu(chan_norm(a, m.n2.to(x.dtype))))
                x = a + h
            elif kind == "attn":
                _, C, H, W = x.shape
                h = chan_norm(x, m.n.to(x.dtype))
                q, k, v = (f(h).reshape(1, C, H * W).transpose(1, 2).reshape(1, 1, H * W, C) for f in (m.q, m.k, m.v))
                sc = torch.matmul(q, k.transpose(-1, -2)) * (1 / math.sqrt(C))
                o = torch.matmul(softmax(sc), v).reshape(1, H * W, C).transpose(1, 2).reshape(1, C, H, W)
                x = x + m.proj(o)
            elif kind == "up":
                if m.same:
                    d = up2(inp if m.ident else m.dup(inp))
                else:
                    d = torch.nn.functional.pixel_shuffle(m.dup(inp), 2)
                x = m.conv(up2(x)) + d
            elif kind == "head":
                x = m.conv(torch.nn.functional.silu(chan_norm(x, m.n.to(x.dtype))))
        return x


def checkpoints(root, text):
    # names as mil.Weights: "dit." the transformer's, "text." Qwen3-VL-8B's language model's
    srcs = []
    t = Path(root) / "transformer"
    for k, f in json.load(open(t / "diffusion_pytorch_model.safetensors.index.json"))["weight_map"].items():
        srcs.append(("dit." + k, str(t / f), k))
    vf = Path(root) / "vae" / "diffusion_pytorch_model.safetensors"
    with safe_open(str(vf), "pt") as st:
        for k in st.keys():
            srcs.append(("vae." + k, str(vf), k))
    if text:
        f = str(Path(text) / "text_encoder" / "model.safetensors")
        with safe_open(f, "pt") as st:
            for k in st.keys():
                if k.startswith("model.language_model."):
                    srcs.append(("text." + k[len("model.language_model."):], f, k))
    where = {n: (f, k) for n, f, k in srcs}
    def get(name):
        f, k = where[name]
        with safe_open(f, "pt") as st:
            return st.get_tensor(k)
    return get


def inputs(kind, cfg, first, count, final, side=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    E, D = cfg["num_attention_heads"] * cfg["attention_head_dim"], cfg["attention_head_dim"]
    mod = torch.cat([1 + 0.1 * r(1, 1, E), 0.1 * r(1, 1, E), 1 + 0.1 * r(1, 1, E), 0.1 * r(1, 1, E)], -1)
    if kind == "text":
        return ["x"], [r(1, TEXT_ROWS, 4096) * 0.25]
    if kind == "vae":
        v = json.load(open(Path(ROOT) / "vae" / "config.json"))
        _, C, H, W = vae_stages(v, side, side)[first]
        return ["x"], [r(1, C, H, W) * (1 if first == 0 else 0.5)]
    if kind == "dittext":
        return ["x", "mod"], [r(1, TEXT_ROWS, E) * 0.5, mod]
    n = (side // 16) ** 2
    x = r(1, n, cfg["in_channels"]) if first == 0 else r(1, n, E) * 0.5
    ang = torch.rand(1, n, 1, D // 2, generator=g) * 2 * math.pi
    cos, sin = torch.cat([ang.cos()] * 2, -1), torch.cat([ang.sin()] * 2, -1)
    tk, tv = r(1, count * TEXT_ROWS, E) * 0.3, r(1, count * TEXT_ROWS, E)
    kbias = torch.zeros(1, 1, 1, n + TEXT_ROWS)
    kbias[..., n + 40:] = -10000  # 40 text rows that count
    names, ins = ["x", "mod", "cos", "sin", "tk", "tv", "kbias"], [x, mod, cos, sin, tk, tv, kbias]
    if final:
        names.append("scale"); ins.append(1 + 0.1 * r(1, 1, E))
    return names, ins


def build(a, cfg, ckpt, luts, dense):
    if a.kind == "dit":
        return Blocks(cfg, ckpt, luts, a.first, a.count, a.final, dense)
    if a.kind == "vae":
        return VaeStage(json.load(open(Path(a.root) / "vae" / "config.json")), ckpt, a.first)
    if a.kind == "dittext":
        return TextBlocks(cfg, ckpt, luts, a.first, a.count, dense)
    tcfg = json.load(open(Path(a.text) / "text_encoder" / "config.json"))["text_config"]
    return TextGroup(tcfg, cfg, ckpt, luts, a.first, a.count, a.final, dense)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["dit", "dittext", "text", "vae"])
    ap.add_argument("root"); ap.add_argument("luts"); ap.add_argument("out")
    ap.add_argument("--text", default="")
    ap.add_argument("--first", type=int, default=0); ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--final", action="store_true"); ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    global ROOT
    ROOT = a.root
    import coreai_torch as ct
    cfg = json.load(open(Path(a.root) / "transformer" / "config.json"))
    ckpt = checkpoints(a.root, a.text)
    names, ins = inputs(a.kind, cfg, a.first, a.count, a.final)
    ins = [v.half() for v in ins]
    import contextlib
    with (safe_open(a.luts, "pt") if a.luts != "-" else contextlib.nullcontext()) as luts:
        t = time.time()
        m = build(a, cfg, ckpt, luts, dense=False).half().eval()
        ep = torch.export.export(m, tuple(ins)).run_decompositions(ct.get_decomp_table())
        prog = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE).add_exported_program(ep).to_coreai()
        prog.save_asset(Path(a.out))
        print(f"saved {a.out} in {time.time() - t:.1f}s")
        del m, ep, prog
        if a.check:
            ref = build(a, cfg, ckpt, luts, dense=True).float().eval()
            with torch.no_grad():
                want = ref(*[v.float() for v in ins]).numpy().ravel().astype(np.float64)
            del ref
    if a.check:
        import asyncio
        from coreai import runtime as rt
        async def run():
            opts = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.neural_engine())
            t = time.time(); mdl = await rt.AIModel.load(a.out, opts); f = mdl.load_function(mdl.function_names[0])
            tl = time.time() - t
            d = f.desc
            feed = {nm: rt.NDArray(ins[i].numpy().astype(np.float16).reshape(d.input_descriptor(nm).shape)) for i, nm in enumerate(d.input_names)}
            t = time.time(); out = await f(feed); tc1 = time.time() - t
            t = time.time(); out = await f(feed); tc2 = time.time() - t
            y = (list(out.values())[0] if isinstance(out, dict) else out).numpy().astype(np.float64).ravel()
            print(json.dumps({"inputs": list(d.input_names), "load": round(tl, 2), "call1": round(tc1, 3), "call2": round(tc2, 3)}))
            return y
        got = asyncio.run(run())
        cos = got @ want / (np.linalg.norm(got) * np.linalg.norm(want))
        rel = np.linalg.norm(got - want) / np.linalg.norm(want)
        print(f"ANE vs torch float32: cosine {cos:.6f}, relative RMS {rel:.4f}, nonfinite {int((~np.isfinite(got)).sum())}")


if __name__ == "__main__":
    main()
