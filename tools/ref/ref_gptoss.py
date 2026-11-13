"""float32 reference of gpt-oss (``gpt_oss``: openai/gpt-oss-20b), a decoder layer at a time.

gpt-oss-20b is 21 B parameters: 84 GB in float32, more than this machine's 60 GB. Its experts
are MXFP4 in the checkpoint (``*_blocks`` and ``*_scales``: 4-bit e2m1 codes, a power of two per
32 in-features of an out-feature), the rest bf16. This reads one layer's weights at a time (the
experts dequantized as transformers' ``convert_moe_packed_tensors`` does, exactly), runs every
sequence through it, and drops it, so that about one layer (3.3 GB in float32) is held.

The forward is written here from transformers' modeling_gpt_oss.py (5.x): RMSNorm in float32
times its weight; q, k, v and o projections with biases; YaRN RoPE (rotate_half, the table's
``attention_scaling`` on cos and sin, transformers' own init); grouped attention with a learned
sink logit per head in the softmax (dropped from the output), a 128-position window on the
``sliding_attention`` layers; the router (a linear with a bias), the top 4 of 32 experts and a
softmax over their logits; each expert's gate and up interleaved (even and odd columns), the
gate clamped at 7 and the up to [-7, 7], ``(up + 1) * gate * sigmoid(1.702 gate)``, down with its
bias, times the routing weight. ``check`` compares it with transformers' modules.

    PY=~/.cache/hwnpu/dev/ref-venv/bin/python
    $PY tools/ref/ref_gptoss.py check MODEL_DIR [--layers 2]        # against transformers' modules
    $PY tools/ref/ref_gptoss.py ref MODEL_DIR WORK [--tokens 32] [--steps 3]   # an export's check/
    $PY tools/ref/ref_gptoss.py logits MODEL_DIR IDS OUT            # every position's logits
    $PY tools/ref/ref_gptoss.py sweep MODEL_DIR OUT.json --variant NAME:PART=FMT,... ...

``ref`` writes iron_qwen35_check.py's files (ref.json, ref_K.bin: the logits after the prompt
and after each greedy step), which ironlm reads from an export's check/. ``logits`` reads token
ids (a uint32 file, as ``ironlm -forced`` writes RUN.bin.ids, or a JSON list) and writes float32
rows, one per position. ``sweep`` measures KL(checkpoint || variant) over new-model-checklist's
texts (5.1) for parts rounded to other formats (5.2): attn, router, experts, head as
bf16 (the checkpoint), q8, q6, q4 (iron_q4's), exact (MXFP4 kept, experts only).
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "amd"))

FP4 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
TEXT = ("The Neural Engine runs small language models at low power. hwnpu is a Go library for CPU "
        "integrated NPUs: Apple, Intel and AMD. 日本の首都は東京です。")
# What the harmony chat format puts before a user's text: a prompt starts as the model sees it.
USER = "<|start|>user<|message|>"


class Checkpoint:
    """The checkpoint's tensors by name, in float32; the experts dequantized."""

    def __init__(self, model_dir):
        self.dir = Path(model_dir)
        self.config = json.loads((self.dir / "config.json").read_text())
        index = self.dir / "model.safetensors.index.json"
        if index.exists():
            self.where = json.loads(index.read_text())["weight_map"]
        else:
            with safe_open(str(self.dir / "model.safetensors"), framework="pt") as f:
                self.where = dict.fromkeys(f.keys(), "model.safetensors")
        self.files = {}

    def raw(self, name):
        f = self.where[name]
        if f not in self.files:
            self.files[f] = safe_open(str(self.dir / f), framework="pt")
        return self.files[f].get_tensor(name)

    def get(self, name):
        return self.raw(name).float()

    def mxfp4(self, prefix):
        """``prefix``_blocks and _scales as (experts, out, in) float32, w = e2m1 * 2^(scale - 127)."""
        blocks, scales = self.raw(prefix + "_blocks"), self.raw(prefix + "_scales")
        E, N, G, B = blocks.shape
        lo, hi = FP4[(blocks & 15).long()], FP4[(blocks >> 4).long()]
        w = torch.stack([lo, hi], -1).reshape(E, N, G, 2 * B)
        w = torch.ldexp(w, (scales.to(torch.int32) - 127)[..., None])
        return w.reshape(E, N, G * 2 * B)

    def mxfp4_codes(self, prefix):
        """The raw codes (experts, out, in) uint8 0..15 and exponents (experts, out, in / 32)."""
        blocks, scales = self.raw(prefix + "_blocks"), self.raw(prefix + "_scales")
        E, N, G, B = blocks.shape
        codes = torch.stack([blocks & 15, blocks >> 4], -1).reshape(E, N, G * 2 * B)
        return codes, scales

    def layer(self, i):
        p = f"model.layers.{i}."
        w = {k: self.get(p + k) for k in (
            "input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.sinks",
            "self_attn.q_proj.weight", "self_attn.q_proj.bias", "self_attn.k_proj.weight",
            "self_attn.k_proj.bias", "self_attn.v_proj.weight", "self_attn.v_proj.bias",
            "self_attn.o_proj.weight", "self_attn.o_proj.bias", "mlp.router.weight", "mlp.router.bias",
            "mlp.experts.gate_up_proj_bias", "mlp.experts.down_proj_bias")}
        w["mlp.experts.gate_up_proj"] = self.mxfp4(p + "mlp.experts.gate_up_proj")  # (E, 2I, hidden)
        w["mlp.experts.down_proj"] = self.mxfp4(p + "mlp.experts.down_proj")  # (E, hidden, I)
        return w


def yarn(config):
    """transformers' YaRN inverse frequencies and attention_scaling for this config."""
    from transformers import GptOssConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    c = GptOssConfig(**{k: v for k, v in config.items() if k != "quantization_config"})
    inv_freq, scaling = ROPE_INIT_FUNCTIONS[c.rope_parameters["rope_type"]](c)
    return inv_freq.float(), float(scaling)


class GptOss:
    """The model's forward over sequences, a layer at a time (module docstring)."""

    def __init__(self, ck: Checkpoint):
        c = ck.config
        if c.get("model_type") != "gpt_oss":
            raise SystemExit(f"model_type {c.get('model_type')} is not gpt_oss")
        self.ck, self.c = ck, c
        self.H, self.G, self.D = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.E, self.top = c["num_local_experts"], c["num_experts_per_tok"]
        self.eps, self.window = c["rms_norm_eps"], c["sliding_window"]
        self.limit, self.alpha = c.get("swiglu_limit", 7.0), 1.702
        self.layers = c["num_hidden_layers"]
        self.inv_freq, self.rope_scale = yarn(c)
        self._embed = None

    def embed(self, ids):
        if self._embed is None:
            self._embed = self.ck.raw("model.embed_tokens.weight")  # bf16, rows read as needed
        return self._embed[torch.as_tensor(ids)].float()

    def norm(self, x, w):
        v = x.pow(2).mean(-1, keepdim=True)
        return w * (x * torch.rsqrt(v + self.eps))

    def rope(self, n, start=0):
        pos = torch.arange(start, start + n, dtype=torch.float32)
        f = pos[:, None] * self.inv_freq[None]
        return torch.cos(f) * self.rope_scale, torch.sin(f) * self.rope_scale

    @staticmethod
    def rotate(x, cos, sin):  # x (n, heads, D)
        a, b = x.chunk(2, -1)
        cos, sin = cos[:, None], sin[:, None]
        return torch.cat([a * cos - b * sin, b * cos + a * sin], -1)

    def attention(self, i, x, w, cache=None):
        """Layer i's self-attention of x (n, hidden) normed. cache: (k, v) of earlier positions
        (rows, G, D) or None; returns the output and the new (k, v)."""
        n, H, G, D = x.shape[0], self.H, self.G, self.D
        q = (x @ w["self_attn.q_proj.weight"].T + w["self_attn.q_proj.bias"]).view(n, H, D)
        k = (x @ w["self_attn.k_proj.weight"].T + w["self_attn.k_proj.bias"]).view(n, G, D)
        v = (x @ w["self_attn.v_proj.weight"].T + w["self_attn.v_proj.bias"]).view(n, G, D)
        start = 0 if cache is None else cache[0].shape[0]
        cos, sin = self.rope(n, start)
        q, k = self.rotate(q, cos, sin), self.rotate(k, cos, sin)
        if cache is not None:
            k, v = torch.cat([cache[0], k]), torch.cat([cache[1], v])
        T = k.shape[0]
        kk = k.repeat_interleave(H // G, 1)  # (T, H, D): repeat_kv's order
        vv = v.repeat_interleave(H // G, 1)
        s = torch.einsum("nhd,thd->hnt", q, kk) * D ** -0.5
        qpos = torch.arange(start, start + n)[:, None]
        kpos = torch.arange(T)[None]
        keep = kpos <= qpos
        if self.c["layer_types"][i] == "sliding_attention":
            keep &= qpos - kpos < self.window
        s = s.masked_fill(~keep[None], float("-inf"))
        sink = w["self_attn.sinks"].view(H, 1, 1).expand(H, n, 1)
        full = torch.cat([s, sink], -1)
        full = full - full.max(-1, keepdim=True).values
        p = torch.softmax(full, -1)[..., :-1]
        o = torch.einsum("hnt,thd->nhd", p, vv).reshape(n, H * D)
        return o @ w["self_attn.o_proj.weight"].T + w["self_attn.o_proj.bias"], (k, v)

    def route(self, x, w):
        logits = x @ w["mlp.router.weight"].T + w["mlp.router.bias"]
        top, ids = torch.topk(logits, self.top, dim=-1)
        return ids, torch.softmax(top, -1), logits

    def glu(self, gu):
        gate, up = gu[..., ::2], gu[..., 1::2]
        gate = gate.clamp(max=self.limit)
        up = up.clamp(-self.limit, self.limit)
        return (up + 1) * (gate * torch.sigmoid(gate * self.alpha))

    def experts(self, x, w, ids, rw):
        out = torch.zeros_like(x)
        gu, down = w["mlp.experts.gate_up_proj"], w["mlp.experts.down_proj"]
        gb, db = w["mlp.experts.gate_up_proj_bias"], w["mlp.experts.down_proj_bias"]
        for e in torch.unique(ids).tolist():
            tok, slot = torch.where(ids == e)
            a = self.glu(x[tok] @ gu[e].T + gb[e])
            y = a @ down[e].T + db[e]
            out.index_add_(0, tok, y * rw[tok, slot, None])
        return out

    def layer(self, i, h, w, cache=None):
        a, kv = self.attention(i, self.norm(h, w["input_layernorm.weight"]), w, cache)
        h = h + a
        x = self.norm(h, w["post_attention_layernorm.weight"])
        ids, rw, _ = self.route(x, w)
        return h + self.experts(x, w, ids, rw), kv

    def head(self, h, transform=None):
        nw = self.ck.get("model.norm.weight")
        lm = self.ck.get("lm_head.weight")
        if transform is not None:
            lm = transform("head", lm)
        return self.norm(h, nw) @ lm.T

    def run(self, seqs, variants=None, hidden=None, last=None):
        """Logits of each sequence (a causal pass each) for each variant: {name: [logits]}.

        variants maps a name to a function (part, weight) -> weight that rounds the parts it
        names (None: the checkpoint's). hidden, a list, gets each layer's output (variant
        order, then sequence). last: the layers run (all if None)."""
        variants = variants or {"ref": None}
        hs = {v: [self.embed(s) for s in seqs] for v in variants}
        for i in range(last if last is not None else self.layers):
            t0 = time.time()
            w0 = self.ck.layer(i)
            for name, tf in variants.items():
                w = w0 if tf is None else round_layer(w0, tf)
                hs[name] = [self.layer(i, h, w)[0] for h in hs[name]]
                if hidden is not None:
                    hidden.append([h.clone() for h in hs[name]])
            print(f"  layer {i}: {time.time() - t0:.1f} s", file=sys.stderr, flush=True)
        return {v: [self.head(h, variants[v]) for h in hs[v]] for v in variants}

    def generate(self, ids, steps):
        """Greedy steps after ids with a cache: the logits after the prompt and after each step."""
        outs, kv, h = [], [None] * self.layers, self.embed(ids)
        toks = list(ids)
        for k in range(steps + 1):
            for i in range(self.layers):
                w = self.ck.layer(i)
                h, kv[i] = self.layer(i, h, w, kv[i])
            logits = self.head(h[-1:])[0]
            outs.append(logits)
            toks.append(int(logits.argmax()))
            h = self.embed([toks[-1]])
        return outs, toks


PARTS = {
    "attn": ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
             "self_attn.o_proj.weight"),
    "router": ("mlp.router.weight",),
    "experts": ("mlp.experts.gate_up_proj", "mlp.experts.down_proj"),
}


def rounding(fmt):
    """w (out, in) float32 rounded to fmt and back, as the export pads and quantizes it (iron_q4:
    out to a multiple of 64, in to a multiple of 512, with zeros)."""
    if fmt in ("bf16", "exact"):
        return None
    import iron_q4

    def f(w):
        N, K = w.shape
        Np, Kp = -(-N // 64) * 64, -(-K // 512) * 512
        p = np.zeros((Np, Kp), np.float32)
        p[:N, :K] = w
        q = iron_q4.quantize_fmt(p, fmt)
        return iron_q4.dequantize_fmt(q, Np, Kp, fmt)[:N, :K]

    return f


def variant(spec):
    """'attn=q8,experts=q4,head=q4' -> a function (part, weight) -> weight (Experts as
    (E, out, in): each expert's matrix rounded alone)."""
    fmts = dict(p.split("=") for p in spec.split(",") if p)
    for part in fmts:
        if part not in (*PARTS, "head"):
            raise SystemExit(f"part {part}: one of {', '.join([*PARTS, 'head'])}")
    fns = {part: rounding(f) for part, f in fmts.items()}

    def tf(part, w):
        fn = fns.get(part)
        if fn is None:
            return w
        x = w.numpy()
        if x.ndim == 3:
            return torch.from_numpy(np.stack([fn(m) for m in x]))
        return torch.from_numpy(fn(x))

    return tf


def round_layer(w, tf):
    out = dict(w)
    for part, names in PARTS.items():
        for n in names:
            out[n] = tf(part, w[n])
    return out


def tokenizer(model_dir):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))


def kl_stats(ref, q):
    a = torch.log_softmax(ref.double(), -1)
    b = torch.log_softmax(q.double(), -1)
    kl = (a.exp() * (a - b)).sum(-1).numpy()
    top = float((ref.argmax(-1) == q.argmax(-1)).float().mean())
    return dict(mean=float(kl.mean()), median=float(np.median(kl)), p95=float(np.percentile(kl, 95)), top1=top)


# new-model-checklist 5.1's texts: the README (not here: it is hwnpu's), GPL-3, wiki40b-ja,
# wikitext-2, Go code
def texts(length):
    root = HERE.parents[1]
    out = {}
    readme = (root / "README.md").read_text(encoding="utf-8")
    out["readme"] = readme
    for name, path in (("gpl", root / "LICENSE"), ("wiki-ja", Path.home() / "models/calib/wiki40b-ja-eval.txt"),
                       ("wikitext", Path.home() / "models/calib/wikitext2-test.txt"),
                       ("go", root / "internal/chat/engine.go")):
        if path.exists():
            out[name] = path.read_text(encoding="utf-8")
    return out


def cmd_check(a):
    """Our forward against transformers' GptOss modules on the same float32 weights."""
    from transformers import GptOssConfig
    from transformers.models.gpt_oss import modeling_gpt_oss as m

    ck = Checkpoint(a.model_dir)
    g = GptOss(ck)
    cfg = {k: v for k, v in ck.config.items() if k != "quantization_config"}
    cfg["num_hidden_layers"] = a.layers
    cfg["layer_types"] = cfg["layer_types"][: a.layers]
    hc = GptOssConfig(**cfg)
    hc._attn_implementation = "eager"
    tok = tokenizer(a.model_dir)
    ids = tok.encode(USER + a.text, add_special_tokens=False).ids[: a.tokens]
    if len(ids) < a.tokens:
        long = (a.text + " ") * (a.tokens // len(ids) + 2)
        ids = tok.encode(USER + long, add_special_tokens=False).ids[: a.tokens]
    print(f"{len(ids)} tokens, {a.layers} layers")
    with torch.device("meta"):
        model = m.GptOssForCausalLM(hc)
    model.model.rotary_emb = m.GptOssRotaryEmbedding(config=hc)
    model.eval()

    def put(module, sd):
        for n, p in list(module.named_parameters(recurse=True)):
            owner, _, leaf = n.rpartition(".")
            mm = module.get_submodule(owner) if owner else module
            mm._parameters[leaf] = torch.nn.Parameter(sd[n], requires_grad=False)

    put(model.model.embed_tokens, {"weight": ck.get("model.embed_tokens.weight")})
    put(model.model.norm, {"weight": ck.get("model.norm.weight")})
    put(model.lm_head, {"weight": ck.get("lm_head.weight")})
    for i in range(a.layers):
        w = ck.layer(i)
        sd = dict(w)
        # transformers keeps the experts' weights (E, in, out)
        sd["mlp.experts.gate_up_proj"] = w["mlp.experts.gate_up_proj"].transpose(1, 2).contiguous()
        sd["mlp.experts.down_proj"] = w["mlp.experts.down_proj"].transpose(1, 2).contiguous()
        put(model.model.layers[i], sd)
    hidden = {}
    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(lambda mod, args, out, i=i: hidden.__setitem__(i, out[0] if isinstance(out, tuple) else out))
    with torch.no_grad():
        want = model(torch.tensor([ids])).logits[0]
        mine_h = []
        got = g.run([ids], hidden=mine_h, last=a.layers)["ref"][0]
    worst = 0.0
    for i in range(a.layers):
        x, y = mine_h[i][0], hidden[i][0]
        r = float((x - y).norm() / y.norm())
        worst = max(worst, r)
        print(f"layer {i}: relative error {r:.2e}, max |h| {float(y.abs().max()):.1f}")
    r = float((got - want).norm() / want.norm())
    print(f"logits: relative error {r:.2e}, top-1 {float((got.argmax(-1) == want.argmax(-1)).float().mean()):.3f}")
    print("PASS" if max(worst, r) < 1e-4 else "FAIL")


def cmd_ref(a):
    ck = Checkpoint(a.model_dir)
    g = GptOss(ck)
    tok = tokenizer(a.model_dir)
    ids = tok.encode(USER + a.text, add_special_tokens=False).ids[: a.tokens]
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        outs, toks = g.generate(ids, a.steps)
    calls = []
    for k, logits in enumerate(outs):
        fn = f"ref_{k}.bin"
        logits.numpy().astype(np.float32).tofile(work / fn)
        calls.append(dict(tokens=len(ids) + k, logits=fn, top=int(logits.argmax())))
    (work / "ref.json").write_text(json.dumps(dict(model=str(a.model_dir), prompt=ids, calls=calls)) + "\n")
    print(f"{work}/ref.json: a prompt of {len(ids)} tokens and {a.steps} steps, tops {[c['top'] for c in calls]}")
    print(repr(tok.decode(toks[len(ids):], skip_special_tokens=False)))


def read_ids(path):
    p = Path(path)
    if p.suffix == ".json":
        return json.loads(p.read_text())
    return np.fromfile(p, np.uint32).astype(np.int64).tolist()


def cmd_logits(a):
    g = GptOss(Checkpoint(a.model_dir))
    ids = read_ids(a.ids)
    with torch.no_grad():
        out = g.run([ids])["ref"][0]
    out.numpy().astype(np.float32).tofile(a.out)
    print(f"{a.out}: {len(ids)} rows of {out.shape[1]}")


def cmd_forced(a):
    """KL(checkpoint || NPU) of ``ironlm -forced`` runs: RUN.bin rows are the logits after the
    first --split tokens of RUN.bin.ids, then after each token more."""
    g = GptOss(Checkpoint(a.model_dir))
    ids = {r: read_ids(r + ".ids") for r in a.runs}
    seqs = list({tuple(v): v for v in ids.values()}.values())
    with torch.no_grad():
        ref = g.run(seqs)["ref"]
    by = {tuple(s): r for s, r in zip(seqs, ref)}
    out = {}
    for run in a.runs:
        r = by[tuple(ids[run])]
        n = len(ids[run])
        want = r[a.split - 1 : n - 1]
        got = torch.from_numpy(np.fromfile(run, np.float32).reshape(-1, r.shape[1] if np.fromfile(run, np.float32).size % r.shape[1] == 0 else -1))
        got = got[:, : r.shape[1]][: want.shape[0]]
        st = kl_stats(want[: got.shape[0]], got)
        out[run] = st
        print(f"{run}: KL mean {st['mean']:.4f} median {st['median']:.4f} p95 {st['p95']:.4f}, top-1 {st['top1']:.3f} over {got.shape[0]} positions")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


def cmd_sweep(a):
    ck = Checkpoint(a.model_dir)
    g = GptOss(ck)
    tok = tokenizer(a.model_dir)
    seqs = {}
    for name, text in texts(a.length).items():
        ids = tok.encode(USER + text, add_special_tokens=False).ids[: a.length]
        seqs[name] = ids
    variants = {"ref": None}
    for v in a.variant:
        name, _, spec = v.partition(":")
        variants[name] = variant(spec)
    t0 = time.time()
    with torch.no_grad():
        out = g.run(list(seqs.values()), variants)
    result = dict(length=a.length, texts=list(seqs), variants={})
    for name in variants:
        if name == "ref":
            continue
        rows = {}
        for k, t in enumerate(seqs):
            rows[t] = kl_stats(out["ref"][k], out[name][k])
        result["variants"][name] = rows
        print(name, " ".join(f"{t} {r['mean']:.4f}/{r['top1']:.3f}" for t, r in rows.items()))
    result["seconds"] = time.time() - t0
    Path(a.out).write_text(json.dumps(result, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("model_dir")
    c.add_argument("--layers", type=int, default=2)
    c.add_argument("--tokens", type=int, default=160, help="past the 128-position window")
    c.add_argument("--text", default=TEXT)
    r = sub.add_parser("ref")
    r.add_argument("model_dir")
    r.add_argument("work")
    r.add_argument("--tokens", type=int, default=32)
    r.add_argument("--steps", type=int, default=3)
    r.add_argument("--text", default=TEXT)
    lg = sub.add_parser("logits")
    lg.add_argument("model_dir")
    lg.add_argument("ids")
    lg.add_argument("out")
    fo = sub.add_parser("forced")
    fo.add_argument("model_dir")
    fo.add_argument("runs", nargs="+")
    fo.add_argument("--split", type=int, default=256)
    fo.add_argument("--out")
    s = sub.add_parser("sweep")
    s.add_argument("model_dir")
    s.add_argument("out")
    s.add_argument("--variant", action="append", default=[], help="NAME:attn=q8,experts=q4,head=q4")
    s.add_argument("--length", type=int, default=400)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    {"check": cmd_check, "ref": cmd_ref, "logits": cmd_logits, "sweep": cmd_sweep, "forced": cmd_forced}[a.cmd](a)


if __name__ == "__main__":
    main()
