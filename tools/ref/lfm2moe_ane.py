"""LFM2-MoE (``lfm2_moe``: LiquidAI/LFM2-8B-A1B) with its experts on the Neural Engine.

The model's own layers (the short convolutions, the attention, the two dense layers, the
embedding and the head) run here in float32 with transformers' modules; each MoE block is either
the reference (``ref``: the router and the chosen experts from the bf16 checkpoint in float32,
transformers' Lfm2MoeSparseMoeBlock written out, the experts read on demand so the 15 GB of
experts are never all in memory) or ``aicheck moeserve`` (``ane``: the router and the experts in
int8 on the Neural Engine, every MoE layer's experts held, docs/handoff-ane-gather.md 1b).

    PY=~/models/ref-venv/bin/python
    go build -o /tmp/aicheck ./tools/coreai/aicheck
    $PY -I tools/ref/lfm2moe_ane.py run ref MODEL_DIR WORK [--tokens 48]
    $PY -I tools/ref/lfm2moe_ane.py run ane MODEL_DIR WORK --server /tmp/aicheck
    $PY -I tools/ref/lfm2moe_ane.py compare WORK

``run`` generates greedily from the prompt (the text and the time a token), then runs the
prompt and the reference's generated tokens (WORK/ref_ids.json, written by ``run ref``) in one
forward, writing every position's logits (WORK/MODE_logits.npy) and the experts each MoE layer
chose (WORK/MODE_experts.npy). ``compare`` prints the KL(ref || ane) and top-1 agreement of the
generated positions and how many of each layer's experts agree.
"""

import argparse
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from torch import nn
from transformers import AutoTokenizer
from transformers.models.lfm2_moe.configuration_lfm2_moe import Lfm2MoeConfig
from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeForCausalLM, Lfm2MoeRotaryEmbedding

PROMPT = "Explain in three sentences why the sky is blue, then say which city is the capital of Japan."


class Checkpoint:
    def __init__(self, d):
        self.dir = Path(d)
        self.map = json.loads((self.dir / "model.safetensors.index.json").read_text())["weight_map"]
        self.files = {}

    def get(self, name):
        f = self.map[name]
        if f not in self.files:
            self.files[f] = safe_open(self.dir / f, framework="pt")
        return self.files[f].get_tensor(name)


class Choices:
    """The experts each MoE layer chose, by layer, for the forward being recorded."""

    def __init__(self):
        self.on = False
        self.by_layer = {}

    def add(self, layer, chosen):
        if self.on:
            self.by_layer.setdefault(layer, []).append(np.asarray(chosen, dtype=np.int32))


class RefMoE(nn.Module):
    def __init__(self, ck, cfg, layer, choices):
        super().__init__()
        p = f"model.layers.{layer}.feed_forward."
        self.ck, self.p, self.layer, self.choices = ck, p, layer, choices
        self.gate = ck.get(p + "gate.weight").float()
        self.bias = ck.get(p + "expert_bias").float() if cfg.use_expert_bias else None
        self.k, self.norm, self.scale = cfg.num_experts_per_tok, cfg.norm_topk_prob, cfg.routed_scaling_factor

    def forward(self, h):
        shape = h.shape
        x = h.reshape(-1, shape[-1]).float()
        w = (x @ self.gate.T).sigmoid()
        s = w + self.bias if self.bias is not None else w
        sel = torch.topk(s, self.k, dim=-1).indices
        rw = torch.gather(w, 1, sel)
        if self.norm:
            rw = rw / (rw.sum(-1, keepdim=True) + 1e-6)
        rw = rw * self.scale
        out = torch.zeros_like(x)
        for e in sel.unique().tolist():
            rows, pos = torch.where(sel == e)
            q = f"{self.p}experts.{e}."
            w1, w3, w2 = (self.ck.get(q + n + ".weight").float() for n in ("w1", "w3", "w2"))
            xe = x[rows]
            y = (nn.functional.silu(xe @ w1.T) * (xe @ w3.T)) @ w2.T
            out.index_add_(0, rows, y * rw[rows, pos, None])
        self.choices.add(self.layer, sel.numpy())
        return out.reshape(shape).to(h.dtype)


class Server:
    def __init__(self, exe, model_dir, flags=()):
        self.p = subprocess.Popen([exe, "moeserve", *flags, str(model_dir)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def call(self, layer, x, k):
        rows, hidden = x.shape
        self.p.stdin.write(struct.pack("<II", layer, rows) + x.astype("<f4").tobytes())
        self.p.stdin.flush()
        y = np.frombuffer(self.read(rows * hidden * 4), dtype="<f4").reshape(rows, hidden)
        chosen = np.frombuffer(self.read(rows * k * 4), dtype="<i4").reshape(rows, k)
        return y, chosen

    def read(self, n):
        b = bytearray()
        while len(b) < n:
            c = self.p.stdout.read(n - len(b))
            if not c:
                raise RuntimeError("moeserve ended")
            b += c
        return bytes(b)

    def stats(self):
        self.p.stdin.write(struct.pack("<II", 0xFFFFFFFF, 0))
        self.p.stdin.flush()


class AneMoE(nn.Module):
    def __init__(self, server, cfg, layer, choices):
        super().__init__()
        self.server, self.layer, self.k, self.choices = server, layer, cfg.num_experts_per_tok, choices

    def forward(self, h):
        shape = h.shape
        y, chosen = self.server.call(self.layer, h.reshape(-1, shape[-1]).float().numpy(), self.k)
        self.choices.add(self.layer, chosen)
        return torch.from_numpy(y.copy()).reshape(shape).to(h.dtype)


def build(mode, model_dir, server_exe):
    cfg = Lfm2MoeConfig.from_pretrained(model_dir)
    ck = Checkpoint(model_dir)
    with torch.device("meta"):
        model = Lfm2MoeForCausalLM(cfg)
    choices = Choices()
    server = None
    if mode == "ane":
        server = Server(server_exe, model_dir, SERVER_FLAGS)
    for l, layer in enumerate(model.model.layers):
        if l >= cfg.num_dense_layers:
            layer.feed_forward = RefMoE(ck, cfg, l, choices) if mode == "ref" else AneMoE(server, cfg, l, choices)
    sd = {}
    for name, t in model.state_dict().items():
        if t.device.type != "meta":
            continue
        key = name if name in ck.map else None
        if key is None and name == "lm_head.weight":
            key = "model.embed_tokens.weight"
        if key is None:
            raise KeyError(name)
        sd[name] = ck.get(key).float()
    model.load_state_dict(sd, strict=False, assign=True)
    model.model.pos_emb = Lfm2MoeRotaryEmbedding(config=cfg)
    left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.device.type == "meta"]
    if left:
        raise RuntimeError(f"still on meta: {left[:5]}")
    model.eval()
    return model, cfg, choices, server


SERVER_FLAGS = ()


def run(args):
    global SERVER_FLAGS
    SERVER_FLAGS = tuple(args.server_flags.split()) if args.server_flags else ()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    model, cfg, choices, server = build(args.mode, args.model, args.server)
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True, return_tensors="pt")
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"]
    with torch.no_grad():
        model.generate(ids, max_new_tokens=2, do_sample=False)  # warm up
        if server:
            server.stats()
        t = time.time()
        out = model.generate(ids, max_new_tokens=args.tokens, do_sample=False)
        dt = time.time() - t
    gen = out[0, ids.shape[1]:].tolist()
    print(f"[{args.mode}] {len(gen)} tokens in {dt:.1f}s ({dt / max(len(gen), 1) * 1e3:.0f} ms a token, all of the model)", flush=True)
    print(tok.decode(gen, skip_special_tokens=True), flush=True)
    if server:
        server.stats()
        time.sleep(0.2)
    name = args.mode + (f"-{args.tag}" if args.tag else "")
    (work / f"{name}_gen.json").write_text(json.dumps({"prompt": ids[0].tolist(), "gen": gen, "text": tok.decode(gen)}))
    if args.mode == "ref":
        (work / "ref_ids.json").write_text(json.dumps({"prompt": ids[0].tolist(), "gen": gen}))
    ref = json.loads((work / "ref_ids.json").read_text())
    seq = torch.tensor([ref["prompt"] + ref["gen"]])
    choices.on = True
    with torch.no_grad():
        logits = model(seq).logits[0].float().numpy()
    choices.on = False
    np.save(work / f"{name}_logits.npy", logits)
    layers = sorted(choices.by_layer)
    np.save(work / f"{name}_experts.npy", np.stack([np.concatenate(choices.by_layer[l]) for l in layers]))
    print(f"[{args.mode}] forward of {seq.shape[1]} positions written", flush=True)
    if server:
        server.p.stdin.close()
        server.p.wait()


def compare(args):
    work = Path(args.work)
    other = args.other
    ref = json.loads((work / "ref_ids.json").read_text())
    n0 = len(ref["prompt"])
    a, b = np.load(work / "ref_logits.npy"), np.load(work / f"{other}_logits.npy")
    a, b = a[n0 - 1:-1], b[n0 - 1:-1]  # the positions predicting the generated tokens

    def logsm(x):
        x = x - x.max(-1, keepdims=True)
        return x - np.log(np.exp(x).sum(-1, keepdims=True))

    la, lb = logsm(a.astype(np.float64)), logsm(b.astype(np.float64))
    kl = (np.exp(la) * (la - lb)).sum(-1)
    top = (a.argmax(-1) == b.argmax(-1)).mean()
    print(f"generated positions {len(a)}: KL(ref || {other}) mean {kl.mean():.4f} max {kl.max():.4f}, top-1 agree {top * 100:.1f}%")
    ea, eb = np.load(work / "ref_experts.npy"), np.load(work / f"{other}_experts.npy")
    agree = np.array([[len(set(x) & set(y)) / len(x) for x, y in zip(ea[l], eb[l])] for l in range(len(ea))])
    print(f"experts agree {agree.mean() * 100:.1f}% (each layer: {' '.join(f'{v * 100:.0f}' for v in agree.mean(1))})")
    for m in ("ref", other):
        g = work / f"{m}_gen.json"
        if g.exists():
            print(f"--- {m}:\n{json.loads(g.read_text())['text']}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("mode", choices=["ref", "ane"])
    r.add_argument("model")
    r.add_argument("work")
    r.add_argument("--tokens", type=int, default=48)
    r.add_argument("--server", default="/tmp/aicheck")
    r.add_argument("--server-flags", default="", help="flags of aicheck moeserve, e.g. \"-bits 4 -group 64\"")
    r.add_argument("--tag", default="", help="written as MODE-TAG_*: another run of the same mode")
    c = sub.add_parser("compare")
    c.add_argument("work")
    c.add_argument("--other", default="ane", help="the run compared with the reference: ane, or ane-TAG")
    args = ap.parse_args()
    run(args) if args.cmd == "run" else compare(args)


if __name__ == "__main__":
    main()
