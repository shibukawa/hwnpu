"""A reference of gpt-oss (``gpt_oss``: openai/gpt-oss-20b) on the CPU, for scoring a conversion.

The model runs in float32 with transformers' modules except its experts: they stay in the
checkpoint's MXFP4 and each forward dequantizes the experts its rows chose (transformers'
convert_moe_packed_tensors, GptOssExperts' gated unit), so the 12 GB of experts are never all in
memory. The prompt is a user message in the chat template (harmony). It generates greedily, then
runs the prompt and the answer in one forward and writes, as tools/ref/ref_logits.py does (for
hwnpu-dev generate -score):

    OUT_PREFIX.ids.bin     int32 (seq)          the prompt's and the answer's tokens
    OUT_PREFIX.logits.bin  float32 (seq, vocab) the logits after each
    OUT_PREFIX.json        model_type, seq, sizes, the text, from (the answer's first token)

    PY=~/models/ref-venv/bin/python
    $PY -I tools/ref/gptoss_ref.py MODEL_DIR OUT_PREFIX [--tokens 40] [--prompt "..."]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from torch import nn
from transformers import AutoTokenizer
from transformers.integrations.mxfp4 import convert_moe_packed_tensors
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM, GptOssRotaryEmbedding

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

    def slice(self, name, e):
        f = self.map[name]
        if f not in self.files:
            self.files[f] = safe_open(self.dir / f, framework="pt")
        return self.files[f].get_slice(name)[e]


class Experts(nn.Module):
    """GptOssExperts with the chosen experts dequantized on demand from MXFP4."""

    def __init__(self, ck, cfg, layer):
        super().__init__()
        self.ck, self.p = ck, f"model.layers.{layer}.mlp.experts."
        self.gb = ck.get(self.p + "gate_up_proj_bias").float()
        self.db = ck.get(self.p + "down_proj_bias").float()
        self.alpha, self.limit = cfg.swiglu_alpha, cfg.swiglu_limit

    def weight(self, kind, e):
        b = self.ck.slice(self.p + kind + "_blocks", e)
        s = self.ck.slice(self.p + kind + "_scales", e)
        # (1, out, blocks, 16) in, (1, in, out) out: x @ it
        return convert_moe_packed_tensors(b[None], s[None], dtype=torch.float32)[0]

    def forward(self, hidden_states, router_indices=None, routing_weights=None):
        out = torch.zeros_like(hidden_states, dtype=torch.float32)
        x = hidden_states.float()
        for e in router_indices.unique().tolist():
            tok, pos = torch.where(router_indices == e)
            gu = x[tok] @ self.weight("gate_up_proj", e) + self.gb[e]
            gate, up = gu[..., ::2], gu[..., 1::2]
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            h = (up + 1) * gate * torch.sigmoid(gate * self.alpha)
            y = h @ self.weight("down_proj", e) + self.db[e]
            out.index_add_(0, tok, y * routing_weights[tok, pos, None].float())
        return out.to(hidden_states.dtype)


def build(model_dir):
    raw = json.loads((Path(model_dir) / "config.json").read_text())
    raw.pop("quantization_config", None)
    cfg = GptOssConfig(**raw)
    cfg._attn_implementation = "eager"
    ck = Checkpoint(model_dir)
    with torch.device("meta"):
        model = GptOssForCausalLM(cfg)
    for l, layer in enumerate(model.model.layers):
        layer.mlp.experts = Experts(ck, cfg, l)
    sd = {}
    for name, t in model.state_dict().items():
        if t.device.type == "meta":
            sd[name] = ck.get(name).float()
    model.load_state_dict(sd, strict=False, assign=True)
    model.model.rotary_emb = GptOssRotaryEmbedding(config=cfg)
    left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.device.type == "meta"]
    if left:
        raise RuntimeError(f"still on meta: {left[:5]}")
    model.eval()
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("prefix")
    ap.add_argument("--tokens", type=int, default=40)
    ap.add_argument("--prompt", default=PROMPT)
    args = ap.parse_args()
    torch.set_num_threads(8)
    model, cfg = build(args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok.apply_chat_template([{"role": "user", "content": args.prompt}], add_generation_prompt=True, return_tensors="pt")
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"]
    with torch.no_grad():
        t = time.time()
        out = model.generate(ids, max_new_tokens=args.tokens, do_sample=False)
        dt = time.time() - t
        gen = out[0, ids.shape[1]:].tolist()
        print(f"{len(gen)} tokens in {dt:.0f}s", flush=True)
        print(tok.decode(gen), flush=True)
        seq = out[:, : ids.shape[1] + len(gen)]
        logits = model(seq).logits[0].float().numpy()
    seq[0].numpy().astype("<i4").tofile(args.prefix + ".ids.bin")
    logits.astype("<f4").tofile(args.prefix + ".logits.bin")
    meta = {"model_type": "gpt_oss", "seq": int(seq.shape[1]), "hidden_size": cfg.hidden_size, "num_layers": cfg.num_hidden_layers,
            "vocab_size": cfg.vocab_size, "dtype": "float32 (transformers, the experts dequantized from MXFP4)", "output": "logits",
            "from": int(ids.shape[1]), "text": tok.decode(gen)}
    Path(args.prefix + ".json").write_text(json.dumps(meta))
    print(f"written {seq.shape[1]} positions from {ids.shape[1]}", flush=True)


if __name__ == "__main__":
    main()
