"""float32 reference of a Qwen2 model (``qwen2``: Qwen2.5, Qwen2.5-Coder), a decoder layer at a time.

Qwen2.5-Coder-7B is 30 GB in float32 and the 14B 59 GB, more than a 16 GB Mac holds. This builds
transformers' own modules (Qwen2DecoderLayer, Qwen2RotaryEmbedding, Qwen2RMSNorm) one layer at a
time, its weights read from the safetensors checkpoint in float32 (about 0.9 GB for a 7B layer),
runs the text through it and drops it. It writes what ref_logits.py writes, for bonsaiai-dev check
and bonsaiai-dev generate -score:

    OUT_PREFIX.ids.bin     int32 (seq)                       token ids
    OUT_PREFIX.hidden.bin  float32 (layers + 1, seq, hidden) token embeddings, then each layer's output
    OUT_PREFIX.logits.bin  float32 (seq, vocab) logits
    OUT_PREFIX.json        model_type, seq, sizes, text, dtype, output

    PY=~/.cache/bonsaiai/dev/ref-venv/bin/python
    $PY tools/ref/ref_qwen2.py MODEL_DIR OUT_PREFIX [--seq 64] [--text "..."] [--chat]

--chat puts the text in a user turn of the model's chat template (the system turn and the
assistant's start included), as bonsaiai-chat would give it.

--stats OUT.safetensors --calib FILE... instead runs the files' texts (up to --calib-tokens
each) through the model and writes the mean square of each linear layer's inputs by in-feature,
float32, named WEIGHT.x2 after the weights that read them (q_proj, k_proj and v_proj the same,
gate_proj and up_proj the same; lm_head.weight.x2 the final norm's output): what an
activation-aware fitting of the weights' tables weighs each column's error by (tools/coreai/lutfit).
Needs torch and transformers.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

TEXT = ("Write a Go function that returns the n-th Fibonacci number using iteration, and explain "
        "its time complexity in one sentence. bonsaiai は CPU 内蔵の NPU で動く Go のライブラリです。")


class Checkpoint:
    """The checkpoint's tensors by name, in float32."""

    def __init__(self, model_dir):
        self.dir = Path(model_dir)
        index = self.dir / "model.safetensors.index.json"
        if index.exists():
            self.where = json.loads(index.read_text())["weight_map"]
        else:
            with safe_open(str(self.dir / "model.safetensors"), framework="pt") as f:
                self.where = dict.fromkeys(f.keys(), "model.safetensors")
        self.files = {}

    def get(self, name):
        f = self.where[name]
        if f not in self.files:
            self.files[f] = safe_open(str(self.dir / f), framework="pt")
        return self.files[f].get_tensor(name).float()

    def load(self, module, prefix):
        """module's parameters from the tensors under prefix."""
        sd = {k: self.get(prefix + k) for k, _ in module.named_parameters()}
        module.load_state_dict(sd, strict=True)
        return module


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--text", default=TEXT)
    ap.add_argument("--chat", action="store_true", help="the text as a user turn of the chat template")
    ap.add_argument("--stats", help="write the inputs' mean squares to this safetensors file instead")
    ap.add_argument("--calib", nargs="*", default=[], help="--stats: text files to run")
    ap.add_argument("--calib-tokens", type=int, default=512)
    a = ap.parse_args()
    if a.stats:
        return stats(a)

    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen2 import modeling_qwen2 as M

    cfg = AutoConfig.from_pretrained(a.model_dir)
    if cfg.model_type != "qwen2":
        raise SystemExit(f"model_type {cfg.model_type} is not qwen2")
    cfg._attn_implementation = "eager"
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    text = a.text
    if a.chat:
        text = tok.apply_chat_template([{"role": "user", "content": a.text}], tokenize=False, add_generation_prompt=True)
    ids = tok(text, add_special_tokens=False)["input_ids"][: a.seq]
    n = len(ids)
    ck = Checkpoint(a.model_dir)
    torch.set_grad_enabled(False)

    emb = ck.get("model.embed_tokens.weight")
    h = emb[torch.tensor(ids)][None]  # (1, n, hidden)
    del emb
    hidden = [h[0].clone()]
    pos = torch.arange(n)[None]
    rope = M.Qwen2RotaryEmbedding(config=cfg)
    cos_sin = rope(h, pos)
    mask = torch.full((n, n), float("-inf")).triu(1)[None, None]
    for i in range(cfg.num_hidden_layers):
        t0 = time.time()
        layer = ck.load(M.Qwen2DecoderLayer(cfg, i), f"model.layers.{i}.")
        out = layer(h, attention_mask=mask, position_ids=pos, position_embeddings=cos_sin)
        h = out[0] if isinstance(out, tuple) else out
        hidden.append(h[0].clone())
        del layer
        print(f"layer {i}: {time.time() - t0:.1f} s, max |h| {h.abs().max().item():.1f}", flush=True)
    norm = ck.load(M.Qwen2RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps), "model.norm.")
    head = "model.embed_tokens.weight" if cfg.tie_word_embeddings else "lm_head.weight"
    logits = norm(h)[0] @ ck.get(head).T

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(ids, dtype=np.int32).tofile(str(out) + ".ids.bin")
    torch.stack(hidden).numpy().astype(np.float32).tofile(str(out) + ".hidden.bin")
    logits.numpy().astype(np.float32).tofile(str(out) + ".logits.bin")
    meta = {"model_type": cfg.model_type, "seq": n, "hidden": cfg.hidden_size, "layers": cfg.num_hidden_layers,
            "vocab": int(logits.shape[-1]), "text": text, "dtype": "float32", "output": "logits"}
    Path(str(out) + ".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"{n} tokens, top token after the last: {int(logits[-1].argmax())} ({tok.decode([int(logits[-1].argmax())])!r})")


def stats(a):
    """--stats: the mean square of each linear's inputs over the --calib texts, by in-feature."""
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen2 import modeling_qwen2 as M

    cfg = AutoConfig.from_pretrained(a.model_dir)
    cfg._attn_implementation = "eager"
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    seqs = [tok(Path(f).read_text(), add_special_tokens=False)["input_ids"][: a.calib_tokens] for f in a.calib]
    ck = Checkpoint(a.model_dir)
    torch.set_grad_enabled(False)
    emb = ck.get("model.embed_tokens.weight")
    hs = [emb[torch.tensor(s)][None] for s in seqs]
    del emb
    rope = M.Qwen2RotaryEmbedding(config=cfg)
    sums, count = {}, 0
    for s in seqs:
        count += len(s)

    def add(name, x):
        v = x.reshape(-1, x.shape[-1]).double().pow(2).sum(0)
        sums[name] = sums.get(name, 0) + v

    for i in range(cfg.num_hidden_layers):
        t0 = time.time()
        layer = ck.load(M.Qwen2DecoderLayer(cfg, i), f"model.layers.{i}.")
        p = f"model.layers.{i}."
        hooks = []
        for mod, names in ((layer.self_attn.q_proj, ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]),
                           (layer.self_attn.o_proj, ["self_attn.o_proj"]),
                           (layer.mlp.gate_proj, ["mlp.gate_proj", "mlp.up_proj"]),
                           (layer.mlp.down_proj, ["mlp.down_proj"])):
            def hook(_m, args, names=names):
                for n in names:
                    add(p + n + ".weight.x2", args[0])
            hooks.append(mod.register_forward_pre_hook(hook))
        for j, h in enumerate(hs):
            n = h.shape[1]
            pos = torch.arange(n)[None]
            mask = torch.full((n, n), float("-inf")).triu(1)[None, None]
            out = layer(h, attention_mask=mask, position_ids=pos, position_embeddings=rope(h, pos))
            hs[j] = out[0] if isinstance(out, tuple) else out
        for hk in hooks:
            hk.remove()
        del layer
        print(f"layer {i}: {time.time() - t0:.1f} s", flush=True)
    norm = ck.load(M.Qwen2RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps), "model.norm.")
    head = "model.embed_tokens.weight" if cfg.tie_word_embeddings else "lm_head.weight"
    for h in hs:
        add(head + ".x2", norm(h))
    save_file({k: (v / count).float().contiguous() for k, v in sums.items()}, a.stats)
    print(f"{len(sums)} inputs over {count} tokens of {len(seqs)} texts")


if __name__ == "__main__":
    main()
