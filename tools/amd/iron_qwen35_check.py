#!/usr/bin/env python3
"""Check iron_qwen35.py (IRON's graph run on the host) against transformers' modeling_qwen3_5.py.

Two steps in two environments, sharing a directory of files:

  # in a venv with torch and transformers (qwen3_5): the reference logits
  python tools/amd/iron_qwen35_check.py ref  MODEL_DIR WORK [--text T] [--steps K]
  # in IRON's venv: the same through IRON's graph on the host (iron.lm.model.OnHost)
  python tools/amd/iron_qwen35_check.py iron MODEL_DIR WORK [--max-seq-len L] [--chunk C]

The prompt is the tokens of ``--text`` (a truncated checkpoint, ~/models/dl/partial.py, holds
the embedding rows of its text only). The reference runs it and K greedy steps after it in
float32 and writes each call's logits; ``iron`` feeds the same tokens and prints, for each call,
the cosine, the KL divergence of the softmaxes and whether the top token agrees.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEXT = ("The Neural Engine runs small language models at low power. hwnpu is a Go library for CPU "
        "integrated NPUs: Apple, Intel and AMD. 日本の首都は東京です。")


def ref(a) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    tokens = AutoTokenizer.from_pretrained(a.model_dir)(a.text)["input_ids"][: a.tokens]
    model = streamed(a.model_dir) if a.stream else AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=torch.float32)
    model.eval()
    calls, history = [], list(tokens)
    with torch.no_grad():
        for k in range(a.steps + 1):
            logits = model(torch.tensor([history])).logits[0, -1].float().numpy()
            fn = f"ref_{k}.bin"
            logits.astype(np.float32).tofile(work / fn)
            calls.append(dict(tokens=len(history), logits=fn, top=int(np.argmax(logits))))
            history.append(int(np.argmax(logits)))
    (work / "ref.json").write_text(json.dumps(dict(model=str(a.model_dir), prompt=tokens, calls=calls)) + "\n")
    print(f"{work}/ref.json: a prompt of {len(tokens)} tokens and {a.steps} steps, tops {[c['top'] for c in calls]}")


def streamed(model_dir):
    """The checkpoint's text model in float32, a decoder layer's weights read only while it runs.

    For a checkpoint whose float32 weights do not fit in memory (Qwen3.6-35B-A3B: 140 GB):
    the model is built on the meta device, the embeddings, final norm and head are read once,
    and each layer reads its weights from the shards before it runs and drops them after.
    """
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_dir)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    index = Path(model_dir) / "model.safetensors.index.json"
    if index.exists():
        where = json.loads(index.read_text())["weight_map"]
    else:
        with safe_open(str(Path(model_dir) / "model.safetensors"), framework="pt") as f:
            where = dict.fromkeys(f.keys(), "model.safetensors")
    files = {}

    def tensor(name):
        # The HF names of a multimodal checkpoint's text model are under model.language_model.
        if name == "lm_head.weight" and name not in where and config.get_text_config().tie_word_embeddings:
            name = "model.embed_tokens.weight"
        stored = name if name in where else name.replace("model.", "model.language_model.", 1)
        f = where[stored]
        if f not in files:
            files[f] = safe_open(str(Path(model_dir) / f), framework="pt")
        return files[f].get_tensor(stored).float()

    def put(module, prefix, load):
        for n, p in list(module.named_parameters(recurse=True)):
            owner, _, leaf = n.rpartition(".")
            m = module.get_submodule(owner) if owner else module
            t = tensor(prefix + n) if load else torch.empty(p.shape, device="meta")
            m._parameters[leaf] = torch.nn.Parameter(t, requires_grad=False)

    for name in ("model.embed_tokens", "model.norm", "lm_head"):
        put(model.get_submodule(name), name + ".", True)
    if config.get_text_config().tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    rotary = model.model.rotary_emb
    model.model.rotary_emb = type(rotary)(config=rotary.config)  # its buffers, off the meta device
    for i, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{i}."
        layer.register_forward_pre_hook(lambda m, args, prefix=prefix: put(m, prefix, True))
        layer.register_forward_hook(lambda m, args, out, prefix=prefix: put(m, prefix, False))
    return model


def compare(got, want):
    cos = float(np.dot(got, want) / (np.linalg.norm(got) * np.linalg.norm(want)))
    p = np.exp(want - want.max())
    p /= p.sum()
    q = np.exp(got - got.max())
    q /= q.sum()
    kl = float(np.sum(p * (np.log(p + 1e-30) - np.log(q + 1e-30))))
    return cos, kl


def iron(a) -> int:
    sys.path.insert(0, str(HERE))
    import dataclasses

    from iron.lm.model import OnHost
    import iron_qwen35
    from iron_qwen35 import Qwen35, Qwen35Moe

    if a.q4:
        iron_qwen35.install()  # the export's --q4
    if a.moe:
        return iron_decode(a, OnHost, Qwen35Moe)

    work = Path(a.work)
    r = json.loads((work / "ref.json").read_text())
    base = Qwen35.default_config.__func__

    def default_config(cls, weights, _base=base, _chunk=a.chunk):
        return dataclasses.replace(_base(cls, weights), prefill_chunk=_chunk)

    Qwen35.default_config = classmethod(default_config)
    Host = type("Host", (OnHost, Qwen35), {})
    model = Host.from_checkpoint(a.model_dir, max_seq_len=a.max_seq_len)
    print(f"max_seq_len {model.config.max_seq_len}, chunk {model.config.prefill_chunk}")
    history = list(r["prompt"])
    worst, agree = 0.0, 0
    for c in r["calls"]:
        want = np.fromfile(work / c["logits"], np.float32)
        got = np.asarray(model.logits(history[: c["tokens"]]), np.float32).reshape(-1)[: want.size]
        cos, kl = compare(got, want)
        top = int(np.argmax(got))
        agree += top == c["top"]
        worst = max(worst, kl)
        print(f"  {c['tokens']:5d} tokens: cosine {cos:.6f}, KL {kl:.2e}, top {top} (want {c['top']})")
        if len(history) == c["tokens"]:
            history.append(c["top"])
    print(f"tops agree {agree}/{len(r['calls'])}, worst KL {worst:.2e}")
    return 0


def iron_decode(a, OnHost, Model) -> int:
    """The same a token at a time through the decode step (a model with no prompt chunk)."""
    work = Path(a.work)
    r = json.loads((work / "ref.json").read_text())
    Host = type("Host", (OnHost, Model), {})
    model = Host.from_checkpoint(a.model_dir, max_seq_len=a.max_seq_len)
    want_at = {c["tokens"]: c for c in r["calls"]}
    history = list(r["prompt"])
    worst, agree, p = 0.0, 0, 0
    while p < len(history):
        logits, _ = model(token=np.int32(history[p]), position=np.int32(p))
        p += 1
        c = want_at.get(p)
        if c is None:
            continue
        want = np.fromfile(work / c["logits"], np.float32)
        got = np.asarray(logits.numpy(), np.float32).reshape(-1)[: want.size]
        got.tofile(work / f"iron_{c['logits']}")  # the host graph's, a check for the device
        cos, kl = compare(got, want)
        top = int(np.argmax(got))
        agree += top == c["top"]
        worst = max(worst, kl)
        print(f"  {p:5d} tokens: cosine {cos:.6f}, KL {kl:.2e}, top {top} (want {c['top']})", flush=True)
        if p == len(history) and p < r["calls"][-1]["tokens"]:
            history.append(c["top"])
    print(f"tops agree {agree}/{len(r['calls'])}, worst KL {worst:.2e}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["ref", "iron"])
    ap.add_argument("model_dir")
    ap.add_argument("work")
    ap.add_argument("--text", default=TEXT, help="ref: the prompt")
    ap.add_argument("--tokens", type=int, default=32, help="ref: the prompt's first tokens")
    ap.add_argument("--steps", type=int, default=3, help="ref: greedy steps after the prompt")
    ap.add_argument("--max-seq-len", type=int, default=64, help="iron: the context")
    ap.add_argument("--chunk", type=int, default=16, help="iron: the prompt chunk")
    ap.add_argument("--q4", action="store_true", help="iron: the projections as the export's --q4 makes them")
    ap.add_argument("--stream", action="store_true", help="ref: read a layer's weights only while it runs")
    ap.add_argument("--moe", action="store_true", help="iron: Qwen35Moe, a token at a time (decode only)")
    a = ap.parse_args()
    if a.step == "ref":
        ref(a)
        return 0
    return iron(a)


if __name__ == "__main__":
    sys.exit(main())
