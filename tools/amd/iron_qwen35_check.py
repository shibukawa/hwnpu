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
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=torch.float32)
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
    from iron_qwen35 import Qwen35, Qwen35Moe

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
    ap.add_argument("--moe", action="store_true", help="iron: Qwen35Moe, a token at a time (decode only)")
    a = ap.parse_args()
    if a.step == "ref":
        ref(a)
        return 0
    return iron(a)


if __name__ == "__main__":
    sys.exit(main())
