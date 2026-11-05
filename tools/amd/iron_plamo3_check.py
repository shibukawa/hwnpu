#!/usr/bin/env python3
"""Check iron_plamo3.py (IRON's graph run on the host) against Hugging Face's modeling_plamo.py.

Two steps in two environments, sharing a directory of files:

  # in a venv with torch and transformers: the reference logits (and a tiny checkpoint)
  python tools/amd/iron_plamo3_check.py ref  MODEL_DIR WORK [--tiny] [--tokens N] [--steps K]
  # in IRON's venv: the same through IRON's graph on the host (iron.lm.model.OnHost)
  python tools/amd/iron_plamo3_check.py iron MODEL_DIR WORK [--max-seq-len L] [--chunk C]

``ref --tiny`` writes a small random PLaMo 3 into WORK/tiny (4 layers, every second one full,
a window of 8, YaRN as the real model's) from MODEL_DIR's modeling_plamo.py, and checks that
instead of MODEL_DIR's weights; ``iron`` then reads WORK/tiny. The reference runs the prompt
(N random tokens, or WORK/prompt.json's ids when it exists) and K greedy steps after it in
float32, and writes each call's logits. ``iron`` feeds the same tokens and prints, for each
call, the cosine, the KL divergence of the softmaxes and whether the top token agrees.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

TINY = dict(
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    intermediate_size=512,
    vocab_size=1024,
    window_size=8,
    sliding_window_pattern=2,
    rope_theta=1000000,
    rope_local_theta=10000,
    rope_scaling_factor=256.0,
    initial_context_length=4096,
    max_position_embeddings=262144,
    tie_word_embeddings=False,
    rms_norm_eps=1e-6,
    bos_token_id=1,
    eos_token_id=1,
    pad_token_id=3,
)


def ref(a) -> None:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    model_dir = Path(a.model_dir)
    if a.tiny:
        tiny = work / "tiny"
        tiny.mkdir(exist_ok=True)
        cfg = json.loads((model_dir / "config.json").read_text())
        cfg.update(TINY)
        cfg.pop("rope_parameters", None)
        (tiny / "config.json").write_text(json.dumps(cfg, indent=1))
        shutil.copyfile(model_dir / "modeling_plamo.py", tiny / "modeling_plamo.py")
        torch.manual_seed(0)
        config = AutoConfig.from_pretrained(tiny, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, dtype=torch.float32)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name.endswith("norm.weight"):
                    p.normal_(0, 0.3)  # PLaMo's norms start at zero (offset + weight)
                else:
                    p.normal_(0, 0.05)
        model.to(torch.bfloat16).save_pretrained(tiny, safe_serialization=True)
        shutil.copyfile(model_dir / "modeling_plamo.py", tiny / "modeling_plamo.py")
        model_dir = tiny
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, dtype=torch.float32)
    model.eval()
    vocab = model.config.vocab_size
    prompt_file = work / "prompt.json"
    if prompt_file.exists() and not a.tiny:
        tokens = json.loads(prompt_file.read_text())
    else:
        rng = np.random.default_rng(1)
        tokens = [1] + [int(t) for t in rng.integers(4, vocab, a.tokens - 1)]
    calls = []
    history = list(tokens)
    with torch.no_grad():
        for k in range(a.steps + 1):
            logits = model(torch.tensor([history])).logits[0, -1].float().numpy()
            fn = f"ref_{k}.bin"
            logits.astype(np.float32).tofile(work / fn)
            calls.append(dict(tokens=len(history), logits=fn, top=int(np.argmax(logits))))
            history.append(int(np.argmax(logits)))
    (work / "ref.json").write_text(json.dumps(dict(model=str(model_dir), prompt=tokens, calls=calls)) + "\n")
    print(f"{work}/ref.json: a prompt of {len(tokens)} tokens and {a.steps} steps, tops {[c['top'] for c in calls]}")


def iron(a) -> int:
    sys.path.insert(0, str(HERE))
    import dataclasses

    from iron.lm.model import OnHost
    from iron_plamo3 import Plamo3

    work = Path(a.work)
    r = json.loads((work / "ref.json").read_text())
    if a.chunk:
        base = Plamo3.default_config.__func__

        def default_config(cls, weights, _base=base, _chunk=a.chunk):
            return dataclasses.replace(_base(cls, weights), prefill_chunk=_chunk)

        Plamo3.default_config = classmethod(default_config)
    Host = type("Host", (OnHost, Plamo3), {})
    model = Host.from_checkpoint(r["model"], max_seq_len=a.max_seq_len)
    rings = [layer.cache.ring for layer in model.layers]
    print(f"max_seq_len {model.config.max_seq_len}, chunk {model.config.prefill_chunk}, rings {rings}")
    history = list(r["prompt"])
    worst = 0.0
    agree = 0
    for c in r["calls"]:
        want = np.fromfile(work / c["logits"], np.float32)
        got = np.asarray(model.logits(history[: c["tokens"]]), np.float32).reshape(-1)[: want.size]
        cos = float(np.dot(got, want) / (np.linalg.norm(got) * np.linalg.norm(want)))
        p = np.exp(want - want.max())
        p /= p.sum()
        q = np.exp(got - got.max())
        q /= q.sum()
        kl = float(np.sum(p * (np.log(p + 1e-30) - np.log(q + 1e-30))))
        top = int(np.argmax(got))
        agree += top == c["top"]
        worst = max(worst, kl)
        print(f"  {c['tokens']:5d} tokens: cosine {cos:.6f}, KL {kl:.2e}, top {top} (want {c['top']})")
        if len(history) == c["tokens"]:
            history.append(c["top"])
    print(f"tops agree {agree}/{len(r['calls'])}, worst KL {worst:.2e}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["ref", "iron"])
    ap.add_argument("model_dir")
    ap.add_argument("work")
    ap.add_argument("--tiny", action="store_true", help="ref: check a small random PLaMo 3 instead")
    ap.add_argument("--tokens", type=int, default=40, help="ref: random prompt tokens")
    ap.add_argument("--steps", type=int, default=4, help="ref: greedy steps after the prompt")
    ap.add_argument("--max-seq-len", type=int, default=64, help="iron: the context")
    ap.add_argument("--chunk", type=int, default=16, help="iron: the prompt chunk")
    a = ap.parse_args()
    if a.step == "ref":
        ref(a)
        return 0
    return iron(a)


if __name__ == "__main__":
    sys.exit(main())
