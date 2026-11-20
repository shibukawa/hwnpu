"""float32 reference of a Qwen3.5-MoE text model (qwen3_5_moe: Qwen3.6-35B-A3B) from the checkpoint
hwnpu-dev mxfp4 writes (its experts MXFP4, dequantized here exactly; the gate and up projections
interleaved there, the gate the even rows), a decoder layer at a time: transformers'
Qwen3_5MoeForCausalLM is built on the meta device and each layer's weights are read just before it
runs and dropped after (stream_weights.py), about 4 GB held.

    PY=~/models/ref-venv/bin/python
    $PY tools/ref/ref_qwen36_moe.py DERIVED_DIR --chat "日本の首都はどこですか？"   # stats, next tokens
    $PY tools/ref/ref_qwen36_moe.py DERIVED_DIR --ids IDS.json --out OUT.f32       # every row's logits

It prints the largest magnitude of every layer's output and, inside, of the inputs of its norms
(what an fp16 program squares), and the top tokens after the prompt.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stream_weights import Checkpoint, streamed  # noqa: E402

FP4 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def dequant(blocks, scales):
    """MXFP4 (rows, nb, 16) bytes and (rows, nb) e8m0 scales to float32 (rows, nb*32)."""
    b = blocks.to(torch.int64)
    lo, hi = FP4[b & 15], FP4[b >> 4]
    v = torch.stack([lo, hi], dim=-1).reshape(*b.shape[:-1], 32)  # the even value in the low nibble
    s = torch.pow(2.0, scales.to(torch.float32) - 127).unsqueeze(-1)
    return (v * s).reshape(*b.shape[:-2], -1)


class Derived(Checkpoint):
    """The derived checkpoint with its experts' MXFP4 tensors seen as transformers' bf16 ones."""

    def names(self):
        out = []
        for n in self.index:
            if n.endswith("_proj_blocks"):
                out.append(n[: -len("_blocks")])
            elif not n.endswith("_proj_scales"):
                out.append(n)
        return out

    def get(self, name):
        if name.endswith("experts.gate_up_proj") or name.endswith("experts.down_proj"):
            w = dequant(super().get(name + "_blocks"), super().get(name + "_scales"))
            if name.endswith("gate_up_proj"):  # interleaved rows back to the gate's then the up's
                w = torch.cat([w[:, 0::2], w[:, 1::2]], dim=1)
            return w
        return super().get(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--chat", help="a user message, in the chat format without thinking")
    ap.add_argument("--ids", help="token ids (JSON list)")
    ap.add_argument("--out", help="write every row's logits (float32)")
    a = ap.parse_args()
    d = Path(a.dir)
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

    cfg = json.loads((d / "config.json").read_text())
    tc = Qwen3_5MoeTextConfig(**cfg["text_config"])
    tok = AutoTokenizer.from_pretrained(str(d))
    if a.ids:
        ids = json.loads(Path(a.ids).read_text())
    else:
        text = f"<|im_start|>user\n{a.chat}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ids = tok(text)["input_ids"]
    stats = []
    norm_in = []
    ck = Derived.__new__(Derived)
    Checkpoint.__init__(ck, d)

    import stream_weights

    stream_weights.Checkpoint = lambda _d: ck  # the derived view
    model = streamed(lambda: Qwen3_5MoeForCausalLM(tc), d, ["model.layers"], torch.float32,
                     rename=lambda n: n.replace("model.language_model.", "model."), skip=("model.visual.", "mtp."), stats=stats)
    for name, mod in model.named_modules():
        if name.endswith("layernorm") or name.endswith("q_norm") or name.endswith("k_norm"):
            mod.register_forward_pre_hook(lambda m, args, name=name: norm_in.append((name, float(args[0].abs().max()))))
    with torch.no_grad():
        out = model(torch.tensor([ids]), use_cache=False)
    for n, v in stats:
        print(f"{n}: max |out| {v:.1f}")
    worst = sorted(norm_in, key=lambda x: -x[1])[:10]
    print("largest norm inputs:", ", ".join(f"{n} {v:.1f}" for n, v in worst))
    lg = out.logits[0].float()
    top = torch.topk(lg[-1], 5)
    print("next:", [(tok.decode([int(i)]), round(float(v), 2)) for v, i in zip(top.values, top.indices)])
    if a.out:
        np.asarray(lg.numpy(), dtype=np.float32).tofile(a.out)
        print("ids:", ids)


if __name__ == "__main__":
    main()
