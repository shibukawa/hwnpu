"""Convert sentence-transformers/all-MiniLM-L6-v2 to a Core ML model with a fixed sequence length.

The model takes input_ids and attention_mask, int32 of shape (batch, seq), and returns the
sentence embedding (batch, 384): mean pooling over the unmasked tokens, then L2 normalization,
as sentence-transformers does. Compute is fp16.

    python convert_minilm.py OUT.mlpackage [--seq 128] [--batch 1]

The script downloads the model from Hugging Face, checks where Core ML places each operation,
and compares Core ML's embeddings (CPU and Neural Engine) with PyTorch's fp32 embeddings for a
few sentences (cosine similarity). It also writes OUT.inputs.npz with those sentences' token ids
for reuse. Needs torch, transformers, coremltools >= 8 and numpy.
"""
import argparse
import os
import sys

import numpy as np

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SENTENCES = [
    "The Neural Engine runs small models at low power.",
    "hwnpu is a Go library for CPU integrated NPUs.",
    "It is raining in Tokyo today.",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--batch", type=int, default=1)
    a = ap.parse_args()

    import coremltools as ct
    import torch
    from transformers import AutoModel, AutoTokenizer

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ane_pad import placement

    tok = AutoTokenizer.from_pretrained(MODEL)
    bert = AutoModel.from_pretrained(MODEL, attn_implementation="eager").eval()

    class Embed(torch.nn.Module):
        """Mean pooled, normalized embedding (what sentence-transformers computes)."""

        def __init__(self, m, manual, shape=None):
            super().__init__()
            self.m, self.manual, self.shape = m, manual, shape

        def forward(self, input_ids, attention_mask):
            h = self.encode(input_ids, attention_mask) if self.manual else \
                self.m(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(h.dtype)
            e = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return torch.nn.functional.normalize(e, dim=-1)

        # The same BERT encoder written out with the model's own modules. Tracing the
        # transformers implementation pulls in mask helpers that coremltools cannot convert.
        def encode(self, input_ids, attention_mask):
            emb, cfg = self.m.embeddings, self.m.config
            b, s = self.shape or input_ids.shape  # Python ints, so the trace has no shape ops
            pos = torch.arange(s).unsqueeze(0)
            h = emb.word_embeddings(input_ids) + emb.position_embeddings(pos) + emb.token_type_embeddings(torch.zeros_like(input_ids))
            h = emb.LayerNorm(h)
            heads = cfg.num_attention_heads
            dh = cfg.hidden_size // heads
            bias = (1.0 - attention_mask.to(h.dtype))[:, None, None, :] * -1e4  # fp16 safe
            for layer in self.m.encoder.layer:
                at = layer.attention.self

                def split(t):
                    return t.view(b, s, heads, dh).transpose(1, 2)

                q, k, v = split(at.query(h)), split(at.key(h)), split(at.value(h))
                a = torch.softmax(q @ k.transpose(-1, -2) / dh ** 0.5 + bias, dim=-1)
                o = (a @ v).transpose(1, 2).reshape(b, s, cfg.hidden_size)
                h = layer.attention.output.LayerNorm(layer.attention.output.dense(o) + h)
                f = layer.output.dense(layer.intermediate.intermediate_act_fn(layer.intermediate.dense(h)))
                h = layer.output.LayerNorm(f + h)
            return h

    reference = Embed(bert, False).eval()
    enc = tok(SENTENCES, padding="max_length", truncation=True, max_length=a.seq, return_tensors="pt")
    ids, mask = enc["input_ids"].to(torch.int32), enc["attention_mask"].to(torch.int32)
    with torch.no_grad():
        want = reference(ids.long(), mask.long()).numpy()
        manual = Embed(bert, True)(ids, mask).numpy()
        model = Embed(bert, True, (a.batch, a.seq)).eval()
        print(f"hand written encoder vs transformers: max |diff| = {np.abs(manual - want).max():.2e}")
        traced = torch.jit.trace(model, (ids[: a.batch], mask[: a.batch]))

    shape = (a.batch, a.seq)
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input_ids", shape=shape, dtype=np.int32),
                ct.TensorType(name="attention_mask", shape=shape, dtype=np.int32)],
        outputs=[ct.TensorType(name="embedding")],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS14,
    )
    ml.short_description = f"{MODEL}, seq {a.seq}, batch {a.batch}, fp16 (hwnpu tools/coreml/convert_minilm.py)"
    on_ne, off = placement(ml)
    print(f"placement: {on_ne} op(s) on the Neural Engine, elsewhere: {sorted(set(off)) or 'none'}")
    ml.save(a.out)

    for units in (ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_NE):
        m = ct.models.MLModel(a.out, compute_units=units)
        cos = []
        for i in range(0, len(SENTENCES) - a.batch + 1, a.batch):
            got = m.predict({"input_ids": ids[i:i + a.batch].numpy(), "attention_mask": mask[i:i + a.batch].numpy()})["embedding"]
            for g, w in zip(np.asarray(got, np.float64), want[i:i + a.batch]):
                cos.append(float(g @ w / np.linalg.norm(g) / np.linalg.norm(w)))
        print(f"{units.name}: cosine similarity to PyTorch fp32 = {', '.join(f'{c:.5f}' for c in cos)}")
    np.savez(os.path.splitext(a.out)[0] + ".inputs.npz", input_ids=ids.numpy(), attention_mask=mask.numpy(), embedding=want)


if __name__ == "__main__":
    main()
