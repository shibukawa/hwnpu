"""Reference activations of a Hugging Face causal language model, for checking a conversion.

    python ref_logits.py MODEL_DIR OUT_PREFIX [--seq 32] [--text "..."]

MODEL_DIR holds config.json, the tokenizer and the checkpoint (and modeling_*.py for models
that need trust_remote_code; symlinks to downloaded files are fine). Runs on the CPU in float32
and writes, for readers without numpy (hwnpu-check and the Go tests), little endian and row major:

    OUT_PREFIX.ids.bin     int32 (seq)                       token ids
    OUT_PREFIX.hidden.bin  float32 (layers + 1, seq, hidden) embeddings, then each decoder layer's output
    OUT_PREFIX.logits.bin  float32 (seq, vocab)
    OUT_PREFIX.json        model_type, seq, sizes, text, dtype

and the same arrays in OUT_PREFIX.npz. The hidden states are taken with hooks on the decoder
layers rather than from output_hidden_states, whose last entry is after the final norm in many
implementations: here entry i + 1 is always the output of layer i, before the final norm.
Needs torch and transformers.
"""
import argparse
import json
import time

import numpy as np

TEXT = "The Neural Engine runs small language models at low power. hwnpu is a Go library for CPU integrated NPUs: Apple, Intel and AMD."


def decoder_layers(model, n):
    """The ModuleList of the n decoder layers (the longest list of that length)."""
    import torch

    found = [(name, m) for name, m in model.named_modules() if isinstance(m, torch.nn.ModuleList) and len(m) == n]
    if not found:
        raise SystemExit(f"no module list of {n} layers in the model")
    name, layers = found[0]
    print(f"decoder layers: {name} ({type(layers[0]).__name__})", flush=True)
    return layers


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--text", default=TEXT)
    a = ap.parse_args()
    prefix = a.out.removesuffix(".npz")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
    ids = tok(a.text, return_tensors="pt", add_special_tokens=True)["input_ids"][:, : a.seq]
    if ids.shape[1] < a.seq:
        raise SystemExit(f"text has only {ids.shape[1]} tokens, need {a.seq}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, trust_remote_code=True, dtype=torch.float32, attn_implementation="eager").eval()
    print(f"loaded in {time.time() - t0:.1f}s: {sum(p.numel() for p in model.parameters()) / 1e9:.3f} B parameters", flush=True)
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    layers = decoder_layers(model, cfg.num_hidden_layers)

    captured = {}

    def first(x):
        return x[0] if isinstance(x, (tuple, list)) else x

    def pre_hook(_module, args, kwargs):
        captured["embeddings"] = (args[0] if args else kwargs["hidden_states"]).detach()

    def hook(i):
        def f(_module, _args, out):
            captured[i] = first(out).detach()

        return f

    handles = [layers[0].register_forward_pre_hook(pre_hook, with_kwargs=True)]
    handles += [layer.register_forward_hook(hook(i)) for i, layer in enumerate(layers)]
    with torch.no_grad():
        t0 = time.time()
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        print(f"forward in {time.time() - t0:.1f}s", flush=True)
    for h in handles:
        h.remove()
    hs = torch.stack([captured["embeddings"]] + [captured[i] for i in range(len(layers))], 0)[:, 0].numpy()
    logits = out.logits[0].numpy()
    last = out.hidden_states[-1][0].numpy()
    if not np.allclose(last, hs[-1], atol=1e-5):
        print("note: output_hidden_states[-1] differs from the last layer's output (it is after the final norm)")

    np.savez(prefix + ".npz", input_ids=ids[0].numpy().astype(np.int32), hidden=hs, logits=logits)
    ids[0].numpy().astype("<i4").tofile(prefix + ".ids.bin")
    hs.astype("<f4").tofile(prefix + ".hidden.bin")
    logits.astype("<f4").tofile(prefix + ".logits.bin")
    meta = {
        "model_type": cfg.model_type,
        "seq": int(ids.shape[1]),
        "hidden_size": int(hs.shape[-1]),
        "num_layers": len(layers),
        "vocab_size": int(logits.shape[-1]),
        "text": a.text,
        "dtype": "float32",
    }
    with open(prefix + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    top = logits.argmax(-1)
    print("tokens:", ids[0].tolist())
    print("next-token argmax:", top.tolist())
    print("decoded continuation:", repr(tok.decode(top[-8:])))
    print(f"hidden {hs.shape}, logits {logits.shape}")
    print("mean |hidden| per layer:", [f"{np.abs(h).mean():.3f}" for h in hs])
    print("max |hidden| per layer:", [f"{np.abs(h).max():.1f}" for h in hs])


if __name__ == "__main__":
    main()
