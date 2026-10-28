"""Reference activations of LFM3-1.7B from the Hugging Face implementation, for checking a conversion.

    python ref_logits.py MODEL_DIR OUT.npz [--seq 32] [--text "..."]

MODEL_DIR holds config.json, modeling_lfm3.py, tokenizer.json and model.safetensors (a symlink to
the downloaded file is fine). Runs on the CPU in float32 and writes the token ids, the embedding
output, the hidden state after every layer (before the final norm) and the logits. Needs torch and
transformers.
"""
import argparse
import time

import numpy as np

TEXT = "The Neural Engine runs small language models at low power. hwnpu is a Go library for CPU integrated NPUs: Apple, Intel and AMD."


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--text", default=TEXT)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
    ids = tok(a.text, return_tensors="pt", add_special_tokens=True)["input_ids"][:, : a.seq]
    if ids.shape[1] < a.seq:
        raise SystemExit(f"text has only {ids.shape[1]} tokens, need {a.seq}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, trust_remote_code=True, dtype=torch.float32, attn_implementation="eager").eval()
    print(f"loaded in {time.time() - t0:.1f}s: {sum(p.numel() for p in model.parameters()) / 1e9:.3f} B parameters", flush=True)
    with torch.no_grad():
        t0 = time.time()
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        print(f"forward in {time.time() - t0:.1f}s", flush=True)
    hs = torch.stack(out.hidden_states, 0)[:, 0].numpy()  # (layers + 1, seq, d): embeddings, then after each layer
    logits = out.logits[0].numpy()
    np.savez(a.out, input_ids=ids[0].numpy().astype(np.int32), hidden=hs, logits=logits)
    # raw copies for readers without numpy (the Go tests): little endian, row major
    base = a.out.removesuffix(".npz")
    ids[0].numpy().astype("<i4").tofile(base + ".ids.bin")
    hs.astype("<f4").tofile(base + ".hidden.bin")
    logits.astype("<f4").tofile(base + ".logits.bin")
    top = logits.argmax(-1)
    print("tokens:", ids[0].tolist())
    print("next-token argmax:", top.tolist())
    print("decoded continuation:", repr(tok.decode(top[-8:])))
    print(f"hidden {hs.shape}, logits {logits.shape}")
    print("mean |hidden| per layer:", [f"{np.abs(h).mean():.3f}" for h in hs])
    print("max |hidden| per layer:", [f"{np.abs(h).max():.1f}" for h in hs])


if __name__ == "__main__":
    main()
