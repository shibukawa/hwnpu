"""Reference activations of a Hugging Face model, for checking a conversion.

    python ref_logits.py MODEL_DIR OUT_PREFIX [--seq 32] [--text "..."] [--encoder] [--attn sdpa]

MODEL_DIR holds config.json, the tokenizer and the checkpoint (and modeling_*.py for models
that need trust_remote_code; symlinks to downloaded files are fine). Runs on the CPU in float32
and writes, for readers without numpy (hwnpu-dev check and the Go tests), little endian and row major:

    OUT_PREFIX.ids.bin     int32 (seq)                       token ids
    OUT_PREFIX.hidden.bin  float32 (layers + 1, seq, hidden) token embeddings, then each layer's output
    OUT_PREFIX.logits.bin  float32 (seq, vocab) logits, or (seq, hidden) with --encoder
    OUT_PREFIX.json        model_type, seq, sizes, text, dtype, output

and the same arrays in OUT_PREFIX.npz. The hidden states are taken with hooks rather than from
output_hidden_states, so that they mean the same for every model, as hwnpu's programs expect:
entry 0 is the output of the token embedding table (before position embeddings, scaling or an
embedding norm, which the program does: an embedding module with an embed_scale, as Gemma's,
has it divided out), and entry i + 1 the output of layer i (before a final norm;
output_hidden_states ends after it in many implementations). A looped model (Ouro, whose layers
run total_ut_steps times) has an entry for each layer in each pass, the last layer of a pass but the
last after the final norm the next pass starts from, and the logits of the last pass.

Causal language models load with AutoModelForCausalLM. --encoder loads with AutoModel instead
(BERT and the like) and writes the last hidden state as the output; there is no padding, so the
attention mask is all ones. --attn picks transformers' attention implementation: eager by default,
sdpa for models whose remote code has no other (PLaMo's PlamoBiModel). Needs torch and transformers.
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
    ap.add_argument("--encoder", action="store_true", help="an encoder: AutoModel, last hidden state as output")
    ap.add_argument("--attn", default="eager", help="attention implementation (eager, sdpa)")
    a = ap.parse_args()
    prefix = a.out.removesuffix(".npz")

    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
    ids = tok(a.text, return_tensors="pt", add_special_tokens=True)["input_ids"][:, : a.seq]
    if ids.shape[1] < a.seq:
        raise SystemExit(f"text has only {ids.shape[1]} tokens, need {a.seq}")
    t0 = time.time()
    auto = AutoModel if a.encoder else AutoModelForCausalLM
    # transformers before 4.56 names the dtype torch_dtype (Ouro's remote code needs one of them)
    import transformers

    major, minor = (int(v) for v in transformers.__version__.split(".")[:2])
    dtype = {"dtype" if (major, minor) >= (4, 56) else "torch_dtype": torch.float32}
    model = auto.from_pretrained(a.model_dir, trust_remote_code=True, attn_implementation=a.attn, **dtype).eval()
    print(f"loaded in {time.time() - t0:.1f}s: {sum(p.numel() for p in model.parameters()) / 1e9:.3f} B parameters", flush=True)
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    layers = decoder_layers(model, cfg.num_hidden_layers)

    captured = {}
    handles = hook_hidden(model, layers, captured)
    passes = getattr(cfg, "total_ut_steps", 1) if cfg.model_type == "ouro" else 1
    kwargs = {}
    if passes > 1:
        # a looped model (Ouro): its layers run once a pass, the final norm after each; the logits
        # of the last pass, as early_exit_threshold 1.0 means
        handles += hook_passes(model, captured)
        # its OuroModel returns a tuple, which output_hidden_states's wrapper cannot fill
        kwargs = {"exit_at_step": passes - 1, "output_hidden_states": False}
    else:
        kwargs["output_hidden_states"] = True
    with torch.no_grad():
        t0 = time.time()
        out = model(input_ids=ids, use_cache=False, **kwargs)
        print(f"forward in {time.time() - t0:.1f}s", flush=True)
    for h in handles:
        h.remove()
    hs = stack_hidden(captured, len(layers)) if passes == 1 else stack_passes(captured, len(layers), passes)
    logits = (out.last_hidden_state if a.encoder else out.logits)[0].numpy()
    last = out.hidden_states[-1][0].numpy() if out.hidden_states is not None else hs[-1]
    if not np.allclose(last, hs[-1], atol=1e-5):
        print("note: output_hidden_states[-1] differs from the last layer's output (it is after the final norm)")
    save(prefix, ids, hs, logits, cfg, a.text, a.encoder, tok)


def hook_hidden(model, layers, captured):
    """Hooks that put the token embeddings (with an embed_scale divided out) and each layer's
    output in captured, by "embeddings" and layer index."""

    def first(x):
        return x[0] if isinstance(x, (tuple, list)) else x

    def hook(key):
        def f(module, _args, out):
            out = first(out).detach()
            if key == "embeddings" and hasattr(module, "embed_scale"):
                out = out / module.embed_scale.to(out.dtype)
            captured[key] = out

        return f

    emb = model.get_input_embeddings()
    if hasattr(emb, "embed_scale"):
        print(f"embedding scale {float(emb.embed_scale):.6g} divided out of entry 0", flush=True)
    handles = [emb.register_forward_hook(hook("embeddings"))]
    return handles + [layer.register_forward_hook(hook(i)) for i, layer in enumerate(layers)]


def hook_passes(model, captured):
    """For a looped model: every output of each layer (one a pass) by "runs", layer index, and of
    the final norm by "norms"."""

    def first(x):
        return x[0] if isinstance(x, (tuple, list)) else x

    def hook(key):
        def f(module, _args, out):
            captured.setdefault("runs", {}).setdefault(key, []).append(first(out).detach())

        return f

    layers = decoder_layers(model, model.config.num_hidden_layers)
    handles = [layer.register_forward_hook(hook(i)) for i, layer in enumerate(layers)]
    return handles + [model.model.norm.register_forward_hook(hook("norm"))]


def stack_passes(captured, n, passes):
    """The hidden states of a looped model as hwnpu's layers of every pass (internal/llm/ouro): the
    token embeddings, then each layer's output in each pass, the last layer of a pass but the last
    after the final norm, which the next pass starts from."""
    import torch

    runs = captured["runs"]
    for i in range(n):
        if len(runs[i]) != passes:
            raise SystemExit(f"layer {i} ran {len(runs[i])} times, not {passes}")
    out = [captured["embeddings"]]
    for p in range(passes):
        for i in range(n):
            out.append(runs["norm"][p] if i == n - 1 and p < passes - 1 else runs[i][p])
    return torch.stack(out, 0)[:, 0].numpy()


def stack_hidden(captured, n):
    import torch

    return torch.stack([captured["embeddings"]] + [captured[i] for i in range(n)], 0)[:, 0].numpy()


def save(prefix, ids, hs, logits, cfg, text, encoder, tok, extra=None):
    """Writes the reference files. extra holds more arrays by name, written as PREFIX.NAME.bin
    (float32): the values of program inputs besides x that hwnpu-dev check feeds a layer."""
    extra = extra or {}
    np.savez(prefix + ".npz", input_ids=ids[0].numpy().astype(np.int32), hidden=hs, logits=logits, **extra)
    for name, v in extra.items():
        np.asarray(v, dtype="<f4").tofile(f"{prefix}.{name}.bin")
    ids[0].numpy().astype("<i4").tofile(prefix + ".ids.bin")
    hs.astype("<f4").tofile(prefix + ".hidden.bin")
    logits.astype("<f4").tofile(prefix + ".logits.bin")
    meta = {
        "model_type": cfg.model_type,
        "seq": int(ids.shape[1]),
        "hidden_size": int(hs.shape[-1]),
        "num_layers": int(hs.shape[0] - 1),
        "vocab_size": int(cfg.vocab_size),
        "text": text,
        "dtype": "float32",
        "output": "hidden" if encoder else "logits",
    }
    with open(prefix + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print("tokens:", ids[0].tolist())
    if not encoder:
        top = logits.argmax(-1)
        print("next-token argmax:", top.tolist())
        print("decoded continuation:", repr(tok.decode(top[-8:])))
    print(f"hidden {hs.shape}, logits {logits.shape}")
    print("mean |hidden| per layer:", [f"{np.abs(h).mean():.3f}" for h in hs])
    print("max |hidden| per layer:", [f"{np.abs(h).max():.1f}" for h in hs])


if __name__ == "__main__":
    main()
