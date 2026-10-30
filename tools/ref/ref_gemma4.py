"""Reference activations of Gemma 4's text model, for checking internal/llm/gemma4.

    python ref_gemma4.py MODEL_DIR OUT_PREFIX [--seq 32] [--text "..."]

MODEL_DIR holds config.json, the tokenizer and model.safetensors of google/gemma-4-E2B-it (or
another Gemma 4 with per-layer embeddings). Writes what ref_logits.py writes, and the inputs a
layer of the program takes besides the hidden state, for hwnpu-check to feed:

    OUT_PREFIX.per_layer_inputs.bin   float32 (seq, layers * per-layer dim) each layer's input
                                      from the per-layer embeddings
    OUT_PREFIX.shared_k_TYPE.bin      float32 (1, kv heads, seq, head_dim) the keys of the last
    OUT_PREFIX.shared_v_TYPE.bin      TYPE (sliding, full) layer before the shared ones, and its
                                      values, which the shared layers reuse

The whole checkpoint in float32 (5.1 B parameters, 20 GB) does not fit a 16 GB Mac. This builds
transformers' Gemma4ForCausalLM from the text config in float32 without the vision and audio
towers, loads the language model's tensors from the checkpoint one by one, and replaces the
per-layer embedding table (2.3 B parameters) with a module that reads the rows of the tokens from
the checkpoint: 9 GB in all. The forward pass is transformers' own. Needs torch, transformers
(with Gemma 4) and safetensors.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_logits import TEXT, decoder_layers, hook_hidden, save, stack_hidden  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--text", default=TEXT)
    a = ap.parse_args()
    prefix = a.out.removesuffix(".npz")

    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer, Gemma4ForCausalLM

    tok = AutoTokenizer.from_pretrained(a.model_dir)
    ids = tok(a.text, return_tensors="pt", add_special_tokens=True)["input_ids"][:, : a.seq]
    if ids.shape[1] < a.seq:
        raise SystemExit(f"text has only {ids.shape[1]} tokens, need {a.seq}")

    cfg = AutoConfig.from_pretrained(a.model_dir).get_text_config()
    table = "model.language_model.embed_tokens_per_layer.weight"
    per_layer_vocab = cfg.vocab_size_per_layer_input
    cfg.vocab_size_per_layer_input = 1  # the table is replaced below
    cfg._attn_implementation = "eager"
    t0 = time.time()
    torch.manual_seed(0)
    model = Gemma4ForCausalLM(cfg).to(torch.float32).eval()
    print(f"built in {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    path = os.path.join(a.model_dir, "model.safetensors")
    state = model.state_dict()
    loaded, skipped = set(), []
    st = safe_open(path, "pt")
    for key in st.keys():
        if not key.startswith("model.language_model.") or key == table:
            continue
        name = "model." + key.removeprefix("model.language_model.")
        if name not in state:
            skipped.append(key)
            continue
        with torch.no_grad():
            state[name].copy_(st.get_tensor(key).to(torch.float32))
        loaded.add(name)
    missing = sorted(set(state) - loaded - {"lm_head.weight", "model.embed_tokens_per_layer.weight"})
    if missing:
        raise SystemExit(f"not in the checkpoint: {missing[:8]} ({len(missing)})")
    print(f"loaded {len(loaded)} tensors in {time.time() - t0:.1f}s; skipped {len(skipped)} (the shared layers' k_proj, v_proj, k_norm)", flush=True)
    model.tie_weights()
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()

    class Rows(torch.nn.Module):
        """The per-layer embeddings (Gemma4TextScaledWordEmbedding) read from the checkpoint."""

        def __init__(self, scale):
            super().__init__()
            self.slice = st.get_slice(table)
            self.scale = scale

        def forward(self, input_ids):
            flat = [int(i) for i in input_ids.reshape(-1)]
            assert all(0 <= i < per_layer_vocab for i in flat)
            rows = torch.stack([self.slice[i : i + 1][0].to(torch.float32) for i in flat])
            return (rows * self.scale).reshape(*input_ids.shape, -1)

    model.model.embed_tokens_per_layer = Rows(model.model.embed_tokens_per_layer.embed_scale.to(torch.float32))

    layers = decoder_layers(model, cfg.num_hidden_layers)
    captured, per_layer, shared = {}, {}, {}

    def per_layer_hook(i):
        def f(_module, args, kwargs):
            per_layer[i] = (args[1] if len(args) > 1 else kwargs["per_layer_input"]).detach()

        return f

    def shared_hook(_module, _args, kwargs):
        for t, (k, v) in kwargs["shared_kv_states"].items():
            shared[t] = (k.detach().clone(), v.detach().clone())

    handles = hook_hidden(model, layers, captured)
    handles += [layer.register_forward_pre_hook(per_layer_hook(i), with_kwargs=True) for i, layer in enumerate(layers)]
    first_shared = cfg.num_hidden_layers - cfg.num_kv_shared_layers
    if cfg.num_kv_shared_layers:
        handles.append(layers[first_shared].self_attn.register_forward_pre_hook(shared_hook, with_kwargs=True))
    with torch.no_grad():
        t0 = time.time()
        out = model(input_ids=ids, use_cache=False)
        print(f"forward in {time.time() - t0:.1f}s", flush=True)
    for h in handles:
        h.remove()
    hs = stack_hidden(captured, len(layers))
    logits = out.logits[0].numpy()
    extra = {"per_layer_inputs": torch.stack([per_layer[i][0] for i in range(len(layers))], 1).reshape(ids.shape[1], -1).numpy()}
    for t, (k, v) in shared.items():
        short = t.removesuffix("_attention")
        extra[f"shared_k_{short}"], extra[f"shared_v_{short}"] = k.numpy(), v.numpy()
        print(f"shared {t}: keys {tuple(k.shape)}", flush=True)
    save(prefix, ids, hs, logits, cfg, a.text, False, tok, extra)


if __name__ == "__main__":
    main()
