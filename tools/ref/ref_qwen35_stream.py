"""fp32 reference of a Qwen3.5 text model that does not fit in memory in fp32 (Qwen3.5-4B is about
17 GB in fp32; an M3 has 16 GB): the model is built on the meta device and each decoder layer's
weights are read in fp32 just before it runs and dropped after it, so about one layer and the
embedding table are in memory at a time.

    python ref_qwen35_stream.py MODEL_DIR OUT_PREFIX [--seq 32] [--text "..."]
        writes the files of ref_logits.py (for hwnpu-dev check)
    python ref_qwen35_stream.py MODEL_DIR - --chat-file MSG.txt --generate 60
        greedy answer to the user message in MSG.txt (a chat without thinking), with a cache
    python ref_qwen35_stream.py MODEL_DIR - --nll-ids IDS.json
        the negative log likelihood of each next token of IDS.json, "position nll" per line

MODEL_DIR holds config.json, tokenizer.json and the checkpoint split by
model.safetensors.index.json (Qwen3_5ForConditionalGeneration's, tensors under
model.language_model.). Needs torch, transformers 5, safetensors and numpy; a forward of 30
tokens of Qwen3.5-4B takes about 12 seconds on an M3, a token of --generate about 12 more (each
reads the weights again).
"""
import argparse
import json
import time

import torch
from safetensors import safe_open

from ref_logits import TEXT, save

THINK_OFF = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
STOPS = (248044, 248046)  # <|endoftext|>, <|im_end|>


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--seq", type=int, default=32, help="tokens of the text (0: all)")
    ap.add_argument("--text", default=TEXT)
    ap.add_argument("--chat-file", help="the user message of a chat without thinking, in place of --text")
    ap.add_argument("--generate", type=int, default=0, help="greedy tokens to generate instead of writing files")
    ap.add_argument("--nll-ids", help="token ids (JSON): print the nll of each next token instead of writing files")
    a = ap.parse_args()
    from transformers import AutoConfig, AutoTokenizer, DynamicCache
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    tok = AutoTokenizer.from_pretrained(a.model_dir)
    if a.chat_file:
        a.text = "<|im_start|>user\n" + open(a.chat_file, encoding="utf-8").read() + THINK_OFF
    ids = tok(a.text, return_tensors="pt", add_special_tokens=True)["input_ids"]
    if a.seq and not (a.generate or a.nll_ids):
        if ids.shape[1] < a.seq:
            raise SystemExit(f"text has only {ids.shape[1]} tokens, need {a.seq} (--seq 0 takes them all)")
        ids = ids[:, : a.seq]
    cfg = AutoConfig.from_pretrained(a.model_dir).get_text_config()
    # eager attention keeps the reference independent of sdpa's kernels; a long prompt needs sdpa
    cfg._attn_implementation = "sdpa" if (a.generate or a.nll_ids) else "eager"
    with torch.device("meta"):
        model = m.Qwen3_5ForCausalLM(cfg)
    model.eval()
    model.model.rotary_emb = m.Qwen3_5TextRotaryEmbedding(config=cfg)  # its buffers, not on meta

    index = json.load(open(f"{a.model_dir}/model.safetensors.index.json"))["weight_map"]
    prefix = "model.language_model."
    files = {}

    def tensor(name):
        f = index[name]
        if f not in files:
            files[f] = safe_open(f"{a.model_dir}/{f}", framework="pt")
        return files[f].get_tensor(name).to(torch.float32)

    def load(module, sub):
        sd = {k: tensor(prefix + sub + k) for k, _ in module.named_parameters()}
        module.load_state_dict(sd, assign=True, strict=True)

    load(model.model.embed_tokens, "embed_tokens.")
    load(model.model.norm, "norm.")
    model.lm_head.weight = model.model.embed_tokens.weight  # tied

    captured = {}
    model.model.embed_tokens.register_forward_hook(lambda mod, i, o: captured.__setitem__("embeddings", o.detach()))
    for i, layer in enumerate(model.model.layers):

        def pre(mod, args, kwargs, i=i):
            load(mod, f"layers.{i}.")

        def post(mod, args, kwargs, out, i=i):
            captured[i] = (out[0] if isinstance(out, (tuple, list)) else out).detach().clone()
            mod.to_empty(device="meta")

        layer.register_forward_pre_hook(pre, with_kwargs=True)
        layer.register_forward_hook(post, with_kwargs=True)

    if a.nll_ids:
        ids = torch.tensor([json.load(open(a.nll_ids))])
        with torch.no_grad():
            h = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
            w = model.model.embed_tokens.weight
            for i in range(0, h.shape[0] - 1, 256):  # the logits 256 rows at a time
                lg = h[i : min(i + 256, h.shape[0] - 1)] @ w.T
                lp = torch.log_softmax(lg, -1)
                nll = -lp[torch.arange(lg.shape[0]), ids[0, i + 1 : i + 1 + lg.shape[0]]]
                for j, v in enumerate(nll.tolist()):
                    print(f"{i + j} {v:.4f}")
        return
    if a.generate:
        cache = DynamicCache(config=cfg)
        step, gen = ids, []
        print(f"prompt {ids.shape[1]} tokens", flush=True)
        for _ in range(a.generate):
            t0 = time.time()
            with torch.no_grad():
                out = model(input_ids=step, past_key_values=cache, use_cache=True, logits_to_keep=1)
            nxt = int(out.logits[0, -1].argmax())
            gen.append(nxt)
            step = torch.tensor([[nxt]])
            print(f"{time.time() - t0:.1f}s {tok.decode(gen)!r}", flush=True)
            if nxt in STOPS:
                break
        return
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
    print(f"forward in {time.time() - t0:.1f}s", flush=True)
    n = cfg.num_hidden_layers
    hs = torch.stack([captured["embeddings"]] + [captured[i] for i in range(n)], 0)[:, 0].numpy()
    save(a.out.removesuffix(".npz"), ids, hs, out.logits[0].numpy(), cfg, a.text, False, tok)


if __name__ == "__main__":
    main()
