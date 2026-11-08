#!/usr/bin/env python3
"""GPTQ for iron_q4's formats: choose a language model's 4-bit codes (q4nx) against its
activations, for the export to read in place of rounding each weight to its nearest level.

iron_q4.quantize rounds every weight to the nearest of its group's 16 levels: the weight error
is small, the error in the layer's output is not the least. GPTQ (Frantar et al. 2022) quantizes
a weight's in-features in order and moves each one's error onto the in-features not yet
quantized, weighted by H = E[x x^T] of the layer's inputs, so the output's error is what it
minimizes. The format is the same (the same groups of 32 in-features, their bf16 scale and
minimum from iron_q4._fit, fitted when GPTQ reaches the group): only the codes differ, and so
the scale and minimum fitted to the weights GPTQ has already corrected. No cost at run time.

Runs on the CPU in the reference environment (PyTorch, Hugging Face, fp32):

  ~/.cache/hwnpu/dev/ref-venv/bin/python tools/amd/q4_gptq.py lfm3 ~/.cache/hwnpu/models/LFM3-1.7B OUT

writes OUT/codes (each weight's blocks, named by a hash of its shape and bf16 values) and prints
the KL of the quantized model's logits from the bf16 weights' on the evaluation texts (EVAL).
--method rtn measures plain rounding, --method codes an OUT/codes written before. The export
reads the codes, and rounds the weights they do not name:

  python tools/amd/iron_export_lm.py lfm3 CKPT EXPORT --q4 --q4-codes OUT/codes

Calibration is sequential: a pass runs the calibration texts through the model, collects H for
--layers-a-pass layers' quantized projections, quantizes them, and the next pass sees those
layers quantized. The projections q8 holds (attention, LFM3's convolutions) are rounded first,
as the export does, so the later layers' inputs are the device's. H leaves out each text's BOS
(--skip-first) and is damped by 0.1 of its mean diagonal (--damp): GPTQ's usual 0.01 fitted the
calibration texts and made an unseen one (the GPL) worse than rounding.

--fmt-q4, --fmt-q8 and --head lutN do the same for Apple's lookup tables (internal/palette's
Uniform tables of 16 rows): GPTQ chooses each weight's entry of its fixed table, and OUT/codes
holds the indices and tables (KEY.idx, KEY.tab) that hwnpu-convert -codes reads.

The calibration texts (CALIB) include general English and Japanese, in ~/models/calib:
wikitext-2-raw-v1's train split (wikitext2-train.txt) and test split (wikitext2-test.txt, from
character 200000, evaluation only), wiki40b-ja's validation split cleaned of its markers, the
first half of its articles (wiki40b-ja-calib.txt) and the last 200 (wiki40b-ja-eval.txt), from
the parquet files on Hugging Face (Salesforce/wikitext, range3/wiki40b-ja) read with pyarrow.
Missing files are left out.

See docs/amd-npu-models.md (13, GPTQ).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iron_q4  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# By model: the token put before every text, where the decoder layers are, the projections in
# q4nx and in q8 (the export's defaults: iron_hostple, iron_lfm3), and whether the head (tied to
# the embeddings) is q4nx.
MODELS = {
    "gemma4": dict(
        bos=2, layers="model.language_model.layers", remote=False, lean=True,
        q4=("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"),
        q8=("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"),
    ),
    "plamo3": dict(
        bos=1, layers="model.layers.layers", remote=True,
        q4=("mlp.gate_up_proj", "mlp.down_proj"),
        q8=("mixer.qkv_proj", "mixer.o_proj"),
    ),
    # lfm3vl's text model (iron_lfm3.Lfm3 in the export), run on text alone
    "lfm3vl": dict(
        bos=124894, layers="model.language_model.layers", remote=True, auto="AutoModelForImageTextToText",
        text="model.language_model", head="lm_head",
        q4=("feed_forward.w1", "feed_forward.w3", "feed_forward.w2"),
        q8=("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "conv.in_proj", "conv.out_proj"),
    ),
    # lfm3audio's text model (iron_lfm3.Lfm3 in the export), run on text alone
    "lfm3audio": dict(
        bos=124894, layers="model.language_model.layers", remote=True, auto="AutoModelForSeq2SeqLM",
        text="model.language_model", head="lm_head",
        q4=("feed_forward.w1", "feed_forward.w3", "feed_forward.w2"),
        q8=("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "conv.in_proj", "conv.out_proj"),
    ),
    # lfm3omni's text model (iron_lfm3.Lfm3 in the export), run on text alone
    "lfm3omni": dict(
        bos=124894, layers="model.language_model.layers", remote=True, auto="AutoModelForImageTextToText",
        text="model.language_model", head="lm_head",
        q4=("feed_forward.w1", "feed_forward.w3", "feed_forward.w2"),
        q8=("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "conv.in_proj", "conv.out_proj"),
    ),
    # Qwen3.5 (Qwen3_5ForConditionalGeneration)'s text model (iron_qwen35.Qwen35), run on text
    # alone. Its <|endoftext|> stands for the BOS it does not have. Gated DeltaNet's a and b
    # (a row a value head) are left out of q8 here: the export rounds them in the fused
    # projection, which these 16-row weights alone are not whole blocks of.
    "qwen35": dict(
        bos=248044, layers="model.language_model.layers", remote=False, auto="AutoModelForImageTextToText",
        text="model.language_model", head="lm_head",
        q4=("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"),
        q8=("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"),
    ),
    "lfm3": dict(
        bos=124894, layers="model.layers", remote=True,
        q4=("feed_forward.w1", "feed_forward.w3", "feed_forward.w2"),
        q8=("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "conv.in_proj", "conv.out_proj"),
    ),
}

# Calibration and evaluation texts, kept apart. The repository's own notes alone (8K tokens,
# English, Japanese and Go) fitted them and made the GPL's KL worse than rounding (LFM3 0.105 ->
# 0.129): general English and Japanese (wikitext-2's training split and wiki40b-ja's validation
# split, ~/models/calib, CALIB_DATA in docs/amd-npu-models.md) are added when present.
CALIB_DATA = Path.home() / "models" / "calib"
CALIB = ["MODEL_ARCHITECTURE.md", "docs/design.md", "internal/chat/engine.go",
         str(CALIB_DATA / "wikitext2-train.txt"), str(CALIB_DATA / "wiki40b-ja-calib.txt")]
EVAL = ["README.md", "/usr/share/common-licenses/GPL-3", str(CALIB_DATA / "wiki40b-ja-eval.txt"),
        str(CALIB_DATA / "wikitext2-test.txt"), "internal/llm/iron/model.go"]


def key(w) -> str:
    """The name of a weight's codes: a hash of its shape and bf16 values, which the export
    computes from the weight it is about to quantize (iron_q4.load_codes)."""
    return iron_q4.weight_key(w)


# --- GPTQ ------------------------------------------------------------------------------------


# The symmetric formats' largest code (q6 and q8: w = scale * code, code -SYM..SYM).
SYM = {"q6": 31, "q8": 127}


def _group_params(g: np.ndarray, fmt: str):
    """A group's scale and minimum (q4nx, iron_q4._fit) or scale (q8) for g (N, 32)."""
    if fmt == "q4":
        return iron_q4._fit(g)
    s = (np.abs(g).max(-1) / SYM[fmt]).astype(bfloat16).astype(np.float32)
    return s, np.zeros_like(s)


# Apple's lookup tables (internal/palette, the Neural Engine's lutN), to measure the same on the
# CPU: a table of 2^bits fp16 levels for each LUT_GROUP rows (whole rows), the levels evenly
# between the block's smallest and largest weight (palette.Uniform, the converter's default).
LUT_BITS = {"lut4": 4, "lut6": 6, "lut8": 8}
LUT_GROUP = 16


def lut_tables(w, fmt: str) -> np.ndarray:
    """palette.Uniform's tables of w (N, K): (N / LUT_GROUP, 2^bits) float32 of fp16 values."""
    k = 1 << LUT_BITS[fmt]
    g = np.asarray(w, np.float32).reshape(-1, LUT_GROUP * np.shape(w)[1])
    lo, hi = g.min(1), g.max(1)
    t = lo[:, None] + (hi - lo)[:, None] * (np.arange(k, dtype=np.float32) / np.float32(k - 1))[None, :]
    return t.astype(np.float16).astype(np.float32)


def _lut_nearest(x, lo, step, T):
    """For x (N,) torch, each row's table T (N, k) evenly spaced from lo (N,) by step (N,) up to
    the fp16 rounding: the index of the nearest level (the even estimate, then its neighbours)."""
    import torch

    k = T.shape[1]
    i = torch.clamp(torch.round((x - lo) / torch.where(step > 0, step, 1)), 0, k - 1).long()
    cand = torch.stack([torch.clamp(i - 1, 0, k - 1), i, torch.clamp(i + 1, 0, k - 1)], 1)
    vals = torch.gather(T, 1, cand)
    best = torch.argmin((vals - x[:, None]).abs(), 1)
    return cand.gather(1, best[:, None])[:, 0]


def lut_wkmeans(w, fmt: str, d, iters: int = 8) -> np.ndarray:
    """Tables fitted to w (N, K) against the activations: Lloyd's algorithm from the Uniform
    levels (palette.KMeansMinMax), each weight counted by d (K,), its in-feature's E[x^2] (H's
    diagonal), so the levels go where the output's error is (model-fit.md T8, 1)."""
    import torch

    k = 1 << LUT_BITS[fmt]
    N, K = np.shape(w)
    T = torch.as_tensor(lut_tables(w, fmt))
    V = torch.as_tensor(np.asarray(w, np.float32)).reshape(-1, LUT_GROUP * K)
    wt = torch.as_tensor(np.tile(np.asarray(d, np.float32), LUT_GROUP))
    out = torch.empty_like(T)
    step = max(1, (1 << 22) // (LUT_GROUP * K))  # blocks at a time: ~16 MB of values
    for b0 in range(0, V.shape[0], step):
        v, t = V[b0 : b0 + step], T[b0 : b0 + step].clone()
        for _ in range(iters):
            i = torch.searchsorted(((t[:, 1:] + t[:, :-1]) / 2).contiguous(), v)
            num = torch.zeros_like(t).scatter_add_(1, i, v * wt)
            den = torch.zeros_like(t).scatter_add_(1, i, wt.expand_as(v).contiguous())
            t = torch.where(den > 0, num / torch.where(den > 0, den, 1), t)
            t, _ = torch.sort(t, 1)
        out[b0 : b0 + step] = t
    return out.numpy().astype(np.float16).astype(np.float32)


def _lut_nearest_sorted(x, T):
    """For x (N,) torch, the index of the nearest level of each row's sorted table T (N, k)."""
    import torch

    mid = ((T[:, 1:] + T[:, :-1]) / 2).contiguous()
    return torch.searchsorted(mid, x[:, None].contiguous())[:, 0]


def lut_rtn(w, fmt: str) -> np.ndarray:
    """w (N, K) with each weight its table's nearest level (palette.QuantizeBits, Uniform)."""
    import torch

    T = lut_tables(w, fmt)
    W = torch.as_tensor(np.asarray(w, np.float32))
    N, K = W.shape
    Tr = torch.as_tensor(np.repeat(T, LUT_GROUP, 0))
    lo, step = Tr[:, 0], (Tr[:, -1] - Tr[:, 0]) / (Tr.shape[1] - 1)
    out = torch.empty_like(W)
    for c in range(K):
        out[:, c] = torch.gather(Tr, 1, _lut_nearest(W[:, c], lo, step, Tr)[:, None])[:, 0]
    return out.numpy()


def rtn(w, fmt: str) -> np.ndarray:
    """w rounded to fmt and back (float32), as the export (iron_q4) or the converter (palette)."""
    if fmt in LUT_BITS:
        return lut_rtn(w, fmt)
    n, k = np.shape(w)
    return iron_q4.dequantize_fmt(iron_q4.quantize_fmt(w, fmt), n, k, fmt)


def gptq(w, H, fmt: str = "q4", block: int = 128, damp: float = 0.01, lut_fit: str = "uniform"):
    """GPTQ of w (N, K) float32 against H (K, K). q4 and q8: the codes (N, K) and each group's
    scale and minimum (N, K / 32), as iron_q4.pack (q4) or pack8 (q8) lays them out, the group's
    parameters fitted when GPTQ reaches it. lutN: the indices (N, K) and the tables, fixed first
    (lut_tables of w, or lut_wkmeans with lut_fit "wkmeans"), and the quantized weights."""
    import torch

    G = iron_q4.GROUP
    W = torch.as_tensor(np.asarray(w, np.float32)).clone()
    N, K = W.shape
    H = torch.as_tensor(H, dtype=torch.float64).clone()
    diag = torch.diag(H).float().numpy().copy()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(K, dtype=H.dtype)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True).float()
    lut = fmt in LUT_BITS
    codes = torch.empty((N, K), dtype=torch.int16)
    if lut:
        T = lut_wkmeans(w, fmt, diag) if lut_fit == "wkmeans" else lut_tables(w, fmt)
        Tr = torch.as_tensor(np.repeat(T, LUT_GROUP, 0))
        Q = torch.empty_like(W)
    else:
        lo, hi = (0, 15) if fmt == "q4" else (-SYM[fmt], SYM[fmt])
        scale = np.empty((N, K // G), np.float32)
        mn = np.zeros((N, K // G), np.float32)
    for i1 in range(0, K, block):
        i2 = min(i1 + block, K)
        W1 = W[:, i1:i2].clone()
        E1 = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            x = W1[:, i]
            if lut:
                c = _lut_nearest_sorted(x, Tr)
                q = torch.gather(Tr, 1, c[:, None])[:, 0]
                Q[:, col] = q
            else:
                if col % G == 0:
                    s, m = _group_params(W1[:, i : i + G].numpy(), fmt)
                    scale[:, col // G], mn[:, col // G] = s, m
                    st = torch.as_tensor(np.where(s != 0, s, 1))
                    mt = torch.as_tensor(m)
                    s0 = torch.as_tensor(s)
                c = torch.clamp(torch.round((x - mt) / st), lo, hi)
                q = mt + s0 * c
            codes[:, col] = c.to(torch.int16)
            err = (x - q) / Hi[i, i]
            W1[:, i:] -= err[:, None] * Hi[i, i:][None, :]
            E1[:, i] = err
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    if lut:
        return codes.numpy(), T, Q.numpy()
    return codes.numpy(), scale, mn


def blocks_of(codes, scale, mn, fmt: str) -> np.ndarray:
    if fmt == "q4":
        return iron_q4.pack(codes.astype(np.uint8), scale, mn)
    if fmt == "q6":
        return iron_q4.pack6(codes.astype(np.int8), scale)
    return iron_q4.pack8(codes.astype(np.int8), scale)


# --- the model -------------------------------------------------------------------------------


class TextLM:
    """A multimodal checkpoint's text model alone: its decoder (text, a dotted path) and head,
    called with token ids as a causal LM is (the towers are not run)."""

    def __init__(self, m, text: str, head: str):
        self.m, self.text, self.head = m, _get(m, text), _get(m, head)

    def __getattr__(self, name):
        return getattr(self.m, name)

    def __call__(self, input_ids):
        import types

        return types.SimpleNamespace(logits=self.head(self.text(input_ids=input_ids)[0]))

    def get_output_embeddings(self):
        return self.head


class TableRows:
    """A lookup table (an embedding) read from the checkpoint's file as the rows are looked up,
    times scale, in fp32: what Gemma 4's scaled embeddings give, without the table in memory."""

    def __init__(self, path: Path, name: str, scale: float):
        import json as _json
        import struct

        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            e = _json.loads(f.read(n))[name]
        if e["dtype"] != "BF16":
            raise ValueError(f"{name}: {e['dtype']}, expected BF16")
        self.rows = np.memmap(path, np.uint16, "r", 8 + n + e["data_offsets"][0], tuple(e["shape"]))
        self.scale = scale

    def __call__(self, ids):
        import torch

        r = np.ascontiguousarray(self.rows[ids.numpy().reshape(-1)]).astype(np.uint32) << 16
        x = torch.from_numpy(r.view(np.float32)).reshape(*ids.shape, -1)
        return x * torch.tensor(self.scale, dtype=torch.float32)


def load_gemma4_lean(path: Path):
    """Gemma 4's text model in fp32 with its two embedding tables (2.75B of E2B's 4.6B) left in
    the file (TableRows): E2B in about 9 GB instead of 20. The head is the embeddings' own fp32
    copy. As Gemma4ForCausalLM.from_pretrained in fp32, but built with a vocabulary of 1 so that
    the tables are never made."""
    import copy

    import torch
    import transformers
    from safetensors import safe_open
    from transformers.models.gemma4 import modeling_gemma4 as g4

    cfg = transformers.AutoConfig.from_pretrained(path).get_text_config()
    small = copy.deepcopy(cfg)
    small.vocab_size = small.vocab_size_per_layer_input = 1
    small.pad_token_id = 0
    m = g4.Gemma4ForCausalLM(small)
    m.config = m.model.config = cfg
    files = sorted(path.glob("*.safetensors"))
    if len(files) != 1:
        raise ValueError(f"{path}: {len(files)} safetensors files, expected 1")
    pre = "model.language_model."
    del m.model.embed_tokens, m.model.embed_tokens_per_layer
    m.model.embed_tokens = TableRows(files[0], pre + "embed_tokens.weight", cfg.hidden_size**0.5)
    m.model.embed_tokens_per_layer = TableRows(files[0], pre + "embed_tokens_per_layer.weight",
                                               cfg.hidden_size_per_layer_input**0.5)
    m.lm_head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    m.vocab_size = m.model.vocab_size = cfg.vocab_size
    with safe_open(files[0], "pt") as f:
        names = set(f.keys())
        for name, p in m.named_parameters():
            src = pre + "embed_tokens.weight" if name == "lm_head.weight" else pre + name.removeprefix("model.")
            if src not in names:
                # the KV-shared layers' k and v (and their norms) are not in the checkpoint
                if ".self_attn." in name and m.model.layers[int(name.split(".")[2])].self_attn.is_kv_shared_layer:
                    continue
                raise ValueError(f"{name}: {src} not in the checkpoint")
            p.data = f.get_tensor(src).float()
        # buffers the checkpoint holds (each layer's layer_scalar)
        for name, b in m.named_buffers():
            src = pre + name.removeprefix("model.")
            if src in names:
                b.copy_(f.get_tensor(src).float())
    m.eval()
    # the layers where MODELS["gemma4"] looks for them (not a submodule: state_dict would recurse)
    object.__setattr__(m.model, "language_model", m.model)
    return m


def load(path: Path, spec):
    import torch
    import transformers

    torch.set_grad_enabled(False)
    if spec.get("lean"):
        m = load_gemma4_lean(path)
        return m, transformers.AutoTokenizer.from_pretrained(path)
    auto = getattr(transformers, spec.get("auto", "AutoModelForCausalLM"))
    m = auto.from_pretrained(path, dtype=torch.float32, trust_remote_code=spec["remote"])
    m.eval()
    if "text" in spec:
        m = TextLM(m, spec["text"], spec["head"])
    tok = transformers.AutoTokenizer.from_pretrained(path, trust_remote_code=spec["remote"])
    # the head is tied to the embeddings: its own copy, to quantize without touching the lookup
    head = m.get_output_embeddings()
    head.weight = torch.nn.Parameter(head.weight.detach().clone())
    return m, tok


def _get(m, dotted: str):
    for part in dotted.split("."):
        m = getattr(m, part, None)
        if m is None:
            return None
    return m


def projections(m, spec):
    """By layer, its quantized projections: [(name, module, fmt)], and the head."""
    layers = _get(m, spec["layers"])
    out = []
    for i, layer in enumerate(layers):
        mine = []
        for fmt in ("q8", "q4"):
            for p in spec[fmt]:
                lin = _get(layer, p)
                if lin is not None:
                    mine.append((f"{spec['layers']}.{i}.{p}", lin, fmt))
        out.append(mine)
    return out


def texts(paths, tok, bos: int, length: int, count: int):
    """Up to count sequences of length tokens (BOS first) cut from the files, in turn."""
    rr = []
    for p in paths:
        p = Path(p) if Path(p).is_absolute() else REPO / p
        if not p.is_file():
            print(f"{p}: missing, left out")
            continue
        # the start of a long file is enough (16 characters a token is more than any tokenizer takes)
        ids = tok.encode(p.read_text()[: 16 * length * count], add_special_tokens=False)
        rr.append([[bos] + ids[s : s + length - 1] for s in range(0, len(ids) - length + 2, length - 1)])
    # the files' sequences in turn, so a small count still mixes them
    out = []
    while len(out) < count and any(rr):
        for r in rr:
            if r and len(out) < count:
                out.append(r.pop(0))
    return out


def logits_of(m, ids):
    import torch

    return m(input_ids=torch.tensor([ids])).logits[0].float()


def kl_stats(ref, q):
    """Mean, median and 95th percentile of KL(ref || q) per position, and top-1 agreement."""
    import torch

    lp = torch.log_softmax(ref.double(), -1)
    lq = torch.log_softmax(q.double(), -1)
    kl = (lp.exp() * (lp - lq)).sum(-1).numpy()
    top = float((ref.argmax(-1) == q.argmax(-1)).double().mean())
    return float(kl.mean()), float(np.median(kl)), float(np.percentile(kl, 95)), top


def evaluate(m, ev, refs, skip: int):
    """refs: the reference logits of the scored positions alone (from skip)."""
    out = []
    for ids, ref in zip(ev, refs):
        out.append(kl_stats(ref, logits_of(m, ids)[skip:]))
    return out


class _Stop(Exception):
    """Ends a calibration pass at its last layer: the later layers and the head are not needed."""


def set_weight(lin, w):
    import torch

    lin.weight.data = torch.as_tensor(np.asarray(w, np.float32))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=sorted(MODELS))
    ap.add_argument("ckpt", type=Path, help="the Hugging Face checkpoint")
    ap.add_argument("out", type=Path, help="writes OUT/codes (with --method gptq) and OUT/result.json")
    ap.add_argument("--method", choices=("gptq", "rtn", "codes"), default="gptq",
                    help="codes: evaluate the codes in --codes (an earlier run's OUT/codes)")
    ap.add_argument("--codes", type=Path)
    ap.add_argument("--q8", choices=("gptq", "rtn"), default="rtn",
                    help="the q8 projections (attention, convolutions): GPTQ too, or rounded")
    ap.add_argument("--head", choices=("q4", "q6", "q8", "bf16", *LUT_BITS), default="q4")
    ap.add_argument("--head-method", choices=("gptq", "rtn"), default=None, help="the head's (default --method)")
    ap.add_argument("--lut-fit", choices=("uniform", "wkmeans"), default="uniform",
                    help="lutN with GPTQ: the tables palette.Uniform fits (the converter's), or Lloyd's from them "
                         "with each weight counted by its in-feature's E[x^2] (lut_wkmeans)")
    ap.add_argument("--fmt-q4", choices=("q4", "q6", "q8", *LUT_BITS), default="q4",
                    help="the format of the projections the export holds in q4nx (the MLP's)")
    ap.add_argument("--fmt-q8", choices=("q4", "q6", "q8", *LUT_BITS), default="q8",
                    help="the format of those it holds in q8 (attention, convolutions); lutN measures Apple's")
    ap.add_argument("--calib", nargs="*", help="calibration texts (default CALIB)")
    ap.add_argument("--eval", nargs="*", help="evaluation texts (default EVAL)")
    ap.add_argument("--calib-seqs", type=int, default=64)
    ap.add_argument("--calib-len", type=int, default=512)
    ap.add_argument("--eval-len", type=int, default=656)
    ap.add_argument("--eval-skip", type=int, default=256, help="positions of each evaluation text not scored (its prompt)")
    ap.add_argument("--layers-a-pass", type=int, default=4)
    ap.add_argument("--skip-first", type=int, default=1,
                    help="positions at the start of each calibration text left out of H: the BOS's activations "
                         "are outliers (the attention sink) and would make H fit that one position")
    ap.add_argument("--damp", type=float, default=0.1,
                    help="added to H's diagonal, times its mean: 0.01 (GPTQ's) fitted LFM3 to the calibration texts and made the GPL's KL worse than rounding; 0.1 made every text better")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args(argv)

    import torch

    if a.threads:
        torch.set_num_threads(a.threads)
    spec = MODELS[a.model]
    t0 = time.time()
    m, tok = load(a.ckpt, spec)
    bos = spec["bos"]
    global EVAL
    EVAL = [p for p in (a.eval or EVAL) if Path(p).is_absolute() and Path(p).is_file() or (REPO / p).is_file()]
    ev = [texts([p], tok, bos, a.eval_len, 1)[0] for p in EVAL]
    # the scored positions alone (a vocabulary of 262144 makes a text's logits 0.7 GB)
    refs = [logits_of(m, ids)[a.eval_skip :].clone() for ids in ev]
    print(f"loaded, reference logits: {time.time() - t0:.0f} s", flush=True)

    # each projection's format and method: the 4-bit parts' (MODELS' q4, the head), the q8 parts'
    fmts = {"q4": a.fmt_q4, "q8": a.fmt_q8}
    head = m.get_output_embeddings()
    lins_all = [[(name, lin, fmts[role], a.method if role == "q4" else a.q8) for name, lin, role in mine]
                for mine in projections(m, spec)]
    if a.head != "bf16":
        lins_all.append([("lm_head", head, a.head, a.head_method or a.method)])
    codes_dir = a.out / "codes"
    index = {}

    def put(name, lin, fmt, blocks):
        codes_dir.mkdir(parents=True, exist_ok=True)
        k = key(lin.weight.detach().numpy())
        np.save(codes_dir / f"{k}.npy", blocks)
        index[k] = dict(name=name, fmt=fmt, shape=list(lin.weight.shape))

    def put_lut(name, lin, fmt, idx, table):
        """lutN for hwnpu-convert -codes (internal/palette.LoadCodes): the indices (rows, cols)
        uint8 and the tables (rows / LUT_GROUP, 2^bits) float32 (fp16 values), raw."""
        codes_dir.mkdir(parents=True, exist_ok=True)
        k = key(lin.weight.detach().numpy())
        np.ascontiguousarray(idx, np.uint8).tofile(codes_dir / f"{k}.idx")
        np.ascontiguousarray(table, "<f4").tofile(codes_dir / f"{k}.tab")
        index[k] = dict(name=name, fmt=fmt, shape=list(lin.weight.shape), group=LUT_GROUP)

    if a.method == "codes":  # evaluate codes written before (--codes): those, the rest rounded
        n = iron_q4.load_codes(a.codes)
        luts = {k: e for k, e in json.loads((a.codes / "index.json").read_text()).items() if e["fmt"] in LUT_BITS}
        for mine in lins_all:
            for name, lin, fmt, _ in mine:
                w = lin.weight.detach().numpy()
                e = luts.get(key(w)) if fmt in LUT_BITS else None
                if e is not None and e["fmt"] == fmt:
                    rows, cols = w.shape
                    idx = np.fromfile(a.codes / f"{key(w)}.idx", np.uint8).reshape(rows, cols)
                    tab = np.fromfile(a.codes / f"{key(w)}.tab", "<f4").reshape(rows // e["group"], -1)
                    set_weight(lin, np.take_along_axis(np.repeat(tab, e["group"], 0), idx.astype(np.int64), 1))
                    iron_q4.CODES_USED["hit"] += 1
                    continue
                set_weight(lin, iron_q4.dequantize_fmt(iron_q4.quantize_fmt(w, fmt), *w.shape, fmt)
                           if fmt in iron_q4.FORMATS else rtn(w, fmt))
        print(f"{a.codes}: {n} weights, {iron_q4.CODES_USED}")
        lins_all = []
    # Rounded projections first: the later layers' inputs are then those the device computes.
    for mine in lins_all:
        for name, lin, fmt, method in mine:
            if method == "rtn":
                set_weight(lin, rtn(lin.weight.detach().numpy(), fmt))
    gptq_layers = [[x for x in mine if x[3] == "gptq"] for mine in lins_all]
    todo = [i for i, g in enumerate(gptq_layers) if g]
    if todo:
        calib = texts(a.calib or CALIB, tok, bos, a.calib_len, a.calib_seqs)
        print(f"calibration: {len(calib)} x {a.calib_len} tokens", flush=True)
        # the head (the last entry) in a pass of its own: its input is the last layer's output
        body = [i for i in todo if i < len(lins_all) - 1 or a.head == "bf16"]
        groups = [body[i : i + a.layers_a_pass] for i in range(0, len(body), a.layers_a_pass)]
        groups += [[i] for i in todo if i not in body]
        for grp in groups:
            lins = [x for i in grp for x in gptq_layers[i]]
            Hs = {id(lin): [None, 0] for _, lin, _, _ in lins}
            hooks = []
            # x^T x by the input tensor: q, k and v (and gate and up) read one. The tensor is held,
            # so no later one takes its id during the pass.
            same = {}

            def hook(mod, inp, _out):
                x = inp[0]
                if id(x) not in same:
                    xs = x.detach()[0, a.skip_first :]
                    same[id(x)] = (x, xs.T @ xs, xs.shape[0])
                _, xtx, n = same[id(x)]
                h = Hs[id(mod)]
                h[0] = xtx.clone() if h[0] is None else h[0].add_(xtx)
                h[1] += n

            for _, lin, _, _ in lins:
                hooks.append(lin.register_forward_hook(hook))
            if max(grp) < len(lins_all) - 1 or a.head == "bf16":  # not the head's pass

                def stop(*_):
                    raise _Stop

                hooks.append(_get(m, spec["layers"])[max(grp)].register_forward_hook(stop))
            t = time.time()
            for ids in calib:
                same.clear()
                try:
                    m(input_ids=torch.tensor([ids]))
                except _Stop:
                    pass
            same.clear()
            for h in hooks:
                h.remove()
            tc = time.time() - t
            t = time.time()
            for name, lin, fmt, _ in lins:
                H, n = Hs[id(lin)]
                w = lin.weight.detach().numpy().copy()
                if H is None:  # never run (Gemma 4's KV-shared layers' k and v): rounded
                    set_weight(lin, rtn(w, fmt))
                    if fmt in iron_q4.FORMATS:
                        put(name, lin, fmt, iron_q4.quantize_fmt(w, fmt))
                    continue
                c, s, mn = gptq(w, (H.double() / n).numpy(), fmt, damp=a.damp, lut_fit=a.lut_fit)
                if fmt in LUT_BITS:
                    put_lut(name, lin, fmt, c, s)
                    set_weight(lin, mn)
                    continue
                blocks = blocks_of(c, s, mn, fmt)
                put(name, lin, fmt, blocks)
                set_weight(lin, iron_q4.dequantize_fmt(blocks, *w.shape, fmt))
            print(f"layers {grp}: {len(lins)} projections, calibration {tc:.0f} s, GPTQ {time.time() - t:.0f} s",
                  flush=True)
    res = evaluate(m, ev, refs, a.eval_skip)
    if index:
        (codes_dir / "index.json").write_text(json.dumps(index, indent=1))
    out = dict(model=a.model, method=a.method, lut_fit=a.lut_fit, q8=a.q8, head=a.head, head_method=a.head_method, damp=a.damp, fmt_q4=a.fmt_q4, fmt_q8=a.fmt_q8,
               calib_seqs=a.calib_seqs,
               eval={Path(p).name: dict(zip(("kl_mean", "kl_median", "kl_p95", "top1"), r)) for p, r in zip(EVAL, res)})
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "result.json").write_text(json.dumps(out, indent=1))
    for p, (mean, med, p95, top) in zip(EVAL, res):
        print(f"{Path(p).name}: KL {mean:.4f} / {med:.4f} / {p95:.3f}, top-1 {top * 100:.1f} %")
    print(f"{time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
