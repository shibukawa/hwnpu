"""Gemma 4's token tables on the host instead of on the device, and its weights in 4 bits.

IRON's Gemma4 gathers each layer's per-layer input (PLE) from a table of vocab_size rows
of num_hidden_layers * ple_dim: 4.4 GiB of bf16 for E2B, 5.25 GiB for E4B, almost half of
the arena. Every byte of the arena is pinned host memory, yet a call reads one row a token.

With install(), the table stays a file (ple.bin: the rows, each layer's columns scaled by
sqrt(ple_dim), the placeholders' rows the pad token's, as Gemma4.load_state_dict makes
them) and the device holds ple_rows, (num_hidden_layers, prefill_chunk, ple_dim): before
each call the host writes row r of every layer from the table row of the call's token r (the
decode step's token at row 0). The layers read their rows from it in order, so no gather
is left; what the layers compute is the same, bit for bit.

With install(q4=True), the MLP's projections of every layer (gate, up, down) and the head are
q4nx, and the attention's (q, k, v, o) are in ``attn``: q8 (the default), q4 or bf16. They are
0.53 GB of E2B's bf16 but most of the error (KL from the bf16 logits on 400 positions of text:
0.229 nats in q4nx, 0.091 in bf16; the head barely matters: 0.2245). The formats are
iron_q4's: q4nx 5 bits a weight, q8 8.5; decode on QGEMV, prefill dequantized on the device. The head is tied to the token embeddings, whose rows the head no
longer gives in bf16: they go to the host the same way (embed.bin, embed_rows of
(prefill_chunk, hidden_size)), times sqrt(hidden_size) on the device as before.

With install(ple="int8") (iron_export_lm's default), ple.bin holds each layer's slice of a row
(ple_dim values) as an fp16 scale and ple_dim int8 values (ple_int8), and the manifest's entry
says format "int8": the host expands a row to bf16 as it writes it. Half the file (E2B 4.4 ->
2.2 GiB, E4B 5.25 -> 2.65), KL 0.00005 from the bf16 table (tools/amd/ple_quant_kl.py,
docs/improvements.md 12.7). The device's buffer and what the layers compute are unchanged.

The manifest names the tables in host_tables (iron_export_lm.export). hwnpu's iron.Model fills
them.
"""

import functools
import os

import numpy as np
from ml_dtypes import bfloat16

PLE = "hwnpu_ple_rows"
# A decode step's per-layer inputs, every layer's at once before the first layer (Gemma4's
# layer computes each layer's: a GEMV, an RMSNorm, a copy, an add and a product, each a
# reconfiguration of the array, design.md 16.262). HWNPU_GEMMA4_BATCH_PLE=0 keeps a layer's own.
BATCH_PLE = os.environ.get("HWNPU_GEMMA4_BATCH_PLE", "1") != "0"
EMBED = "hwnpu_embed_rows"
# A prompt chunk's commands end where their scratchpad words would pass IRON's limit (32).
# Each of IRON's gathers took words of its own, so the chunk was 15 commands for E2B; without
# them it is 7, one of them 3.7 s at position 3072, near amdxdna's cutoff (tdr_timeout_ms
# 2000, about 4 s without progress), and their control code no longer fits the 64 MiB device
# heap beside the decode step's. A lower limit splits them again (a version with no input,
# the decode step, stays one command). install's words sets another (iron_gemma4.Gemma4Tail's).
SCRATCHPAD_WORDS = 24

# The tables by id of their model, by buffer: IRON takes no array attribute that is not a weight.
_TABLES: dict[int, dict[str, np.ndarray]] = {}
_Q4 = {"on": False, "attn": "q8"}
_PLE = {"fmt": "bf16"}  # how write() stores ple.bin: install's ple
_HOST_EMBED: dict[int, bool] = {}  # by id of the model: its embeddings on the host


def _keep() -> set:
    """What stays bf16 with q4: attention where install's attn is "bf16". HWNPU_Q4_KEEP (comma
    separated: head, attn, mlp) names it instead, to measure what each costs."""
    env = os.environ.get("HWNPU_Q4_KEEP")
    if env is not None:
        return set(filter(None, env.split(",")))
    return {"attn"} if _Q4["attn"] == "bf16" else set()


def install(q4: bool = False, attn: str = "q8", words: int = SCRATCHPAD_WORDS, ple: str = "bf16") -> None:
    """Patch IRON's Gemma4 (and so Gemma4Multimodal's text model) to read its PLE from ple_rows;
    with q4, its projections and head in q4nx and its token embeddings from embed_rows. A prompt
    chunk's command reads up to words scratchpad words. ple is how ple.bin stores the table:
    "int8" (ple_int8) or "bf16"."""
    if ple not in ("int8", "bf16"):
        raise ValueError(f"ple.bin in {ple}: int8 or bf16")
    _PLE["fmt"] = ple
    import iron
    from iron.applications.gemma4 import model as g4
    from iron.lm import CausalLM
    from iron.operators.copy import Copy

    G = g4.Gemma4
    if q4:
        import iron_q4

        iron_q4.install_linear()
        _Q4["on"] = True
        if attn not in ("bf16", "q8", "q4"):
            raise ValueError(f"attention in {attn}: bf16, q8 or q4")
        _Q4["attn"] = attn
    if getattr(G, "hwnpu_host_ple", False):
        return
    base_init = G.__init__

    def __init__(self, config, weights):
        base_init(self, config, weights)
        c = self.config
        self.ple_rows = iron.state((c.num_hidden_layers, c.prefill_chunk, c.ple_dim), bfloat16, name=PLE)
        if _Q4["on"] and "head" not in _keep():
            self.embed_rows = iron.state((c.prefill_chunk, c.hidden_size), bfloat16, name=EMBED)

    def load_state_dict(self, tree, *, path: str = "") -> None:
        CausalLM.load_state_dict(self, tree, path=path)
        c, m = self.config, self.model
        assert m.embed_tokens_per_layer is not None, "loaded once"
        scale = np.float32(bfloat16(np.sqrt(c.ple_dim)))
        table = np.asarray(m.embed_tokens_per_layer.weight)
        # Gemma4.load_state_dict's rows, all layers side by side: a token's row is contiguous
        rows = (table.astype(np.float32) * scale).astype(bfloat16)
        placeholders = [t for t in (c.image_token_id, c.audio_token_id, c.video_token_id) if t is not None]
        rows[placeholders] = rows[c.pad_token_id]
        tables = _TABLES.setdefault(id(self), {})
        tables[PLE] = rows
        m.embed_tokens_per_layer = None
        if _Q4["on"]:
            quantize(self, tables)

    def quantize(self, tables) -> None:
        import os

        import iron_q4

        keep = _keep()
        m = self.model
        attn = None if "attn" in keep else _Q4["attn"]
        for layer in m.layers:
            a, mlp = layer.self_attn, layer.mlp
            lins = []
            if attn is not None:
                lins += [(lin, attn) for lin in (a.q_proj, a.k_proj, a.v_proj, a.o_proj)]
            if "mlp" not in keep:
                lins += [(lin, "q4") for lin in (mlp.gate_proj, mlp.up_proj, mlp.down_proj)]
            for lin, fmt in lins:
                if lin is not None:
                    lin.weight = iron_q4.q_param(np.asarray(lin.weight, np.float32), fmt)
        if "head" in keep:
            return
        # the head is tied to the embeddings: its own copy in q4nx, the embeddings to the host
        embed = np.asarray(m.embed_tokens.weight)
        tables[EMBED] = embed.astype(bfloat16)
        self.lm_head.weight = iron_q4.q4_param(embed.astype(np.float32))
        m.embed_tokens.weight = iron.param(np.zeros((1, embed.shape[1]), bfloat16))  # no longer read
        _HOST_EMBED[id(self)] = True

    def lookup(self, ids):
        # The embeddings from embed_rows, which the host filled with the call's tokens in order,
        # times sqrt(hidden_size) in the weights' dtype as Gemma4.lookup.
        if not _HOST_EMBED.get(id(self)):
            return base_lookup(self, ids)
        s = float(bfloat16(np.sqrt(self.config.hidden_size)))
        shape = getattr(ids, "shape", ())
        if len(shape) == 0:
            rows = self.embed_rows[:1]
        else:
            bound = getattr(ids, "valid", None) or {}
            rows = self.embed_rows[: bound[0]] if 0 in bound else self.embed_rows[: shape[0]]
        return Copy.step(rows) * s

    def layer(self, call, i, w, x):
        c, m, P = self.config, self.model, self.config.ple_dim
        if BATCH_PLE and not call.prompt and getattr(call, "row", None) is None:
            # a decode step: every layer's per-layer input at once, before the first layer
            # (five operators in place of five a layer: the same values, design.md 16.265)
            batch = getattr(self, "_ple_batch", None)
            if i == 0 or batch is None or batch[0] is not call:
                L = c.num_hidden_layers
                ple = g4.F.linear(call.embeddings, m.per_layer_model_projection.weight, precision=c.precision)
                ple = g4.F.rms_norm(ple.reshape(L, P), (P,), m.per_layer_projection_norm.weight,
                                    c.eps * c.hidden_size)
                token = Copy.step(self.ple_rows[:, :1]).reshape(L, P)
                self._ple_batch = batch = (call, (ple + token) * 2**-0.5)
            ple_i = batch[1][i].reshape(1, P)
            if fused_norms(self, x):
                L = len(m.layers)
                xn = w.input_layernorm(x) if i == 0 else self._normed
                nxt = m.layers[i + 1].input_layernorm if i + 1 < L else m.norm
                x, self._normed = fused_layer(w, x, xn, call, ple_i, nxt)
                return x
            return w(x, call, ple_i)
        projection = m.per_layer_model_projection.weight[i * P : (i + 1) * P]
        ple = g4.F.linear(call.embeddings, projection, precision=c.precision)
        norm = m.per_layer_projection_norm.weight
        ple = g4.F.rms_norm(ple, (P,), norm, c.eps * c.hidden_size)
        if call.prompt:
            token = Copy.step(self.ple_rows[i][: call.rows])
        elif getattr(call, "row", None) is not None:
            # a prompt chunk's last row run alone (iron_gemma4.Gemma4Tail): the host wrote it at
            # its row of the chunk, read as the prompt's layers read theirs, then that row
            rows = Copy.step(self.ple_rows[i][: call.rows])
            token = Copy.step(rows[call.row])
        else:
            token = Copy.step(self.ple_rows[i][:1])
        return w(x, call, (ple + token.reshape(ple.shape)) * 2**-0.5)

    def head(self, x):
        xn, self._normed = getattr(self, "_normed", None), None
        if xn is None:
            return base_head(self, x)
        # the last layer's SandwichAdd normed it with the final norm: Gemma4.head's softcap
        logits, cap = self.lm_head(xn), self.config.final_logit_softcapping
        return logits if cap is None else g4.Tanh.step(logits * (1 / cap)) * cap

    base_body = G.body

    @functools.wraps(base_body)
    def body(self, ids=None, **values):
        if ids is not None and _HOST_EMBED.get(id(self)):
            # a version's input must be read, and no table is left to gather from: a copy of
            # the whole ids (4 KiB, unbounded: a bounded copy of 4-byte rows fits no
            # descriptor), which nothing reads
            Copy.step(ids, dtype=np.int32)
        return base_body(self, ids, **values)

    from iron.common.image import packaging

    packaging.SCRATCHPAD_WORDS = min(packaging.SCRATCHPAD_WORDS, words)
    base_lookup = G.lookup
    base_head = G.head
    G.head = head
    G.__init__ = __init__
    G.load_state_dict = load_state_dict
    G.lookup = lookup
    G.body = body
    G.layer = layer
    G.hwnpu_host_ple = True


# A decode step's post-norms, residuals and next norms as SandwichAdd (iron_sandwich): after
# the attention (post-attention norm, residual, pre-feed-forward norm) and after the per-layer
# input (its norm, residual, layer_scalar, the next layer's input norm), in place of seven
# operators a layer (design.md 16.265). HWNPU_GEMMA4_FUSED_NORMS=0 keeps IRON's.
FUSED_NORMS = os.environ.get("HWNPU_GEMMA4_FUSED_NORMS", "1") != "0"


def fused_norms(model, x) -> bool:
    c, m = model.config, model.model
    if not FUSED_NORMS or x.shape[0] != 1:
        return False
    norms = [m.norm] + [n for l in m.layers for n in (
        l.input_layernorm, l.post_attention_layernorm, l.pre_feedforward_layernorm,
        l.post_feedforward_layernorm, l.post_per_layer_input_norm)]
    return all(n.eps == c.eps for n in norms) and all(l.residual_multiplier == 1 for l in m.layers)


def fused_layer(w, x, xn, call, ple, nxt):
    """Gemma 4's layer (PleLayer's body) on x and its input already normed xn, with two
    SandwichAdds: the residual and the next layer's input (normed by nxt)."""
    import iron_sandwich

    eps = w.post_attention_layernorm.eps
    h = w.mix(xn, call)
    x, n = iron_sandwich.sandwich_add(h, x, w.post_attention_layernorm.weight, w.pre_feedforward_layernorm.weight, eps)
    h = w.mlp(n)
    x = x + w.post_feedforward_layernorm(h).reshape(x.shape)
    g = w.mlp.act(w.per_layer_input_gate(x)) * ple
    g = w.per_layer_projection(g)
    return iron_sandwich.sandwich_add(g, x, w.post_per_layer_input_norm.weight, nxt.weight, eps, scale=w.scale)


def text_model(model):
    """The causal text model of model (a multimodal model's text)."""
    return model.text if hasattr(model, "towers") else model


def ple_int8(rows: np.ndarray, layers: int, path, chunk: int = 8192) -> None:
    """Write the table rows (vocab, layers * ple_dim, bf16) to path as ple.bin's format "int8": a
    row is layers parts, each an fp16 scale (the slice's largest magnitude / 127) followed by its
    ple_dim values rounded to int8. hwnpu expands a value to bf16(q * scale)."""
    n, width = rows.shape
    e = width // layers
    with open(path, "wb") as f:
        for i in range(0, n, chunk):
            x = np.asarray(rows[i : i + chunk]).astype(np.float32).reshape(-1, layers, e)
            s = (np.abs(x).max(-1) / 127).astype(np.float16)
            d = s.astype(np.float32)
            d[d == 0] = 1
            q = np.clip(np.rint(x / d[..., None]), -127, 127).astype(np.int8)
            part = np.empty((x.shape[0], layers, 2 + e), np.uint8)
            part[..., :2] = s.view(np.uint8).reshape(x.shape[0], layers, 2)
            part[..., 2:] = q.view(np.uint8)
            part.tofile(f)


def write(model, buffers: dict, out) -> list:
    """Write the host tables (ple.bin, embed.bin) and return the manifest's host_tables (empty
    without install())."""
    text = text_model(model)
    tables = _TABLES.get(id(text))
    if not tables:
        return []
    c = text.config
    out_tables = []
    for state, rows in tables.items():
        names = [n for n in buffers if n == state or n.endswith("." + state)]
        if len(names) != 1:
            raise SystemExit(f"host tables: the decode step holds {names} for {state}")
        fn = "ple.bin" if state == PLE else "embed.bin"
        layers = int(c.num_hidden_layers) if state == PLE else 1
        entry = dict(
            buffer=names[0], file=fn, rows=int(rows.shape[0]),
            # A table row is layers parts of part_bytes; the buffer holds, by layer, slots rows of
            # part_bytes: part l of the call's row r goes to slot r of layer l.
            layers=layers, part_bytes=int(rows.shape[1]) * 2 // layers, slots=int(c.prefill_chunk),
        )
        if state == PLE and _PLE["fmt"] == "int8":
            ple_int8(rows, layers, out / fn)
            entry["format"] = "int8"
        else:
            np.ascontiguousarray(rows).view(np.uint8).tofile(out / fn)
        out_tables.append(entry)
    return out_tables
