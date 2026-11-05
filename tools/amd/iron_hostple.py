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

The manifest names the tables in host_tables (iron_export_lm.export). hwnpu's iron.Model fills
them.
"""

import functools
import os

import numpy as np
from ml_dtypes import bfloat16

PLE = "hwnpu_ple_rows"
EMBED = "hwnpu_embed_rows"
# A prompt chunk's commands end where their scratchpad words would pass IRON's limit (32).
# Each of IRON's gathers took words of its own, so the chunk was 15 commands for E2B; without
# them it is 7, one of them 3.7 s at position 3072, near amdxdna's cutoff (tdr_timeout_ms
# 2000, about 4 s without progress), and their control code no longer fits the 64 MiB device
# heap beside the decode step's. A lower limit splits them again (a version with no input,
# the decode step, stays one command).
SCRATCHPAD_WORDS = 24

# The tables by id of their model, by buffer: IRON takes no array attribute that is not a weight.
_TABLES: dict[int, dict[str, np.ndarray]] = {}
_Q4 = {"on": False, "attn": "q8"}
_HOST_EMBED: dict[int, bool] = {}  # by id of the model: its embeddings on the host


def _keep() -> set:
    """What stays bf16 with q4: attention where install's attn is "bf16". HWNPU_Q4_KEEP (comma
    separated: head, attn, mlp) names it instead, to measure what each costs."""
    env = os.environ.get("HWNPU_Q4_KEEP")
    if env is not None:
        return set(filter(None, env.split(",")))
    return {"attn"} if _Q4["attn"] == "bf16" else set()


def install(q4: bool = False, attn: str = "q8") -> None:
    """Patch IRON's Gemma4 (and so Gemma4Multimodal's text model) to read its PLE from ple_rows;
    with q4, its projections and head in q4nx and its token embeddings from embed_rows."""
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
        projection = m.per_layer_model_projection.weight[i * P : (i + 1) * P]
        ple = g4.F.linear(call.embeddings, projection, precision=c.precision)
        norm = m.per_layer_projection_norm.weight
        ple = g4.F.rms_norm(ple, (P,), norm, c.eps * c.hidden_size)
        if call.prompt:
            token = Copy.step(self.ple_rows[i][: call.rows])
        else:
            token = Copy.step(self.ple_rows[i][:1])
        return w(x, call, (ple + token.reshape(ple.shape)) * 2**-0.5)

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

    packaging.SCRATCHPAD_WORDS = min(packaging.SCRATCHPAD_WORDS, SCRATCHPAD_WORDS)
    base_lookup = G.lookup
    G.__init__ = __init__
    G.load_state_dict = load_state_dict
    G.lookup = lookup
    G.body = body
    G.layer = layer
    G.hwnpu_host_ple = True


def text_model(model):
    """The causal text model of model (a multimodal model's text)."""
    return model.text if hasattr(model, "towers") else model


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
        np.ascontiguousarray(rows).view(np.uint8).tofile(out / fn)
        layers = int(c.num_hidden_layers) if state == PLE else 1
        out_tables.append(dict(
            buffer=names[0], file=fn, rows=int(rows.shape[0]),
            # A table row is layers parts of part_bytes; the buffer holds, by layer, slots rows of
            # part_bytes: part l of the call's row r goes to slot r of layer l.
            layers=layers, part_bytes=int(rows.shape[1]) * 2 // layers, slots=int(c.prefill_chunk),
        ))
    return out_tables
