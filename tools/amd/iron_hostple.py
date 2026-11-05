"""Gemma 4's per-layer token table on the host instead of on the device.

IRON's Gemma4 gathers each layer's per-layer input (PLE) from a table of vocab_size rows
of num_hidden_layers * ple_dim: 4.4 GiB of bf16 for E2B, 5.25 GiB for E4B, almost half of
the arena. Every byte of the arena is pinned host memory, yet a call reads one row a token.

With install(), the table stays a file (ple.bin: the rows, each layer's columns scaled by
sqrt(ple_dim), the placeholders' rows the pad token's, as Gemma4.load_state_dict makes
them) and the device holds ple_rows, (num_hidden_layers, prefill_chunk, ple_dim): before
each call the host writes row r of every layer from the table row of the call's token r (the
decode step's token at row 0). The layers read their rows from it in order, so no gather
is left; what the layers compute is the same, bit for bit.

The manifest names it in host_tables (iron_export_lm.export). hwnpu's iron.Model fills it.
"""

import numpy as np
from ml_dtypes import bfloat16

STATE = "hwnpu_ple_rows"
# A prompt chunk's commands end where their scratchpad words would pass IRON's limit (32).
# Each of IRON's gathers took words of its own, so the chunk was 15 commands for E2B; without
# them it is 7, one of them 3.7 s at position 3072, near amdxdna's cutoff (tdr_timeout_ms
# 2000, about 4 s without progress), and their control code no longer fits the 64 MiB device
# heap beside the decode step's. A lower limit splits them again (a version with no input,
# the decode step, stays one command).
SCRATCHPAD_WORDS = 24

# The tables by id of their model: IRON takes no array attribute that is not a weight.
_TABLES: dict[int, np.ndarray] = {}


def install() -> None:
    """Patch IRON's Gemma4 (and so Gemma4Multimodal's text model) to read its PLE from ple_rows."""
    import iron
    from iron.applications.gemma4 import model as g4
    from iron.lm import CausalLM
    from iron.operators.copy import Copy

    G = g4.Gemma4
    if getattr(G, "hwnpu_host_ple", False):
        return
    base_init = G.__init__

    def __init__(self, config, weights):
        base_init(self, config, weights)
        c = self.config
        self.ple_rows = iron.state((c.num_hidden_layers, c.prefill_chunk, c.ple_dim), bfloat16, name=STATE)

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
        _TABLES[id(self)] = rows
        m.embed_tokens_per_layer = None

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

    from iron.common.image import packaging

    packaging.SCRATCHPAD_WORDS = min(packaging.SCRATCHPAD_WORDS, SCRATCHPAD_WORDS)
    G.__init__ = __init__
    G.load_state_dict = load_state_dict
    G.layer = layer
    G.hwnpu_host_ple = True


def text_model(model):
    """The causal text model of model (a multimodal model's text)."""
    return model.text if hasattr(model, "towers") else model


def write(model, buffers: dict, out) -> list:
    """Write ple.bin and return the manifest's host_tables (empty without install())."""
    text = text_model(model)
    rows = _TABLES.get(id(text))
    if rows is None:
        return []
    names = [n for n in buffers if n == STATE or n.endswith("." + STATE)]
    if len(names) != 1:
        raise SystemExit(f"host PLE: the decode step holds {names} for {STATE}")
    c = text.config
    np.ascontiguousarray(rows).view(np.uint8).tofile(out / "ple.bin")
    return [dict(
        buffer=names[0], file="ple.bin", rows=int(rows.shape[0]),
        # A table row is layers parts of part_bytes; the buffer holds, by layer, slots rows of
        # part_bytes: part l of the call's row r goes to slot r of layer l.
        layers=int(c.num_hidden_layers), part_bytes=int(c.ple_dim) * 2, slots=int(c.prefill_chunk),
    )]
