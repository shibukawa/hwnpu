"""The version of a language model that checks drafts (speculative decoding) on IRON.

A call of ``ROWS`` tokens from ``start``: the next token and the drafts after it. It runs as a
prompt chunk does (every row writes its key and value, the recurrent states go from the slot
before ``start`` to the one after the last row), but returns the logits of every row, and its
linears take the rows through the decode step's GEMV at once (iron_q4's QGEMVRows) instead of
the prompt chunk's GEMM, whose cost is the chunk's whatever the rows (a 2-token call costs 4
decode steps on Qwen3.5-0.8B, 9 on PLaMo 3 0.8B).

``install(Model, rows)`` adds the version to a model class (``iron_export_lm.py --verify``):
``shapes`` gains token ids of ``rows``, and ``forward`` keeps every row for the head on those.
The token it draws (from the last row) is carried as a prompt chunk's; hwnpu reads the logits.
"""

import numpy as np
from ml_dtypes import bfloat16

import iron
from iron.operators.copy import Copy


# The rows of the version checking drafts, set by install before a model is built (its modules
# make the buffers the version needs).
ROWS = {"rows": 0}
# Rows up to which a call's rows run through the few-rows forms (iron_q4.ROWS_GEMV's).
MAX_ROWS = 8


def is_rows(x) -> bool:
    """Whether ``x`` is a few rows, all of them run (the version checking drafts'): neither one
    row (a decode step's) nor bounded per call (a prompt chunk's)."""
    return len(x.shape) == 2 and 1 < x.shape[0] <= MAX_ROWS and x.valid.get(0) is None


def is_verify(model, call) -> bool:
    """Whether ``call`` is the version checking drafts: a prompt call of fewer ids than the chunk."""
    rows = getattr(model, "verify_rows", 0)
    shape = getattr(call.ids, "shape", ())
    return bool(rows) and call.prompt and len(shape) == 1 and shape[0] == rows


def causal_conv1d_rows(x, taps, state):
    """iron.functional.causal_conv1d for ``x`` of a few rows, all of them run (the version
    checking drafts): the rows before the call and the call's in the staging window, read as one
    run, without the chunk's whole window."""
    read, write, window = state.read, state.write, state.window
    K, C = taps.shape[0], window.shape[1]
    R = x.shape[0]
    flat = window.reshape(2 * C, x.shape[1])
    first = C - (K - 1)
    flat[first:C] = read
    flat[C : C + R] = x
    y = flat[first : first + R] * taps[0]
    for k in range(1, K):
        y = y + flat[first + k : first + k + R] * taps[k]
    Copy.step(flat[first + R : first + R + K - 1], write)
    return y


def install_functional() -> None:
    """Route iron.functional's calls that read a prompt chunk's per-call rows to their few-rows
    forms when the rows are not bounded (iron.lm's modules call them through F)."""
    import iron.functional as F

    if getattr(F.causal_conv1d, "hwnpu_rows", False):
        return
    base = F.causal_conv1d

    def causal_conv1d(x, taps, state):
        if is_rows(x):
            return causal_conv1d_rows(x, taps, state)
        return base(x, taps, state)

    causal_conv1d.hwnpu_rows = True
    F.causal_conv1d = causal_conv1d


def rope_rows(rope, call, rows: int) -> tuple:
    """Rope.rows for the version checking drafts: the table's ``rows`` rows from the call's start."""

    def read(table):
        return Copy.step(table[call.start : call.start + rows])

    k = read(rope.table)
    return (k if rope.q_table is None else read(rope.q_table)), k


def install_cache() -> None:
    """KVCache.attend for a few rows, all of them run: each row's key and value written and its
    query attended as the decode step's, a row at a time (its MHA takes one query over the
    cache up to it); the outputs gathered in ``rows_o``, made by KVCache's constructor."""
    from iron.lm.blocks import KVCache
    from iron.operators.mha import MHA

    if getattr(KVCache.attend, "hwnpu_rows", False):
        return
    base_init, base_attend = KVCache.__init__, KVCache.attend

    def __init__(self, c, window=None):
        base_init(self, c, window)
        if ROWS["rows"]:
            self.rows_o = iron.state((ROWS["rows"], c.num_attention_heads * c.head_dim), bfloat16)

    def attend(self, call, q, k, v, scale=None):
        n = q.shape[0]
        if not (call.prompt and 1 < n <= MAX_ROWS and q.valid.get(0) is None):
            return base_attend(self, call, q, k, v, scale)
        if self.ring is not None:
            raise NotImplementedError("KVCache: a ring's rows checking drafts")
        _, G, D = self.keys.shape
        H = q.shape[1]
        for j in range(n):
            at = call.start + j
            for x, cache in ((k, self.keys), (v, self.values)):
                if x is not None:
                    cache[at] = x[j].reshape(G, D)
            o = MHA.step(Copy.step(q[j : j + 1]), self.keys[: at + 1], self.values[: at + 1],
                         heads_interleaved=True, kv_interleaved=True, window=self.window, scale=scale)
            self.rows_o[j : j + 1] = o.reshape(1, H * D)
        return self.rows_o

    __init__.hwnpu_rows = attend.hwnpu_rows = True
    KVCache.__init__, KVCache.attend = __init__, attend


def install(Model, rows: int) -> None:
    """Give ``Model`` (an iron.lm.CausalLM) the version of ``rows`` ids checking drafts."""
    if getattr(Model, "verify_rows", 0):
        return
    install_functional()
    Model.verify_rows = rows
    ROWS["rows"] = rows
    install_cache()
    base_shapes, base_forward = Model.shapes, Model.forward

    def shapes(self):
        return base_shapes(self) + [dict(ids=((rows,), np.int32))]

    def forward(self, x, call):
        if not is_verify(self, call):
            return base_forward(self, x, call)
        c = self.config
        call.embeddings = x
        for rope in (layer.rope for layer in self.layers):
            if rope is not None and rope not in call.ropes:
                call.ropes[rope] = rope_rows(rope, call, rows)
        x = self.positions(x, call)
        for i, weights in enumerate(self.layers):
            x = self.layer(call, i, weights, x)
        logits = self.head(x)  # (rows, vocabulary): every row's
        last = Copy.step(logits[rows - 1]).reshape(1, c.vocab_size + c.vocab_size % 2)
        _, carried = self.draw(last, call)
        return logits, carried

    base_lookup = Model.lookup

    def lookup(self, ids):
        # the host's rows (iron_hostple's embed_rows, a chunk's) bounded to the call's ids, not
        # to the per-call rows: every row of the call runs
        shape = getattr(ids, "shape", ())
        if getattr(self, "host_embed", False) and len(shape) == 1 and shape[0] == rows:
            return Copy.step(self.embed_rows[:rows])
        return base_lookup(self, ids)

    Model.shapes, Model.forward, Model.lookup = shapes, forward, lookup
