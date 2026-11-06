"""Sliding-window attention that reads only the keys in its window (``install``).

IRON's ``KVCache`` keeps a sliding window's keys as a ring only when three rings fit under
``max_seq_len`` (Gemma 4's 512 window in a 4096 context does not). Otherwise its ``MHA`` streams
the keys from position 0 up to the query's for every block of queries; mha.cc skips the compute
of the blocks out of the window but still reads and steps over them: a prompt chunk at position
3072 streams 4096 keys per block of queries where 1536 hold its window (design.md 16.84).

Here a windowed layer reads a fixed span of keys, rounded up to whole 64-key blocks:

- A prompt chunk, as IRON's ring form of a chunk (``KVCache._ring``) but over the position-indexed
  cache: the ``chunk + window`` keys ending at ``start + chunk`` are copied into a staging state
  (``Copy`` with a per-call offset, as the cache writes use) and IRON's MHA attends over the copy,
  the queries the chunk's ``chunk`` rows unbounded, the rows past the call's cut from the output.
  Offset and length are expressions of ``start`` alone. The cache has ``chunk`` rows more, so a
  last chunk that starts within a chunk of the context's end still copies inside it.
- A decode step: the ``window + 1`` keys ending at ``position`` read in place, at a per-call DMA
  offset of K and V (``operator``: IRON's MHA with ``kv_offset``).

MHA masks by positions relative to the keys it reads (the queries are their last rows), so the
attention is the same; only where the blocks of keys fall moves. The cache stays position-indexed,
so a prompt can restart at any position (no ring). The MHA subclass keeps IRON's name ``MHA``: the
export writes its derived words (s_q, q_start, ...) for hwnpu by that name
(``iron_export_lm._derived_entry``, Go's ``iron.Derived``), and they read the bounds, not where
the keys start.

The first prompt form read the cache in place at an offset of ``position`` (as the decode step
does). It hung in the full model, as did every form tried, before Go's ``chunkBarrier`` was found
to be what the hang needed (the next chunk's first command); it was not tried again with the
barrier. The copy costs under 1 % of the attention it saves.
"""

from __future__ import annotations

import contextvars
import dataclasses

import numpy as np

SPAN_BLOCK = 64  # mha.cc's key block (a prompt's B_kv; a decode step's divides it)

_offset: contextvars.ContextVar = contextvars.ContextVar("iron_window_offset", default=None)


def _round_up(n: int, k: int = SPAN_BLOCK) -> int:
    return -(-n // k) * k


def spans(chunk: int, window: int) -> tuple[int, int]:
    """The keys a windowed layer reads: a prompt chunk's and a decode step's."""
    return _round_up(chunk + window), _round_up(window + 1)


_STAGES: dict = {}


def _staging(span: int, G: int, D: int):
    """The keys' and values' staging states of a prompt chunk's span, one pair shared by every
    layer of that shape (the layers run one after another)."""
    import iron

    key = (span, G, D)
    if key not in _STAGES:
        _STAGES[key] = iron.state((span, G, D)), iron.state((span, G, D))
    return _STAGES[key]


_MHA = None


def operator():
    """IRON's MHA with ``kv_offset``: the element of K and V the keys start at, per call."""
    global _MHA
    if _MHA is not None:
        return _MHA
    from aie.utils.compile.jit.params import per_call
    from iron.common.declare.operand import Operand
    from iron.operators.mha import MHA as IronMHA

    fill = Operand.fill
    if not getattr(fill, "_iron_window", False):

        def offset_fill(self, tap=None, **kw):
            at = _offset.get()
            if at is not None and at[0] is self.op and self.name in ("K", "V"):
                kw.setdefault("offset_parameter", at[1])
            return fill(self, tap, **kw)

        offset_fill._iron_window = True
        Operand.fill = offset_fill

    class MHA(IronMHA):
        __doc__ = IronMHA.__doc__ + "\n\nK and V are read from ``kv_offset`` (elements) on (iron_window)."

        kv_offset = per_call(derive=lambda op: 0)

        def sequence(self, rt):
            v = self.value("kv_offset")
            token = _offset.set((self, v) if v.per_call else None)
            try:
                return super().sequence(rt)
            finally:
                _offset.reset(token)

        def reference(self, Q, K, V, s_q=None, s_kv=None, kv_offset=0):
            rows = int(kv_offset) // int(np.prod(K.shape[1:]) if self.kv_interleaved else K.shape[-1])
            if rows:
                if self.kv_interleaved:
                    K, V = K[rows:], V[rows:]
                else:
                    K, V = K[:, rows:], V[:, rows:]
                pad = [(0, rows) if a == (0 if self.kv_interleaved else 1) else (0, 0) for a in range(K.ndim)]
                K, V = np.pad(K, pad), np.pad(V, pad)
            return super().reference(Q, K, V, s_q=s_q, s_kv=s_kv)

    MHA.__qualname__ = "MHA"
    _MHA = MHA
    return MHA


def install(decode: bool = True) -> None:
    """Make every windowed ``KVCache`` read only its window's keys (none keeps a ring).

    ``decode`` False leaves a decode step (and a prompt's decode-form rows) to IRON's form."""
    import iron
    from iron.lm import blocks
    from iron.operators.copy import Copy

    KVCache = blocks.KVCache
    if getattr(KVCache, "_iron_window", False):
        return
    init, attend = KVCache.__init__, KVCache.attend

    def __init__(self, c, window=None):
        init(self, c, window)
        if window is None:
            return
        G, D, L = c.num_key_value_heads, c.head_dim, c.max_seq_len
        prompt, decode = spans(self.chunk, window)
        # A span as long as the cache reads it all: IRON's form is the same and simpler.
        self.spans = None if prompt >= L or L % SPAN_BLOCK else (prompt, decode)
        if self.spans is None:
            return
        if self.ring is not None:  # position-indexed instead: the span reads the window
            self.ring = None
            del self.staged
        # A chunk's span ends at start + chunk, past the context for a last chunk that starts
        # within a chunk of its end: the cache has a chunk of rows more, which nothing writes.
        self.keys, self.values = iron.state((L + self.chunk, G, D)), iron.state((L + self.chunk, G, D))
        _staging(prompt, G, D)

    def windowed(self, call, q, k, v, scale=None):
        if (self.window is None or self.ring is not None or getattr(self, "spans", None) is None
                or not (call.prompt or decode)):
            return attend(self, call, q, k, v, scale)
        C, (rows, G, D), (n, H, _) = self.chunk, self.keys.shape, q.shape
        for x, cache in ((k, self.keys), (v, self.values)):
            if x is None:
                continue
            if call.prompt:
                cache[call.start : call.start + C][: call.rows] = x
            else:
                cache[call.position] = x.reshape(G, D)
        if call.prompt:
            # As IRON's ring form of a chunk (KVCache._ring): the span ends at start + chunk, so
            # the chunk's rows, padding included, are the keys' last; the copy's offset and
            # length follow from start alone (a scratchpad value, as the cache writes' do); the
            # queries are not bounded, and the rows past the call's are cut from the output. The
            # keys past the call's last row are stale, but only padding rows reach them.
            span = self.spans[0]
            end = call.start + C
            first = iron.clamp(end - span, lo=0)
            count = iron.clamp(end, hi=span)
            staged = _staging(span, G, D)
            for cache, stage in zip((self.keys, self.values), staged):
                Copy.step(cache[first : first + span][:count], stage[:span])
            q = dataclasses.replace(q, valid={})
            o = blocks.MHA.step(
                q,
                *(stage[:count] for stage in staged),
                heads_interleaved=True,
                kv_interleaved=True,
                window=self.window,
                scale=scale,
            )
            return o[: call.rows].reshape(n, H * D)
        span = self.spans[1]
        end = call.position + 1
        first = iron.clamp(end - span, lo=0)
        count = iron.clamp(end, hi=span)
        keys, values = self.keys[:count], self.values[:count]
        o = operator().step(
            q,
            keys,
            values,
            heads_interleaved=True,
            kv_interleaved=True,
            window=self.window,
            scale=scale,
            kv_offset=first * (G * D),
        )
        return o.reshape(n, H * D)

    KVCache.__init__ = __init__
    KVCache.attend = windowed
    KVCache._iron_window = True
