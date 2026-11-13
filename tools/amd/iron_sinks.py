"""IRON's MHA with a learned sink logit per head (gpt-oss's attention), on the device.

gpt-oss adds to each head's softmax a logit of its own (``self_attn.sinks``), which no key or
value carries: o = sum_j e^{s_j} v_j / (sum_j e^{s_j} + e^{sink}). IRON's MHA keeps each query
row's online softmax (running maximum m and sum l); kernels/mha_sinks.cc adds the sink's term
2^(sink log2 e - m) to l after the last key block, before the division by it. (Seeding m with
the sink instead pushed the scores' exponents far below zero, where the device's exp2 is up to
twice too high.) No compute or transfer is added, and the keys and the cache
are as without sinks.

``sink_class(base)`` subclasses an MHA (IRON's, or iron_window's with ``kv_offset``) with a
``sinks`` parameter (each head's sink, a tuple of floats; None is the base's). Its build is the
base's with the P*V workers' program replaced by the same program that also calls mha_sinks
before the last block's rescale (``_pv``, read from the base's closure). ``SINKS`` is
the step's default: a layer sets it around its cache's attend (``using``), so IRON's
KVCache and iron_window's forms need no change. The class keeps IRON's name ``MHA`` (the
export's derived words go by it).

    python tools/amd/iron_sinks.py check OUT [--decode]   # one MHA with sinks, against float32
"""

import argparse
import contextlib
import contextvars
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).resolve().parent))

SINKS: contextvars.ContextVar = contextvars.ContextVar("iron_sinks", default=None)
LOG2E = 1.4426950408889634


@contextlib.contextmanager
def using(sinks):
    """Steps of a sink class recorded inside take ``sinks`` (a tuple) by default."""
    token = SINKS.set(None if sinks is None else tuple(float(s) for s in sinks))
    try:
        yield
    finally:
        SINKS.reset(token)


def reference(op, Q, K, V, sinks, s_q=None, s_kv=None):
    """IRON's MHA reference (operators/mha.py) with each head's sink in the softmax."""
    if op.heads_interleaved:
        Q = np.swapaxes(Q, 0, 1)
    if op.kv_interleaved:
        K, V = (np.swapaxes(t, 0, 1) for t in (K, V))
    groups = op.num_heads // op.num_KV_heads
    offset = op.kv_tokens - op.q_tokens
    start = offset if op.causal or op.window else 0
    s_q = start + op.q_tokens if s_q is None else int(s_q)
    s_kv = op.kv_tokens if s_kv is None else int(s_kv)
    K, V = K[:, :s_kv], V[:, :s_kv]
    rows = max(s_q - start, 0)
    q = Q[:, :rows].astype(np.float64)
    k, v = (t.astype(np.float64) for t in (K, V))
    key = np.arange(k.shape[1])
    out = np.zeros(Q.shape, dtype=Q.dtype)
    position = offset + np.arange(rows)[:, None]
    masked = np.zeros((rows, k.shape[1]), dtype=bool)
    if op.causal or op.window:
        masked |= key > position + (0 if op.causal else (op.window or 0))
    if op.window:
        masked |= key < position - op.window
    for h in range(q.shape[0]):
        s = q[h] @ k[h // groups].T * op.scale
        s = np.where(masked, -np.inf, s)
        m = np.maximum(s.max(-1, keepdims=True), sinks[h])
        e = np.exp(s - m)
        out[h, :rows] = (e @ v[h // groups]) / (e.sum(-1, keepdims=True) + np.exp(sinks[h] - m))
    if op.heads_interleaved:
        return np.ascontiguousarray(np.swapaxes(out, 0, 1))
    return out


_CLASSES: dict = {}


def sink_class(base):
    """``base`` (an MHA class) with the ``sinks`` parameter (module docstring)."""
    if base in _CLASSES:
        return _CLASSES[base]
    import iron.operators.mha as mha_module
    from aie.dialects.aie import T
    import aie.dialects.index as index
    from aie.iron import Buffer
    from aie.iron.controlflow import range_
    from aie.iron.kernel import ExternalFunction
    from aie.utils.compile.jit.params import param

    class MHA(base):
        sinks: tuple | None = param(default=None, array=True)
        sink_group: int = param(default=lambda op: op.num_heads // op.num_KV_heads, array=True)
        # The form of the build (IRON's compile cache keys on parameters, not on the build's code):
        # 3 adds the sinks at the division (2 wrote a bf16 state; 1 seeded the softmax state).
        sink_form: int = param(default=3, array=True)

        @classmethod
        def step(cls, *operands, **kwargs):
            kwargs.setdefault("sinks", SINKS.get())
            return super().step(*operands, **kwargs)

        def reference(self, Q, K, V, s_q=None, s_kv=None, **kw):
            if self.sinks is None:
                return super().reference(Q, K, V, s_q=s_q, s_kv=s_kv, **kw)
            if kw.get("kv_offset"):  # iron_window's: the keys from that element on
                rows = int(kw["kv_offset"]) // int(np.prod(K.shape[1:]))
                K = np.pad(K[rows:], [(0, rows), (0, 0), (0, 0)])
                V = np.pad(V[rows:], [(0, rows), (0, 0), (0, 0)])
            return reference(self, Q, K, V, np.asarray(self.sinks, np.float64), s_q, s_kv)

        def build(self):
            if self.sinks is None:
                return super().build()
            made = mha_module.Worker
            op = self

            def worker(fn, fn_args=(), **kw):
                import os
                if os.environ.get("SINKS_DEBUG"):
                    print("sinks worker", getattr(fn, "__name__", fn), flush=True)
                if getattr(fn, "__name__", "") == "batched_matmul_pv":
                    fn, fn_args = op._pv(fn, list(fn_args))
                return made(fn, fn_args=fn_args, **kw)

            mha_module.Worker = worker
            try:
                return super().build()
            finally:
                mha_module.Worker = made

        def _pv(self, fn, args):
            """The base's P*V program, mha_sinks before the last block's rescale_O (its
            closure's values as the base's build bound them)."""
            free = dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in fn.__closure__)))
            B_q, num_pipelines, blocks = free["B_q"], free["num_pipelines"], free["blocks"]
            inv_scale = float(bfloat16(self.scale * np.log2(np.e)))  # the base build's (mha.py)
            H = len(self.sinks)
            group = self.sink_group
            steps = H // group // self.num_pipelines if self.packed else 1
            from aie.utils import config

            here = Path(__file__).resolve().parent / "kernels"
            kdir = Path(config.aie_kernels_dir())
            t_ty = np.ndarray[(H,), np.dtype[np.float32]]
            s_ty = np.ndarray[(4 * B_q,), np.dtype[np.float32]]
            kernel = ExternalFunction(
                "mha_sinks", source_file=str(here / "mha_sinks.cc"),
                include_dirs=[str(kdir), config.cxx_header_path()],
                compile_flags=[f"-DSINK_PACKED={int(bool(self.packed))}", f"-DSINK_GROUP={group}",
                               f"-DSINK_STEPS={steps}", f"-DSINK_SCALE={inv_scale!r}f"],
                object_file_name=f"mha_sinks_{int(bool(self.packed))}_{group}_{steps}_{inv_scale:.6g}.o",
                arg_types=[s_ty, t_ty, np.int32, np.int32, np.int32])
            table = Buffer(t_ty, initial_value=(np.asarray(self.sinks, np.float64) * LOG2E).astype(np.float32))
            op = self

            def batched_matmul_pv(of_p, of_v, of_scale, of_o_out, matmul_PV, rescale_O, q_block_bias, rtp,
                                  idx_buffer, acc, table, sinks_k):
                head = [None]

                def q_block(compute: bool):
                    if not compute:
                        of_o_out.acquire(1)
                        for _ in range_(loop_idx_kv):
                            of_p.acquire(1)
                            of_v.acquire(1)
                            of_scale.acquire(1)
                            of_p.release(1)
                            of_v.release(1)
                            of_scale.release(1)
                        of_o_out.release(1)
                        return
                    for kv in range_(loop_idx_kv - 1):
                        elem_in_p = of_p.acquire(1)
                        elem_in_v = of_v.acquire(1)
                        elt = of_scale.acquire(1)
                        matmul_PV(elem_in_p, elem_in_v, acc, elt, B_q, kv, idx_buffer, s_kv)
                        of_p.release(1)
                        of_v.release(1)
                        of_scale.release(1)
                        idx_buffer[0] += 1
                    elem_in_p = of_p.acquire(1)
                    elem_in_v = of_v.acquire(1)
                    elt = of_scale.acquire(1)
                    matmul_PV(elem_in_p, elem_in_v, acc, elt, B_q, loop_idx_kv - 1, idx_buffer, s_kv)
                    of_p.release(1)
                    of_v.release(1)
                    elem_o_out = of_o_out.acquire(1)
                    sinks_k(elt, table, head[0], q_block_bias, B_q)
                    rescale_O(acc, elem_o_out, elt, B_q, idx_buffer)
                    of_scale.release(1)
                    of_o_out.release(1)
                    idx_buffer[0] = 0
                    idx_buffer[1] += num_pipelines

                def each(heads, q_blocks, q_valid, first, bias, idx, block):
                    # each_q_block's walk, its head index given to mha_sinks
                    q_start, q_offset = first
                    idx[2] = q_offset
                    for h in range_(heads):
                        head[0] = index.casts(T.i32(), h)
                        idx[0] = 0
                        idx[1] = q_start + bias
                        for _ in range_(q_valid):
                            block(True)
                        for _ in range_(q_blocks - q_valid):
                            block(False)

                words = rtp.read()
                heads, q_blocks, q_valid, loop_idx_kv = (words[v] for v in blocks)
                s_kv = words[op.s_kv]
                first = words[op.q_start], words[op.q_offset]
                each(heads, q_blocks, q_valid, first, q_block_bias, idx_buffer, q_block)

            return batched_matmul_pv, [*args, table, kernel]

    MHA.__qualname__ = MHA.__name__ = "MHA"
    _CLASSES[base] = MHA
    return MHA


def _check(a) -> int:
    import iron
    from iron.operators.mha import MHA as IronMHA
    from iron_graph import bind_device, export_graph, write_check

    bind_device("npu2", 8)
    rng = np.random.default_rng(a.seed)
    H, G, D = 64, 8, 64
    sinks = rng.standard_normal(H) * 2 + 1
    if a.sink_const is not None:
        sinks = np.full(H, a.sink_const)
    S = 1 if a.decode else a.rows
    T = a.keys
    cls = sink_class(IronMHA)

    class Gr(iron.Graph):
        def body(self, q, k, v):
            return cls.step(q, k, v, heads_interleaved=True, kv_interleaved=True, sinks=tuple(sinks),
                            window=a.window or None)

        def reference(self, q, k, v):
            op = cls(num_heads=H, num_KV_heads=G, seq_len=S, kv_len=T, heads_interleaved=True,
                     kv_interleaved=True, window=a.window or None, sinks=tuple(sinks))
            return reference(op, np.asarray(q, np.float32), np.asarray(k, np.float32),
                             np.asarray(v, np.float32), sinks).astype(bfloat16)

    g = Gr()
    cg = g.compile(q=(S, H, D), k=(T, G, D), v=(T, G, D))
    m = export_graph(cg, a.out, "sinkmha", dict(decode=a.decode))
    q = (rng.standard_normal((S, H, D)) * 1.0).astype(bfloat16)
    k = (rng.standard_normal((T, G, D)) * 1.0).astype(bfloat16)
    v = rng.standard_normal((T, G, D)).astype(bfloat16)
    write_check(g, m, a.out, dict(q=q, k=k, v=v))
    print(f"{a.out}: MHA with sinks, {S} queries over {T} keys")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("out", type=Path)
    c.add_argument("--decode", action="store_true", help="one query (the packed form)")
    c.add_argument("--rows", type=int, default=256)
    c.add_argument("--keys", type=int, default=512)
    c.add_argument("--window", type=int, default=0)
    c.add_argument("--seed", type=int, default=2)
    c.add_argument("--sink-const", type=float, default=None, help="every head's sink this value")
    return _check(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
