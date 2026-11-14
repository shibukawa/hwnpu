"""IRON's MHA with the blocks and the product formats an image model's attention runs fastest at.

An image model's attention attends every query to every key (no causal mask, no window) over
thousands of rows: Anima's DiT 4096 x 4096 in 16 heads of 128, its cross-attention 4096 x 512.
IRON's MHA takes there B_q = B_kv = 32 (64 does not fit a core's memory at d = 128) and native
bf16 products, and measured (Ryzen AI 9 365, 4096 x 4096 x 16 x 128) 146 ms. Two things held it:

- mha.cc's vectorized softmax (``partial_softmax_rows``) takes 64-key rows only; at 32 keys it
  goes row by row, and that core bounded the pipeline. Unmasked, the key block need not be the
  query block (mha.cc's causal skip is what compares them), so keys go 64 at a time.
- AIE2P's native bf16 mmul ran QK^T at about 24 macs a cycle (32 x 128 x 64 in 10 800 cycles);
  through bfp16 (``emulate_bf16_mmul_with_bfp16``, as ``Precision.MEDIUM``'s GEMMs) 1 400. IRON
  keeps the attention's products in bf16 because P*V's bfp16 shares an exponent over 8 keys
  and the softmax bounded the time; with P*V on cores of its own (``pv_cores`` 2) QK^T's object
  is its own and takes bfp16 alone. P*V may too (``pv_bfp16``).

146 -> 59 ms (QK^T bfp16), 54 ms (and P*V), against the reference's cosine 0.99998 -> 0.99994,
0.99988. AIE2P's exp2 instruction (``accurate_exp2`` off) would save 10 ms more and costs 4x the
error, so the softmax keeps its polynomial (design.md 16.147).

``zero_keys``: a cross-attention to a text whose rows past the prompt are zeros (Anima's
conditioning) need not take those keys. Bounded to the prompt's keys (``k[:n]``, MHA's
kv_valid, a per-call word), the call runs only their blocks, and kernels/mha_zero_keys.cc adds
the left-out keys' term n e^{0 - m} to each row's softmax sum before the division: the output is
the 512-key call's, each zero key scoring 0 (design.md 16.204).
"""

from pathlib import Path

import numpy as np
from aie.utils.compile.jit.params import param

from iron.operators.mha import MHA


class ImageMHA(MHA):
    """MHA whose unmasked key blocks are 64 wide, and whose products may be bfp16.

    ``qk_bfp16`` and ``pv_bfp16`` are declared parameters, so IRON's compile cache (keyed on an
    operator's parameters) tells the builds apart. Pass ``B_q=32, pv_cores=2`` at d = 128.
    """

    qk_bfp16: bool = param(default=True, array=True)
    pv_bfp16: bool = param(default=False, array=True)
    # The keys a call stands for, those past its bound zeros counted in the softmax (module
    # docstring); 0: off. It may be less than kv_len, K's and V's rows in memory, which the
    # bound keeps it from reading past.
    zero_keys: int = param(default=0, array=True)

    def unmasked(self) -> bool:
        return not self.packed and not self.causal and self.window is None

    def compatible(self) -> None:
        # MHA's queries are its keys' last rows (kv_len >= seq_pad), which only its causal and
        # window masks read; unmasked, zero_keys's K and V may be shorter than the queries
        # (Anima's 4096 patches over 512 text rows, one call)
        short = self.zero_keys and self.unmasked() and self.kv_len < self.seq_pad
        if not short:
            return super().compatible()
        kept = vars(self)["kv_len"]
        vars(self)["kv_len"] = self.seq_pad
        try:
            super().compatible()
        finally:
            vars(self)["kv_len"] = kept

    def ops(self) -> int:
        if not self.zero_keys:
            return super().ops()
        return 4 * self.num_heads * self.d * self.seq_len * int(self.zero_keys)

    def key_block(self, B_q, dev):
        if self.unmasked() and self.kv_len % 64 == 0 and B_q <= 64:
            return 64
        return super().key_block(B_q, dev)

    def mha_kernel(self, B_q: int, width: int, B_kv: int, accurate_exp2=False):
        from aie.iron import kernels

        # With one P*V core, QK^T and P*V share mha.cc's object: bfp16 then only if both are.
        qk_object = width == self.d
        if qk_object and self.pv_cores == 1:
            emulate = self.qk_bfp16 and self.pv_bfp16
        else:
            emulate = self.qk_bfp16 if qk_object else self.pv_bfp16
        return kernels.linalg.mha(
            B_q,
            width,
            B_kv,
            b_col_maj=True,
            accurate_exp2=accurate_exp2,
            causal=self.causal,
            window=self.window or 0,
            packed=self.packed and self.window is not None,
            emulate_bf16_mmul_with_bfp16=emulate and self.unmasked(),
        )

    def reference(self, Q, K, V, s_q=None, s_kv=None, **kw):
        if not self.zero_keys:
            return super().reference(Q, K, V, s_q=s_q, s_kv=s_kv, **kw)
        # the bounded keys, then zeros up to kv_len (keys first: kv_interleaved)
        n = self.kv_tokens if s_kv is None else int(s_kv)
        rows = int(self.zero_keys)
        axis = 0 if self.kv_interleaved else 1

        def padded(t):
            t = np.asarray(t)
            keep = np.take(t, np.arange(min(n, t.shape[axis])), axis=axis)
            shape = list(t.shape)
            shape[axis] = rows
            out = np.zeros(shape, t.dtype)
            idx = [slice(None)] * t.ndim
            idx[axis] = slice(0, keep.shape[axis])
            out[tuple(idx)] = keep
            return out

        return super().reference(Q, padded(K), padded(V), s_q=s_q, s_kv=rows, **kw)

    def build(self):
        if not self.zero_keys:
            return super().build()
        import iron.operators.mha as mha_module

        made = mha_module.Worker
        op = self

        def worker(fn, fn_args=(), **kw):
            if getattr(fn, "__name__", "") == "batched_matmul_pv":
                fn, fn_args = op._zero_keys_pv(fn, list(fn_args))
            return made(fn, fn_args=fn_args, **kw)

        mha_module.Worker = worker
        try:
            return super().build()
        finally:
            mha_module.Worker = made

    def _zero_keys_pv(self, fn, args):
        """The base's P*V program with mha_zero_keys before the last block's rescale_O (the
        closure's values as the base's build bound them; iron_sinks.sink_class's way)."""
        from aie.iron.controlflow import range_
        from aie.iron.kernel import ExternalFunction
        from aie.utils import config
        from ml_dtypes import bfloat16

        free = dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in fn.__closure__)))
        B_q, num_pipelines, blocks = free["B_q"], free["num_pipelines"], free["blocks"]
        each_q_block = free["each_q_block"]
        inv_scale = float(bfloat16(self.scale * np.log2(np.e)))  # the base build's (mha.py)
        here = Path(__file__).resolve().parent / "kernels"
        kdir = Path(config.aie_kernels_dir())
        s_ty = np.ndarray[(4 * B_q,), np.dtype[np.float32]]
        kernel = ExternalFunction(
            "mha_zero_keys", source_file=str(here / "mha_zero_keys.cc"),
            include_dirs=[str(kdir), config.cxx_header_path()],
            compile_flags=[f"-DZK_SCALE={inv_scale!r}f", f"-DZK_KV_LEN={int(self.zero_keys)}"],
            object_file_name=f"mha_zero_keys_{int(self.zero_keys)}_{inv_scale:.6g}.o",
            arg_types=[s_ty, np.int32, np.int32])
        op = self

        def batched_matmul_pv(of_p, of_v, of_scale, of_o_out, matmul_PV, rescale_O, q_block_bias, rtp,
                              idx_buffer, acc, zero_keys):
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
                zero_keys(elt, s_kv, B_q)
                rescale_O(acc, elem_o_out, elt, B_q, idx_buffer)
                of_scale.release(1)
                of_o_out.release(1)
                idx_buffer[0] = 0
                idx_buffer[1] += num_pipelines

            words = rtp.read()
            heads, q_blocks, q_valid, loop_idx_kv = (words[v] for v in blocks)
            s_kv = words[op.s_kv]
            first = words[op.q_start], words[op.q_offset]
            each_q_block(heads, q_blocks, q_valid, first, q_block_bias, idx_buffer, q_block)

        return batched_matmul_pv, [*args, kernel]


def attention(q, k, v, *, d: int, qk_bfp16: bool = True, pv_bfp16: bool = False, zero_keys: int = 0):
    """Unmasked multi-head attention of (rows, heads, d) q, k and v, interleaved as projections
    write them, on eight pipelines: ImageMHA at d = 128 (B_q 32, P*V on two cores), else MHA.
    ``zero_keys``: k and v are bounded (``k[:n]``) and stand for that many keys, those past n
    zeros (ImageMHA's)."""
    if zero_keys:
        if d != 128:
            raise ValueError("zero_keys: ImageMHA's, at d = 128")
        return ImageMHA.step(q, k, v, heads_interleaved=True, kv_interleaved=True, causal=False,
                             num_pipelines=8, B_q=32, pv_cores=2, qk_bfp16=qk_bfp16, pv_bfp16=pv_bfp16,
                             zero_keys=int(zero_keys))
    if d != 128:
        return MHA.step(q, k, v, heads_interleaved=True, kv_interleaved=True, causal=False,
                        num_pipelines=8)
    return ImageMHA.step(q, k, v, heads_interleaved=True, kv_interleaved=True, causal=False,
                         num_pipelines=8, B_q=32, pv_cores=2, qk_bfp16=qk_bfp16, pv_bfp16=pv_bfp16)


def _check(argv=None) -> int:
    """One zero_keys attention (--rows queries over the first --keys of --kv keys, the rest
    zeros) as a graph with its reference, for tools/amd/irongraph; prints the reference against
    the float32 attention over every key, the zeros attended (design.md 16.204):

        python tools/amd/iron_mha.py OUT --rows 4096 --kv 512 --keys 77   # Anima's cross-attention
        go run ./tools/amd/irongraph OUT
    """
    import argparse
    import sys

    import iron
    from iron import Scratchpad
    from ml_dtypes import bfloat16

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from iron_graph import bind_device, export_graph, write_check

    ap = argparse.ArgumentParser(description=_check.__doc__)
    ap.add_argument("out", type=Path)
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--kv", type=int, default=512)
    ap.add_argument("--keys", type=int, default=77, help="the call's keys (the rest zeros)")
    ap.add_argument("--plain", action="store_true", help="every key, no bound (ImageMHA as the DiT's)")
    ap.add_argument("--stands", type=int, default=0, help="the keys the call stands for (default --kv)")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--spread", type=float, default=0, help="scores of about this standard deviation")
    ap.add_argument("--negative", type=float, default=0,
                    help="scores about this far below 0 (q and k of opposite signs): the zero keys' 0 "
                         "then above every prompt key's score")
    a = ap.parse_args(argv)
    bind_device("npu2", 8)
    H, D, L, S = 16, 128, a.rows, a.kv
    T = a.stands or S  # the keys attended in the answer: the first T rows of k and v

    class G(iron.Graph):
        def body(self, q, k, v, *, n: Scratchpad[np.int32]):
            if a.plain:
                return attention(q, k, v, d=D)
            return attention(q, k[:n], v[:n], d=D, zero_keys=T)

    g = G()
    cg = g.compile(q=(L, H, D), k=(S, H, D), v=(S, H, D))
    m = export_graph(cg, a.out, "zkmha", dict(keys=a.keys, stands=T))
    rng = np.random.default_rng(a.seed)
    # scores as an RMS-normed q and k give (|s| up to tens), values about 1
    q = (rng.standard_normal((L, H, D)) * 0.6).astype(bfloat16)
    k = (rng.standard_normal((S, H, D)) * 0.6).astype(bfloat16)
    v = rng.standard_normal((S, H, D)).astype(bfloat16)
    if a.spread:
        # q . k / sqrt(D) of standard deviation spread: q and k each N(0, spread)
        q = (rng.standard_normal((L, H, D)) * np.sqrt(a.spread)).astype(bfloat16)
        k = (rng.standard_normal((S, H, D)) * np.sqrt(a.spread)).astype(bfloat16)
    if a.negative:
        # |q| |k| over 128 dimensions, scaled: scores about -negative (to about +-30 %)
        gain = np.sqrt(a.negative * np.sqrt(D) / (D * 0.64))
        q = (np.abs(rng.standard_normal((L, H, D))) * gain).astype(bfloat16)
        k = (-np.abs(rng.standard_normal((S, H, D))) * gain).astype(bfloat16)
    k[a.keys:] = 0
    v[a.keys:] = 0
    if T < S:  # rows past those the call stands for, which it must not read
        k[T:] = 1e4
        v[T:] = 1e4
    write_check(g, m, a.out, dict(q=q, k=k, v=v), dict(n=a.keys))
    # the float32 answer over the T keys (the zeros attended), against the graph's reference
    qf, kf, vf = (np.swapaxes(np.asarray(t, np.float32), 0, 1) for t in (q, k[:T], v[:T]))
    s = qf @ np.swapaxes(kf, 1, 2) / np.sqrt(D)
    e = np.exp(s - s.max(-1, keepdims=True))
    full = np.swapaxes((e / e.sum(-1, keepdims=True)) @ vf, 0, 1)
    ref = np.fromfile(a.out / "check" / f"out_{m['graph']['outputs'][0]['name']}.bin", np.uint16)
    ref = (ref.astype(np.uint32) << 16).view(np.float32).reshape(full.shape)
    err = np.abs(ref - full).max() / np.abs(full).max()
    print(f"{a.out}: {L} queries over {a.keys} of {T} keys ({S} rows); reference vs all keys attended: max rel {err:.2e}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_check())
