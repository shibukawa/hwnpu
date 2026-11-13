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
"""

from aie.utils.compile.jit.params import param

from iron.operators.mha import MHA


class ImageMHA(MHA):
    """MHA whose unmasked key blocks are 64 wide, and whose products may be bfp16.

    ``qk_bfp16`` and ``pv_bfp16`` are declared parameters, so IRON's compile cache (keyed on an
    operator's parameters) tells the builds apart. Pass ``B_q=32, pv_cores=2`` at d = 128.
    """

    qk_bfp16: bool = param(default=True, array=True)
    pv_bfp16: bool = param(default=False, array=True)

    def unmasked(self) -> bool:
        return not self.packed and not self.causal and self.window is None

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


def attention(q, k, v, *, d: int, qk_bfp16: bool = True, pv_bfp16: bool = False):
    """Unmasked multi-head attention of (rows, heads, d) q, k and v, interleaved as projections
    write them, on eight pipelines: ImageMHA at d = 128 (B_q 32, P*V on two cores), else MHA."""
    if d != 128:
        return MHA.step(q, k, v, heads_interleaved=True, kv_interleaved=True, causal=False,
                        num_pipelines=8)
    return ImageMHA.step(q, k, v, heads_interleaved=True, kv_interleaved=True, causal=False,
                         num_pipelines=8, B_q=32, pv_cores=2, qk_bfp16=qk_bfp16, pv_bfp16=pv_bfp16)
