//===- short_conv.cc - LFM2/LFM3's gated short convolution, a decode step -*- C++ -*-===//
//
// One row of a gated short convolution (LFM2 and LFM3's conv layers): in_proj's row gives B,
// C and x; u = B * x goes through a depth-wise causal filter of SC_K taps over the SC_K - 1
// rows before it (the state) and C gates the result: y = C * conv(u). The state's next rows
// are the last SC_K - 2 of the old and u. tools/amd/iron_shortconv.py (ShortConvStep),
// docs/design.md 16.264.
//
// IRON's graph ran this as about 16 operators (copies into a window, a multiply and an add a
// tap, the gate's two multiplies), each a reconfiguration of the array: about 1.1 ms a layer.
// Here the sums are float32, u, conv(u) and y each rounded once to bf16 (nearest even), as
// IRON's rounds u and C * conv(u) (and each partial sum, which this does not).
//===----------------------------------------------------------------------===//
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef SC_W
#define SC_W 256 // channels a core
#endif
#ifndef SC_K
#define SC_K 7 // taps
#endif

constexpr int W = SC_W, K = SC_K, L = 16;
static_assert(W % L == 0 && K >= 2, "short_conv: whole vectors, two taps or more");

extern "C" {

// xt: B, C, x (W each), then the taps (K rows of W, the oldest row's first). st: the K - 1 rows
// before this one (oldest first). y: C * conv(u). fin: the K - 1 rows ending with u.
void short_conv_step(const bfloat16 *restrict xt, const bfloat16 *restrict st, bfloat16 *restrict y,
                     bfloat16 *restrict fin) {
  const aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  const bfloat16 *b = xt, *c = xt + W, *x = xt + 2 * W, *taps = xt + 3 * W;
  for (int i = 0; i < W; i += L) {
    const aie::vector<bfloat16, L> u = aie::mul(aie::load_v<L>(b + i), aie::load_v<L>(x + i)).to_vector<bfloat16>();
    aie::accum<accfloat, L> acc = aie::mul(aie::load_v<L>(taps + (K - 1) * W + i), u);
    for (int k = 0; k < K - 1; k++)
      acc = aie::mac(acc, aie::load_v<L>(taps + k * W + i), aie::load_v<L>(st + k * W + i));
    const aie::vector<bfloat16, L> v = acc.to_vector<bfloat16>();
    aie::store_v(y + i, aie::mul(aie::load_v<L>(c + i), v).to_vector<bfloat16>());
    for (int k = 0; k < K - 2; k++)
      aie::store_v(fin + k * W + i, aie::load_v<L>(st + (k + 1) * W + i));
    aie::store_v(fin + (K - 2) * W + i, u);
  }
  aie::set_rounding(saved);
}

} // extern "C"
