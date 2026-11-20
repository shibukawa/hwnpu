//===- short_conv.cc - LFM2/LFM3's gated short convolution, a decode step -*- C++ -*-===//
//
// One row of a gated short convolution (LFM2 and LFM3's conv layers): in_proj's row gives B,
// C and x; u = B * x goes through a depth-wise causal filter of SC_K taps over the SC_K - 1
// rows before it (the state) and C gates the result: y = C * conv(u). The state's next rows
// are the last SC_K - 2 of the old and u. tools/amd/iron_shortconv.py (ShortConvStep),
// docs/design.md 16.268.
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

// A prompt chunk's (ShortConvChunk): rows in tiles of SC_T, the K - 1 rows before each tile and
// its own in hist ((K - 1 + SC_T) rows of W). The same sums and roundings as short_conv_step.
// With SC_PLAIN (Qwen3.5's Gated DeltaNet, before GatedDeltaRule): a row is x alone, u = x, and
// y = silu(conv(u)) on the first W - SC_RAW channels (q, k, v), conv(u) on the last SC_RAW (a, b),
// silu in float32 rounded once as GdnSiLU's.
#ifndef SC_T
#define SC_T 8
#endif
constexpr int T = SC_T;
#if defined(SC_PLAIN) || defined(SC_GDN)
#include "common/exp_f32.h"
#include "common/f32_split.h"
namespace {
inline aie::vector<bfloat16, L> silu_v(aie::vector<bfloat16, L> x) {
  const aie::accum<accfloat, L> e = exp_f32<L>(aie::mul(x, aie::broadcast<bfloat16, L>(-1.0f)));
  const aie::vector<float, L> d =
      aie::add(e, aie::accum<accfloat, L>(aie::broadcast<float, L>(1.0f))).to_vector<float>();
  const aie::accum<accfloat, L> q(aie::inv(d));
  return mul_limbs(bf16_limbs(q), x).to_vector<bfloat16>();
}
} // namespace
#endif
#ifdef SC_PLAIN
constexpr int RAW = SC_RAW;
static_assert(RAW % L == 0 && RAW < W, "short_conv: whole vectors of a and b");
#endif

extern "C" {

// ts: the taps (K rows of W), then the state (the K - 1 rows before the call, oldest first).
void short_conv_begin(const bfloat16 *restrict ts, bfloat16 *restrict hist) {
  const bfloat16 *st = ts + K * W;
  for (int i = 0; i < (K - 1) * W; i += L)
    aie::store_v(hist + i, aie::load_v<L>(st + i));
}

// Tile t of the call's rows (p: SC_T rows of B, C, x, W each; with SC_PLAIN of x), the call n rows: those of the
// tile below n, each u = B * x, y = C * conv(u); then hist's first K - 1 rows the last before
// the next tile's.
void short_conv_tile(const bfloat16 *restrict p, const bfloat16 *restrict ts, bfloat16 *restrict hist,
                     bfloat16 *restrict y, int32_t t, int32_t n) {
  int rows = n - t * T;
  if (rows > T)
    rows = T;
  if (rows < T) // the rows past the call's: zeros (what reads past them reads no stale words)
    for (int i = (rows < 0 ? 0 : rows) * W; i < T * W; i += L)
      aie::store_v(y + i, aie::zeros<bfloat16, L>());
  if (rows <= 0)
    return;
  const aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  const bfloat16 *taps = ts;
  for (int r = 0; r < rows; r++) {
    bfloat16 *h = hist + r * W;
#ifdef SC_PLAIN
    const bfloat16 *x = p + r * W;
    for (int i = 0; i < W; i += L) {
      const aie::vector<bfloat16, L> u = aie::load_v<L>(x + i);
#else
    const bfloat16 *b = p + r * 3 * W, *c = b + W, *x = b + 2 * W;
    for (int i = 0; i < W; i += L) {
      const aie::vector<bfloat16, L> u = aie::mul(aie::load_v<L>(b + i), aie::load_v<L>(x + i)).to_vector<bfloat16>();
#endif
      aie::store_v(h + (K - 1) * W + i, u);
      aie::accum<accfloat, L> acc = aie::mul(aie::load_v<L>(taps + (K - 1) * W + i), u);
      for (int k = 0; k < K - 1; k++)
        acc = aie::mac(acc, aie::load_v<L>(taps + k * W + i), aie::load_v<L>(h + k * W + i));
      const aie::vector<bfloat16, L> v = acc.to_vector<bfloat16>();
#ifdef SC_PLAIN
      aie::store_v(y + r * W + i, i < W - RAW ? silu_v(v) : v);
#else
      aie::store_v(y + r * W + i, aie::mul(aie::load_v<L>(c + i), v).to_vector<bfloat16>());
#endif
    }
  }
  for (int j = 0; j < K - 1; j++)
    for (int i = 0; i < W; i += L)
      aie::store_v(hist + j * W + i, aie::load_v<L>(hist + (rows + j) * W + i));
  aie::set_rounding(saved);
}

// The state after the call: hist's first K - 1 rows.
void short_conv_end(const bfloat16 *restrict hist, bfloat16 *restrict fin) {
  for (int i = 0; i < (K - 1) * W; i += L)
    aie::store_v(fin + i, aie::load_v<L>(hist + i));
}

} // extern "C"

#ifdef SC_GDN
// Gated DeltaNet's convolution of a prompt chunk (GdnConvChunk), a core's N = SC_W channels of a
// column's: a row a call, the last K rows in a ring (hist, K rows of W: row t of the call at
// (t + K - 1) % K, the state's K - 1 rows before it). y = silu(conv(x)) but on the last raw
// channels (a and b), as IRON's convolution then GdnSiLU (float32 sums rounded once).
extern "C" {

void gdn_conv_begin(const bfloat16 *restrict ts, bfloat16 *restrict hist) {
  const bfloat16 *st = ts + K * W;
  for (int i = 0; i < (K - 1) * W; i += L)
    aie::store_v(hist + i, aie::load_v<L>(st + i));
}

void gdn_conv_row(const bfloat16 *restrict x, const bfloat16 *restrict ts, bfloat16 *restrict hist,
                  bfloat16 *restrict y, int32_t t, int32_t n, int32_t raw) {
  if (t >= n) { // a row past the call's: zeros (GatedDeltaRule reads past them)
    for (int i = 0; i < W; i += L)
      aie::store_v(y + i, aie::zeros<bfloat16, L>());
    return;
  }
  const aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  bfloat16 *now = hist + ((t + K - 1) % K) * W;
  for (int i = 0; i < W; i += L)
    aie::store_v(now + i, aie::load_v<L>(x + i));
  for (int i = 0; i < W; i += L) {
    aie::accum<accfloat, L> acc = aie::mul(aie::load_v<L>(ts + i), aie::load_v<L>(hist + (t % K) * W + i));
    for (int k = 1; k < K; k++)
      acc = aie::mac(acc, aie::load_v<L>(ts + k * W + i), aie::load_v<L>(hist + ((t + k) % K) * W + i));
    const aie::vector<bfloat16, L> v = acc.to_vector<bfloat16>();
    aie::store_v(y + i, i < W - raw ? silu_v(v) : v);
  }
  aie::set_rounding(saved);
}

// The state after the call's n rows: the K - 1 rows ending with row n - 1.
void gdn_conv_end(const bfloat16 *restrict hist, bfloat16 *restrict fin, int32_t n) {
  for (int j = 0; j < K - 1; j++)
    for (int i = 0; i < W; i += L)
      aie::store_v(fin + j * W + i, aie::load_v<L>(hist + ((n + j) % K) * W + i));
}

} // extern "C"
#endif
