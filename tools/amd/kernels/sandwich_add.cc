//===- sandwich_add.cc - a sandwich-normed block's residual and the next norm -*- C++ -*-===//
//
// After a block of a sandwich-normed layer (LFM3's, Gemma's): y = x + rms(h) * w_post, the
// residual, and n = rms(y) * w_pre, the next block's input. IRON's graph runs a decode step's as
// three operators (RMSNorm, ElementwiseAdd, RMSNorm), each a reconfiguration of the array
// (docs/design.md 16.266). One row of SA_N; the sums of squares in float32, rms(h) * w_post,
// y and n each rounded once to bf16 (nearest even). tools/amd/iron_sandwich.py, 16.268.
//===----------------------------------------------------------------------===//
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "common/f32_split.h"
#include "common/scalar_f32.h"

#ifndef SA_N
#define SA_N 2048
#endif
#ifndef SA_EPS
#define SA_EPS 1e-5f
#endif

constexpr int N = SA_N, L = 32;
static_assert(N % L == 0, "sandwich_add: whole vectors");
using bf_t = aie::vector<bfloat16, L>;
using acc_t = aie::accum<accfloat, L>;

namespace {
inline float inv_rms(const bfloat16 *restrict v) {
  acc_t sq = aie::zeros<accfloat, L>();
  for (int i = 0; i < N; i += L) {
    const bf_t x = aie::load_v<L>(v + i);
    sq = aie::mac_square(sq, x);
  }
  const float ms = scalar_mul(aie::reduce_add(sq.to_vector<float>()), 1.0f / N);
  return scalar_invsqrt(ms + SA_EPS);
}

// x (bf16) times s (a float32 scalar) times w, rounded once: x s in three limbs, then times w.
inline bf_t scaled(bf_t x, bf_t w, float s) {
  const Limbs<L> sl = bf16_limbs(acc_t(aie::broadcast<float, L>(s)));
  return mul_limbs(bf16_limbs(mul_limbs(sl, x)), w).to_vector<bfloat16>();
}
} // namespace

extern "C" {

// hw: h, then w_post, then w_pre (N each), with SA_SCALE then scale. x: the residual.
// y: x + rms(h) w_post (with SA_SCALE, that rounded to bf16, then times scale: Gemma 4's
// layer_scalar). n: rms(y) w_pre.
void sandwich_add(const bfloat16 *restrict hw, const bfloat16 *restrict x, bfloat16 *restrict y,
                  bfloat16 *restrict n) {
  const aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  const bfloat16 *h = hw, *wpost = hw + N, *wpre = hw + 2 * N;
  const float sh = inv_rms(h);
  for (int i = 0; i < N; i += L) {
    const bf_t p = scaled(aie::load_v<L>(h + i), aie::load_v<L>(wpost + i), sh);
    const bf_t one = aie::broadcast<bfloat16, L>(1.0f);
    const bf_t r = aie::mac(aie::mul(p, one), aie::load_v<L>(x + i), one).to_vector<bfloat16>();
#ifdef SA_SCALE
    aie::store_v(y + i, aie::mul(r, aie::load_v<L>(hw + 3 * N + i)).to_vector<bfloat16>());
#else
    aie::store_v(y + i, r);
#endif
  }
  const float sy = inv_rms(y);
  for (int i = 0; i < N; i += L)
    aie::store_v(n + i, scaled(aie::load_v<L>(y + i), aie::load_v<L>(wpre + i), sy));
  aie::set_rounding(saved);
}

} // extern "C"
