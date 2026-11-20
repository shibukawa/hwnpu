//===- snake_f32.cc - Snake, x + sin(alpha x)^2 / alpha, in float32 -*- C++ -*-===//
//
// Snake1d of the DAC's decoder (MOSS-SoundEffect, tools/amd/iron_moss_dac.py) over a line of
// pixels of SN_C channels (bf16, channel last): y = x + sin(alpha x)^2 inv, alpha and inv = 1 /
// (alpha + 1e-9) a value a channel, each two bf16 limbs (the row holds alpha's high limbs (SN_C),
// alpha's low limbs, inv's high limbs, inv's low limbs). IRON has no sin; as internal/sfx's
// stages for Intel's NPU do (in fp16), v = alpha x is reduced to r = v - n pi (n = round(v / pi),
// by the 1.5 * 2^23 rounding; pi in three limbs) and sin(r) is its Taylor series to r^7 (|r| <= pi / 2), sin^2
// having the period pi.
//
// The accuracy is bf16's of the larger of x and sin^2 inv (the GEMM after reads bf16): the
// sums are float32 (the accumulator), the products of bf16 limbs exact. alpha x, v / pi and n pi
// take two limbs (pi three), so r is float32's; the series takes r's high limb (its terms are at
// most a third of r), the first term r both limbs, sin r and sin^2 two limbs. Against float64
// over 2e6 random x and alpha (iron_snake.kernel_ref is this arithmetic in numpy): the error
// relative to the larger term mean 0.0013, at most 0.014 (bf16's rounding alone 0.0012 and
// 0.0066), in about 20 products where all in three limbs took 60 (1.4 ms a million values).
//===----------------------------------------------------------------------===//
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "aie_kernel_utils.h"

#ifndef SN_C
#error "snake_f32.cc needs -DSN_C (the channels)"
#endif

namespace {
constexpr unsigned L = 32;
static_assert(SN_C % L == 0);
using acc_t = aie::accum<accfloat, L>;
using bf_t = aie::vector<bfloat16, L>;
using f_t = aie::vector<float, L>;

inline __attribute__((always_inline)) acc_t splat(float c) { return acc_t(aie::broadcast<float, L>(c)); }
inline __attribute__((always_inline)) bf_t bsplat(float c) { return aie::broadcast<bfloat16, L>(c); }

// a = hi + lo to about 2^-17 of a
inline __attribute__((always_inline)) void split2(acc_t a, bf_t &hi, bf_t &lo, bf_t one) {
  hi = a.template to_vector<bfloat16>();
  lo = aie::msc(a, hi, one).template to_vector<bfloat16>();
}

struct Consts {
  bf_t one, iph, ipl, pih, pim, pil;
  f_t magic;
  acc_t c0, c1, c2;
};

inline __attribute__((always_inline)) bf_t snake32(bf_t x, bf_t ah, bf_t al, bf_t ih, bf_t il, const Consts &k) {
  bf_t vh, vl, nh, nl, rh, rl, sh, sl, s2h, s2l;
  const acc_t v = aie::mac(aie::mul(x, al), x, ah); // alpha x: two exact products
  split2(v, vh, vl, k.one);
  // n = round(v / pi)
  const acc_t t = aie::mac(aie::mac(aie::mul(vl, k.iph), vh, k.ipl), vh, k.iph);
  const acc_t n(aie::sub(aie::add(t.template to_vector<float>(), k.magic), k.magic));
  split2(n, nh, nl, k.one); // exact while |n| < 2^16
  // r = v - n pi
  acc_t r = aie::msc(v, nh, k.pih);
  r = aie::msc(r, nh, k.pim);
  r = aie::msc(r, nl, k.pih);
  r = aie::msc(r, nh, k.pil);
  split2(r, rh, rl, k.one);
  // sin r = r + r u (c0 + u (c1 + u c2)), u = r^2
  const bf_t u = aie::mul(rh, rh).template to_vector<bfloat16>();
  acc_t p = aie::mac(k.c1, k.c2.template to_vector<bfloat16>(), u);
  p = aie::mac(k.c0, p.template to_vector<bfloat16>(), u);
  const bf_t q = aie::mul(p.template to_vector<bfloat16>(), u).template to_vector<bfloat16>();
  const acc_t s = aie::mac(aie::mac(aie::mul(rl, k.one), rh, q), rh, k.one);
  split2(s, sh, sl, k.one);
  // sin^2 r = sh^2 + 2 sh sl
  const acc_t s2 = aie::mac(aie::mac(aie::mul(sh, sl), sh, sl), sh, sh);
  split2(s2, s2h, s2l, k.one);
  // y = x + sin^2 inv
  acc_t y = aie::mac(aie::mul(s2l, ih), s2h, il);
  y = aie::mac(aie::mac(y, s2h, ih), x, k.one);
  return y.template to_vector<bfloat16>();
}
} // namespace

extern "C" {

void snake_f32(bfloat16 *restrict x, bfloat16 *restrict row, bfloat16 *restrict y, int32_t size) {
  event0();
  const auto saved_rounding = aie::get_rounding();
  aie::set_rounding(aie::rounding_mode::conv_even);
  Consts k;
  k.one = bsplat(1.0f);
  // 1 / pi and pi as bf16 limbs (each exact in bf16; their sums 1e-7 and 2e-9 off)
  k.iph = bsplat(0.318359375f);
  k.ipl = bsplat(-4.9591064453125e-05f);
  k.pih = bsplat(3.140625f);
  k.pim = bsplat(0.00096893310546875f);
  k.pil = bsplat(-1.2814998626708984e-06f);
  k.magic = aie::broadcast<float, L>(12582912.0f);
  k.c0 = splat(-1.0f / 6);
  k.c1 = splat(1.0f / 120);
  k.c2 = splat(-1.0f / 5040);
  const unsigned pixels = (uint32_t)size / SN_C;
  for (unsigned b = 0; b < SN_C / L; b++) {
    const bf_t ah = aie::load_v<L>(row + b * L), al = aie::load_v<L>(row + SN_C + b * L);
    const bf_t ih = aie::load_v<L>(row + 2 * SN_C + b * L), il = aie::load_v<L>(row + 3 * SN_C + b * L);
    const bfloat16 *restrict p = x + b * L;
    bfloat16 *restrict o = y + b * L;
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (unsigned i = 0; i < pixels; i++) {
      aie::store_v(o, snake32(aie::load_v<L>(p), ah, al, ih, il, k));
      p += SN_C;
      o += SN_C;
    }
  }
  aie::set_rounding(saved_rounding);
  event1();
}

} // extern "C"
