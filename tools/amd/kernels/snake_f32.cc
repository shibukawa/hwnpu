//===- snake_f32.cc - Snake, x + sin(alpha x)^2 / alpha, in float32 -*- C++ -*-===//
//
// Snake1d of the DAC's decoder (MOSS-SoundEffect, tools/amd/iron_moss_dac.py) over a line of
// pixels of SN_C channels (bf16, channel last): y = x + sin(alpha x)^2 inv, alpha and inv = 1 /
// (alpha + 1e-9) a value a channel, each two bf16 limbs (the row holds alpha's high limbs (SN_C),
// alpha's low limbs, inv's high limbs, inv's low limbs). IRON has no sin; as internal/sfx's
// stages for Intel's NPU do (in fp16), v = alpha x is reduced to r = v - n pi (n = round(v / pi),
// by the 1.5 * 2^23 rounding; pi in three limbs) and sin(r) is its Taylor series to r^11 (|r| <=
// pi / 2: 6e-8 at the ends), sin^2 having the period pi. All in float32 (f32_split.h's limbs),
// rounded once to bf16, to nearest even. The products of an alpha down to 8e-6 (the first
// DecoderBlock's) stay normal: sin(r)^2 is about r^2, and r^2 inv about alpha x^2.
//===----------------------------------------------------------------------===//
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "aie_kernel_utils.h"
#include "common/f32_split.h"

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

inline __attribute__((always_inline)) Limbs<L> two(bf_t hi, bf_t lo) {
  Limbs<L> l;
  l.hi = hi;
  l.mid = lo;
  l.lo = aie::zeros<bfloat16, L>();
  return l;
}

inline __attribute__((always_inline)) bf_t snake32(bf_t x, const Limbs<L> &al, const Limbs<L> &inv,
                                                   const Limbs<L> &inv_pi, const Limbs<L> &neg_pi,
                                                   const acc_t (&c)[5]) {
  const bf_t one = aie::broadcast<bfloat16, L>(1.0f);
  const acc_t v = aie::mac(aie::mul(x, al.mid), x, al.hi); // alpha x: two exact products
  const Limbs<L> vl = bf16_limbs(v);
  // n = round(v / pi)
  const f_t magic = aie::broadcast<float, L>(12582912.0f);
  const f_t t = mul_limbs(vl, inv_pi).template to_vector<float>();
  const acc_t n(aie::sub(aie::add(t, magic), magic));
  const acc_t r = mac_limbs(v, bf16_limbs(n), neg_pi);
  const Limbs<L> rl = bf16_limbs(r);
  const Limbs<L> u = bf16_limbs(mul_limbs(rl, rl));
  // sin(r) = r (1 + u (c0 + u (c1 + u (c2 + u (c3 + u c4)))))
  acc_t p = c[4];
  for (int k = 3; k >= 0; k--)
    p = aie::add(mul_limbs(bf16_limbs(p), u), c[k]);
  p = aie::add(mul_limbs(bf16_limbs(p), u), splat(1.0f));
  const Limbs<L> s = bf16_limbs(mul_limbs(bf16_limbs(p), rl));
  const acc_t s2 = mul_limbs(s, s);
  return mac_limbs(acc_t(aie::mul(x, one)), bf16_limbs(s2), inv).template to_vector<bfloat16>();
}
} // namespace

extern "C" {

void snake_f32(bfloat16 *restrict x, bfloat16 *restrict row, bfloat16 *restrict y, int32_t size) {
  event0();
  const auto saved_rounding = aie::get_rounding();
  aie::set_rounding(aie::rounding_mode::conv_even);
  const Limbs<L> inv_pi = bf16_limbs(splat(0.318309886f));
  const Limbs<L> neg_pi = bf16_limbs(splat(-3.14159265f));
  const acc_t c[5] = {splat(-1.0f / 6), splat(1.0f / 120), splat(-1.0f / 5040), splat(1.0f / 362880),
                      splat(-1.0f / 39916800)};
  const unsigned pixels = (uint32_t)size / SN_C;
  for (unsigned b = 0; b < SN_C / L; b++) {
    const Limbs<L> al = two(aie::load_v<L>(row + b * L), aie::load_v<L>(row + SN_C + b * L));
    const Limbs<L> inv = two(aie::load_v<L>(row + 2 * SN_C + b * L), aie::load_v<L>(row + 3 * SN_C + b * L));
    const bfloat16 *restrict p = x + b * L;
    bfloat16 *restrict o = y + b * L;
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (unsigned i = 0; i < pixels; i++) {
      aie::store_v(o, snake32(aie::load_v<L>(p), al, inv, inv_pi, neg_pi, c));
      p += SN_C;
      o += SN_C;
    }
  }
  aie::set_rounding(saved_rounding);
  event1();
}

} // extern "C"
