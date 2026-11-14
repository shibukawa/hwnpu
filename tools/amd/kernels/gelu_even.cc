//===- gelu_even.cc - tanh GELU, its argument in float32 -------*- C++ -*-===//
//
// gelu(x) ~ 0.5 x (1 + tanh(u)), u = sqrt(2 / pi) (x + 0.044715 x^3), tanh from IRON's table
// (getTanhBf16, as gelu.cc's use_lut build), the argument formed in float32.
// IRON's kernel forms u from bf16 products (x^2 and s 0.044715 x, each 0.4 % off), which
// changed Anima's pictures; here u is formed in float32 (x^2 exact, the coefficients two bf16
// limbs) and the output rounds to nearest even (the cores start in floor mode, 16.78). The
// table's tanh (bf16) is the rest of the error. tools/amd/iron_gelu.py, design.md 16.200.
//===----------------------------------------------------------------------===//
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "aie_kernel_utils.h"
#include "common/activations.h"

using acc_t = aie::accum<accfloat, 32>;
using bf_t = aie::vector<bfloat16, 32>;

static inline __attribute__((always_inline)) bf_t gelu_even_32(bf_t x) {
  const auto half_x = aie::mul(aie::max(x, bfloat16(-8.0f)), bfloat16(0.5f));
  // u = x (s + sb x^2), s = sqrt(2 / pi) in the accumulator, sb = s 0.044715 in two limbs
  const acc_t sq = aie::mul(x, x);  // exact
  const bf_t sq_hi = sq.to_vector<bfloat16>();
  const bf_t sq_lo = aie::msc(sq, sq_hi, bfloat16(1.0f)).to_vector<bfloat16>();
  const bfloat16 sbh = 0.03564453125f, sbl = 0.00003282f;
  acc_t in = aie::mac(acc_t(aie::broadcast<float, 32>(0.7978845608f)), sq_hi, sbh);
  in = aie::mac(in, sq_lo, sbh);
  in = aie::mac(in, sq_hi, sbl);
  const bf_t in_hi = in.to_vector<bfloat16>();
  const bf_t in_lo = aie::msc(in, in_hi, bfloat16(1.0f)).to_vector<bfloat16>();
  const acc_t u = aie::mac(aie::mul(x, in_hi), x, in_lo);
  const bf_t t = tanh_bf16_vec<32>(u.to_vector<float>());
  return aie::mac(half_x, t, half_x.to_vector<bfloat16>()).to_vector<bfloat16>();
}

extern "C" {

void gelu_even_bf16_size(bfloat16 *restrict input, bfloat16 *restrict output, int32_t vector_size) {
  event0();
  const auto saved_rounding = aie::get_rounding();
  aie::set_rounding(aie::rounding_mode::conv_even);
  auto it_in = aie::begin_restrict_vector<32>(input);
  auto it_out = aie::begin_restrict_vector<32>(output);
  AIE_PREPARE_FOR_POSTPIPELINING
  AIE_LOOP_UNROLL(4)
  for (int i = 0; i < vector_size / 32; i++)
    *it_out++ = gelu_even_32(*it_in++);
  aie::set_rounding(saved_rounding);
  event1();
}

} // extern "C"
