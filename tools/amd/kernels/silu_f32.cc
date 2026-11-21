//===- silu_f32.cc - SiLU, its sigmoid in float32 ---------------*- C++ -*-===//
//
// silu(x) = x / (1 + exp(-x)): the exponential (exp_f32) and the inverse in float32, the product
// with x rounded once to bf16, to nearest even (the cores start in floor mode, docs/design/amd.md).
//
// IRON's SiLU multiplies x by a bf16 table's sigmoid: 1 % off on average over [-6, 2] (2 % at
// most), and exactly 0 from x = -8 down. A SwiGLU MLP multiplies it by the other branch, which in
// a language model's few large channels is in the tens or hundreds, so the zeros and the 1 % are
// what its output is made of: Qwen3-4B's hidden states (FLUX.2 klein's text encoder) were at
// cosine 0.9994 of float32's after three layers. tools/amd/iron_silu.py.
//===----------------------------------------------------------------------===//
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "aie_kernel_utils.h"
#include "common/exp_f32.h"
#include "common/f32_split.h"

namespace {
constexpr unsigned L = 16;
using acc_t = aie::accum<accfloat, L>;
using bf_t = aie::vector<bfloat16, L>;

inline __attribute__((always_inline)) bf_t silu16(bf_t x) {
  const acc_t e = exp_f32<L>(acc_t(aie::mul(x, aie::broadcast<bfloat16, L>(-1.0f))));
  const aie::vector<float, L> d = aie::add(e, acc_t(aie::broadcast<float, L>(1.0f))).to_vector<float>();
  const acc_t q(aie::inv(d));
  return mul_limbs(bf16_limbs(q), x).to_vector<bfloat16>();
}
} // namespace

extern "C" {

void silu_f32_bf16_size(bfloat16 *restrict input, bfloat16 *restrict output, int32_t vector_size) {
  event0();
  const auto saved_rounding = aie::get_rounding();
  aie::set_rounding(aie::rounding_mode::conv_even);
  auto it_in = aie::begin_restrict_vector<L>(input);
  auto it_out = aie::begin_restrict_vector<L>(output);
  for (int i = 0; i < vector_size / (int)L; i++)
    *it_out++ = silu16(*it_in++);
  aie::set_rounding(saved_rounding);
  event1();
}

} // extern "C"
