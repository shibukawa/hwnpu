//===- elu_f32.cc - ELU, its exponential in float32 -------------*- C++ -*-===//
//
// elu(x) = x for x > 0, exp(x) - 1 otherwise: the exponential (exp_f32) of min(x, 0) clamped at
// -30 (where it is 0 in bf16), minus 1, plus max(x, 0), in float32 and rounded once to bf16, to
// nearest even. IRON has no ELU (nor an exponential of its own as an operator); SpectroStream's
// decoder (Magenta RealTime 2, tools/amd/iron_magenta.py) runs one before each convolution.
//===----------------------------------------------------------------------===//
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "aie_kernel_utils.h"
#include "common/exp_f32.h"

namespace {
constexpr unsigned L = 16;
using acc_t = aie::accum<accfloat, L>;
using bf_t = aie::vector<bfloat16, L>;

inline __attribute__((always_inline)) bf_t elu16(bf_t x) {
  const bf_t zero = aie::zeros<bfloat16, L>();
  const bf_t neg = aie::max(aie::min(x, zero), aie::broadcast<bfloat16, L>(-30.0f));
  const bf_t pos = aie::max(x, zero);
  const bf_t one = aie::broadcast<bfloat16, L>(1.0f);
  const acc_t e = exp_f32<L>(acc_t(aie::mul(neg, one)));
  const acc_t em1 = aie::add(e, acc_t(aie::broadcast<float, L>(-1.0f)));
  const acc_t r = aie::add(em1, acc_t(aie::mul(pos, one)));
  return r.to_vector<bfloat16>();
}
} // namespace

extern "C" {

void elu_f32_bf16_size(bfloat16 *restrict input, bfloat16 *restrict output, int32_t vector_size) {
  event0();
  const auto saved_rounding = aie::get_rounding();
  aie::set_rounding(aie::rounding_mode::conv_even);
  auto it_in = aie::begin_restrict_vector<L>(input);
  auto it_out = aie::begin_restrict_vector<L>(output);
  for (int i = 0; i < vector_size / (int)L; i++)
    *it_out++ = elu16(*it_in++);
  aie::set_rounding(saved_rounding);
  event1();
}

} // extern "C"
