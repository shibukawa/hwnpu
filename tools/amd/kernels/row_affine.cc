//===- row_affine.cc - each channel times a factor plus a term, in float32 -*- C++ -*-===//
//
// y = x a + b for a line of pixels of RA_C channels (bf16, channel last), a and b a value a
// channel, each two bf16 limbs: the row holds a's high limbs (RA_C), a's low limbs, b's high
// limbs, b's low limbs. The products and the sum are float32 and round to bf16 once, to nearest
// even: a GroupNorm's y = (x - mean) gamma / sqrt(var + eps) + beta with a = gamma / sqrt(var +
// eps) and b = beta - mean a, whose two terms can each be many times the result.
// tools/amd/iron_groupnorm.py, docs/design/image-amd.md.
//===----------------------------------------------------------------------===//
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "aie_kernel_utils.h"

#ifndef RA_C
#error "row_affine.cc needs -DRA_C (the channels)"
#endif

namespace {
constexpr unsigned L = 64;
static_assert(RA_C % L == 0);
using acc_t = aie::accum<accfloat, L>;
using bf_t = aie::vector<bfloat16, L>;
} // namespace

extern "C" {

void row_affine(bfloat16 *restrict x, bfloat16 *restrict row, bfloat16 *restrict y, int32_t size) {
  event0();
  const auto saved_rounding = aie::get_rounding();
  aie::set_rounding(aie::rounding_mode::conv_even);
  const bf_t one = aie::broadcast<bfloat16, L>(1.0f);
  const unsigned pixels = (uint32_t)size / RA_C;
  for (unsigned b = 0; b < RA_C / L; b++) {
    const bf_t ah = aie::load_v<L>(row + b * L);
    const bf_t al = aie::load_v<L>(row + RA_C + b * L);
    const acc_t base = aie::mac(aie::mul(aie::load_v<L>(row + 3 * RA_C + b * L), one),
                                aie::load_v<L>(row + 2 * RA_C + b * L), one);
    const bfloat16 *restrict p = x + b * L;
    bfloat16 *restrict o = y + b * L;
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (unsigned i = 0; i < pixels; i++) {
      const bf_t v = aie::load_v<L>(p);
      aie::store_v(o, aie::mac(aie::mac(base, v, al), v, ah).to_vector<bfloat16>());
      p += RA_C;
      o += RA_C;
    }
  }
  aie::set_rounding(saved_rounding);
  event1();
}

} // extern "C"
