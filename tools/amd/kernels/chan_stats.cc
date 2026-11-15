//===- chan_stats.cc - each channel's sum and sum of squares over a line -*- C++ -*-===//
//
// A line is CS_P pixels of CS_C channels (bf16, channel last). The output is 2 CS_C float32:
// each channel's sum over the line's pixels, then each channel's sum of squares, accumulated in
// float32 (a square of a bf16 is exact there). A GroupNorm's statistics in one pass over the
// pixels: tools/amd/iron_groupnorm.py, design.md 16.208.
//===----------------------------------------------------------------------===//
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "aie_kernel_utils.h"

#ifndef CS_C
#error "chan_stats.cc needs -DCS_C (the channels) and -DCS_P (the pixels a line)"
#endif

namespace {
constexpr unsigned L = 64;
static_assert(CS_C % L == 0);
using acc_t = aie::accum<accfloat, L>;
} // namespace

extern "C" {

void chan_stats(bfloat16 *restrict x, float *restrict y, int32_t size) {
  event0();
  const auto saved_rounding = aie::get_rounding();
  aie::set_rounding(aie::rounding_mode::conv_even);
  const auto one = aie::broadcast<bfloat16, L>(1.0f);
  for (unsigned b = 0; b < CS_C / L; b++) {
    acc_t s = aie::zeros<accfloat, L>();
    acc_t q = aie::zeros<accfloat, L>();
    const bfloat16 *restrict p = x + b * L;
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (unsigned i = 0; i < CS_P; i++) {
      const aie::vector<bfloat16, L> v = aie::load_v<L>(p);
      s = aie::mac(s, v, one);
      q = aie::mac_square(q, v);
      p += CS_C;
    }
    aie::store_v(y + b * L, s.to_vector<float>());
    aie::store_v(y + CS_C + b * L, q.to_vector<float>());
  }
  aie::set_rounding(saved_rounding);
  event1();
}

} // extern "C"
