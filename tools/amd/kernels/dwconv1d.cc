//===- dwconv1d.cc - a depthwise 1-D convolution of K dilated taps and a bias -*- C++ -*-===//
//
// tools/amd/iron_dwconv1d.py's DWConv1d: DW_R output rows of DW_C channels (channels last) from
// DW_R + (DW_K - 1) DW_D rows of the padded input: y[r] = bias + sum_t w[t] x[r + t DW_D], in a
// float32 accumulator, rounded once to bf16 (nearest even). The weights are one buffer
// (DW_K + 1, DW_C): row t the tap t, the last row the bias. Supertonic's ConvNeXt blocks
// (iron_supertonic.py) take 5 taps of dilation 1 to 8 (the flow) and 7 of 1 to 4 (the vocoder);
// IRON's DepthwiseConv1d is 5 causal taps without a bias.
//===----------------------------------------------------------------------===//
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "aie_kernel_utils.h"

#ifndef DW_C
#define DW_C 128
#endif
#ifndef DW_K
#define DW_K 5
#endif
#ifndef DW_D
#define DW_D 1
#endif
#ifndef DW_R
#define DW_R 16
#endif
static_assert(DW_C % 32 == 0);

extern "C" {

void dwconv1d_bias(const bfloat16 *__restrict w, const bfloat16 *__restrict x, bfloat16 *__restrict y) {
  event0();
  const auto saved_rounding = aie::get_rounding();
  aie::set_rounding(aie::rounding_mode::conv_even);
  const aie::vector<bfloat16, 32> one = aie::broadcast<bfloat16, 32>(1.0f);
  AIE_LOOP_NO_UNROLL
  for (int r = 0; r < DW_R; r++) {
    const bfloat16 *xr = x + r * DW_C;
    bfloat16 *yr = y + r * DW_C;
    AIE_LOOP_UNROLL_FULL
    for (int o = 0; o < DW_C; o += 32) {
      aie::accum<accfloat, 32> acc = aie::mul(aie::load_v<32>(w + DW_K * DW_C + o), one);
      AIE_LOOP_UNROLL_FULL
      for (int t = 0; t < DW_K; t++)
        acc = aie::mac(acc, aie::load_v<32>(xr + t * DW_D * DW_C + o), aie::load_v<32>(w + t * DW_C + o));
      aie::store_v(yr + o, acc.to_vector<bfloat16>());
    }
  }
  aie::set_rounding(saved_rounding);
  event1();
}

} // extern "C"
