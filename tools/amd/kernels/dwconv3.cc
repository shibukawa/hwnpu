//===- dwconv3.cc - a k x k depthwise convolution, a row at a time -*- C++ -*-===//
//
// tools/amd/iron_dwconv.py's DWConv: a core holds DW_K rows of a zero-padded image (W + DW_K - 1
// pixels of DW_C channels each, channels last) and writes one output row (W pixels): for each
// pixel and each vector of 32 channels, the bias plus the DW_K^2 taps times their neighbours, in
// a float32 accumulator, rounded once to bf16 (nearest even). The taps are (DW_K^2 + 1, DW_C):
// row dy DW_K + dx a tap, the last row the bias. Sana's GLUMBConv (3 x 3) and its VAE's
// EfficientViT (5 x 5) ran this as a row product and a sum a tap (design.md 16.200).
//===----------------------------------------------------------------------===//
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "aie_kernel_utils.h"

#ifndef DW_W
#define DW_W 32
#endif
#ifndef DW_C
#define DW_C 96
#endif
#ifndef DW_K
#define DW_K 3
#endif
static_assert(DW_C % 32 == 0);
constexpr int KK = DW_K * DW_K;

static inline void dwconv_row(bfloat16 *const *rows, const bfloat16 *restrict taps, bfloat16 *restrict out) {
  const auto saved_rounding = aie::get_rounding();
  aie::set_rounding(aie::rounding_mode::conv_even);
  for (int cv = 0; cv < DW_C; cv += 32) {
    aie::vector<bfloat16, 32> t[KK];
    AIE_LOOP_UNROLL_FULL
    for (int k = 0; k < KK; k++)
      t[k] = aie::load_v<32>(taps + k * DW_C + cv);
    const aie::vector<bfloat16, 32> b = aie::load_v<32>(taps + KK * DW_C + cv);
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int x = 0; x < DW_W; x++) {
      aie::accum<accfloat, 32> acc;
      acc.from_vector(b);
      AIE_LOOP_UNROLL_FULL
      for (int dy = 0; dy < DW_K; dy++) {
        const bfloat16 *p = rows[dy] + x * DW_C + cv;
        AIE_LOOP_UNROLL_FULL
        for (int dx = 0; dx < DW_K; dx++)
          acc = aie::mac(acc, aie::load_v<32>(p + dx * DW_C), t[dy * DW_K + dx]);
      }
      aie::store_v(out + x * DW_C + cv, acc.to_vector<bfloat16>());
    }
  }
  aie::set_rounding(saved_rounding);
}

extern "C" {

#if DW_K == 3
void dwconv3_row(bfloat16 *restrict r0, bfloat16 *restrict r1, bfloat16 *restrict r2, bfloat16 *restrict taps,
                 bfloat16 *restrict out) {
  event0();
  bfloat16 *rows[3] = {r0, r1, r2};
  dwconv_row(rows, taps, out);
  event1();
}
#elif DW_K == 5
void dwconv5_row(bfloat16 *restrict r0, bfloat16 *restrict r1, bfloat16 *restrict r2, bfloat16 *restrict r3,
                 bfloat16 *restrict r4, bfloat16 *restrict taps, bfloat16 *restrict out) {
  event0();
  bfloat16 *rows[5] = {r0, r1, r2, r3, r4};
  dwconv_row(rows, taps, out);
  event1();
}
#else
#error "DW_K is 3 or 5"
#endif

} // extern "C"
