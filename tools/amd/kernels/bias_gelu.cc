//===- bias_gelu.cc - GELU of x plus a bias row, gelu_even's arithmetic -*- C++ -*-===//
//
// tools/amd/iron_biasgelu.py's BiasGELU: BG_R rows of BG_C channels, y = gelu(x + b) with b the
// channels' bias, x + b rounded once to bf16 (as a separate add's output), gelu_even_32 of
// kernels/gelu_even.cc (the tanh form, its argument in float32). One operator for what was a
// RowwiseAdd and a GELU (Supertonic's ConvNeXt blocks, iron_supertonic.py).
//===----------------------------------------------------------------------===//
#include "gelu_even.cc"

#ifndef BG_C
#define BG_C 256
#endif
#ifndef BG_R
#define BG_R 16
#endif
static_assert(BG_C % 32 == 0);

extern "C" {

void bias_gelu_bf16(const bfloat16 *restrict x, const bfloat16 *restrict b, bfloat16 *restrict y) {
  event0();
  const auto saved_rounding = aie::get_rounding();
  aie::set_rounding(aie::rounding_mode::conv_even);
  const bf_t one = aie::broadcast<bfloat16, 32>(1.0f);
  AIE_LOOP_NO_UNROLL
  for (int r = 0; r < BG_R; r++) {
    AIE_PREPARE_FOR_POSTPIPELINING
    AIE_LOOP_UNROLL(2)
    for (int o = 0; o < BG_C; o += 32) {
      acc_t s = aie::mul(aie::load_v<32>(x + r * BG_C + o), one);
      s = aie::mac(s, aie::load_v<32>(b + o), one);
      aie::store_v(y + r * BG_C + o, gelu_even_32(s.to_vector<bfloat16>()));
    }
  }
  aie::set_rounding(saved_rounding);
  event1();
}

} // extern "C"
