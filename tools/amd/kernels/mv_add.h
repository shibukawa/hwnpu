//===- mv_add.h - a GEMV's output with vectors added as it leaves -*- C++ -*-===//
//
// The end of a quantized GEMV (q4_mv.cc, q6_mv.cc, q8_mv.cc) that adds Q_ADDS vectors to its
// output before narrowing it (iron_q4.py's QGEMV with adds: a projection's bias, a residual):
// one operator in place of the GEMV and an add a vector. The vectors ride on x's channel (a
// core has two input channels, the weights' and x's): x's object is x (K in-features), then each
// vector's Q_R values of the column's out-features. A core's OUT out-features of column block cb
// are at cb * 64 + h * OUT of them.
//
// Included after the kernel's K, OUT and narrow_to_bf16; defines NAME_mv_end_add.
//===----------------------------------------------------------------------===//

#ifdef Q_ADDS
#ifndef Q_R
#error "mv_add.h needs -DQ_R with -DQ_ADDS"
#endif

#define MV_ADD_CAT(a, b) a##b
#define MV_ADD_NAME(f) MV_ADD_CAT(f, _mv_end_add)

extern "C" void MV_ADD_NAME(MV_FMT)(float *acc, const bf16 *xa, int32_t cb, int32_t h, bf16 *y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const int off = cb * 64 + h * OUT;
  for (int i = 0; i < OUT; i += 16) {
    aie::vector<float, 16> a = aie::load_v<16>(acc + i);
#pragma clang loop unroll(full)
    for (int j = 0; j < Q_ADDS; j++) {
      aie::accum<accfloat, 16> t;
      t.from_vector(aie::load_v<16>(xa + K + j * Q_R + off + i));
      a = aie::add(a, t.template to_vector<float>());
    }
    aie::store_v(acc + i, a);
  }
  narrow_to_bf16<OUT>(y, acc);
  zero_256<float, OUT>(acc);
}
#endif

// The end of a gated GEMV (iron_q4.py's QGEMVGlu): the gate's and the up's accumulators of the
// same out-features (a column block of each, read in turn from the interleaved table), y =
// silu(g) u, then both back to zero. silu by exp in float32 (exp_f32): AIE2P's vtanh returns
// its argument for |x| <= 0.5.
#ifdef Q_GLU
#include "common/exp_f32.h"
#define MV_GLU_CAT(a, b) a##b
#define MV_GLU_NAME(f) MV_GLU_CAT(f, _mv_end_glu)

extern "C" void MV_GLU_NAME(MV_FMT)(float *g, float *u, bf16 *y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const auto one = aie::broadcast<float, 16>(1.0f);
  for (int i = 0; i < OUT; i += 16) {
    aie::vector<float, 16> gv = aie::load_v<16>(g + i);
    aie::accum<accfloat, 16> z;
    z.from_vector(aie::mul(gv, -1.0f).template to_vector<float>());  // aie::neg of floats does not lower
    aie::vector<float, 16> e = exp_f32<16>(z).template to_vector<float>();
    aie::vector<float, 16> s = aie::inv(aie::add(e, one));
    aie::vector<float, 16> a = aie::mul(gv, s).template to_vector<float>();
    aie::store_v(g + i, aie::mul(a, aie::load_v<16>(u + i)).template to_vector<float>());
  }
  narrow_to_bf16<OUT>(y, g);
  zero_256<float, OUT>(g);
  zero_256<float, OUT>(u);
}
#endif
