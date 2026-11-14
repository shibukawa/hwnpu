//===- head_rms.cc - RMS norm of each head of a row -------------*- C++ -*-===//
//
// q and k of an image model's attention are RMS-normed per head (Anima, Chroma, Qwen-Image:
// heads of 128). As rows of 128 (IRON's RMSNorm over (rows x heads, 128)) a 4096 x 2048 q took
// 1.4 ms, 2.5 times a LayerNorm of the same bytes in rows of 2048: a kernel call and a fifo
// object a head. Here a call takes a whole row (HR_HEADS heads of HR_D) and normalizes each head
// with its own sum of squares, as rms_norm_aie2p.h does a row: the output x (s_hi + s_lo),
// rounded to bf16 once, nearest even. tools/amd/iron_headnorm.py, design.md 16.200.
//===----------------------------------------------------------------------===//
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "aie_kernel_utils.h"
#include "common/scalar_f32.h"

#ifndef HR_D
#define HR_D 128
#endif

constexpr unsigned kLanes = 64;
using bf16xN = aie::vector<bfloat16, kLanes>;
using f32_acc = aie::accum<accfloat, kLanes>;
static_assert(HR_D % kLanes == 0);

extern "C" {

void head_rms_norm_eps(bfloat16 *restrict input, bfloat16 *restrict output, int32_t cols, float epsilon) {
  event0();
  const auto saved_rounding = aie::get_rounding();
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float inv_d = 1.0f / HR_D;
  const unsigned heads = (uint32_t)cols / HR_D;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (unsigned h = 0; h < heads; h++) {
    f32_acc acc = aie::zeros<accfloat, kLanes>();
    bf16xN x[HR_D / kLanes];
    AIE_LOOP_UNROLL_FULL
    for (unsigned i = 0; i < HR_D / kLanes; i++) {
      x[i] = aie::load_v<kLanes>(input + i * kLanes);
      acc = aie::mac_square(acc, x[i]);
    }
    const float sum_sq = aie::reduce_add(acc.to_vector<float>());
    const float inv_rms = scalar_invsqrt(scalar_mul(sum_sq, inv_d) + epsilon);
    f32_acc a;
    a.from_vector(aie::broadcast<float, kLanes>(inv_rms));
    const bf16xN s_hi = a.to_vector<bfloat16>();
    const bf16xN s_lo = aie::msc(a, s_hi, aie::broadcast<bfloat16, kLanes>(1.0f)).to_vector<bfloat16>();
    AIE_LOOP_UNROLL_FULL
    for (unsigned i = 0; i < HR_D / kLanes; i++)
      aie::store_v(output + i * kLanes, aie::mac(aie::mul(x[i], s_lo), x[i], s_hi).to_vector<bfloat16>());
    input += HR_D;
    output += HR_D;
  }
  aie::set_rounding(saved_rounding);
  event1();
}

} // extern "C"
