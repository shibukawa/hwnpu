//===- exp2_mid.h - the softmax's exponential, one bf16 limb a product -*- C++ -*-===//
//
// Included before IRON's linalg/mha.cc (kernels/mha_mid.cc; tools/amd/iron_mha.py, ImageMHA's exp "mid"): it
// takes exp2_bf16, the softmax's 2^x of a row's scores, over from common/exp2_bf16.h.
//
// IRON's accurate exp2_bf16 (-DEXP2_BF16_ACCURATE) evaluates a cubic to float32 accuracy, each
// product on two bf16 limbs (ten macs and six limb splits for 32 values), and then rounds the
// result to bf16. AIE2P's exp2 instruction is one operation and several times less accurate.
// Here the fraction f and each Horner step's value are one bf16 limb: four macs, an error of at
// most about 0.2 % (f's rounding 0.07 %, the steps' 0.1 %), the size of the bf16 rounding the
// result gets anyway. An image DiT's attention (4608 x 4608 scores a head) spent a fifth of its
// time in the accurate exponentials: 102.8 ms a block with them, 83.4 with the instruction
// (FLUX.2 klein at 1024 x 1024, design.md 16.207).
//===----------------------------------------------------------------------===//
#ifndef HWNPU_EXP2_MID_H
#define HWNPU_EXP2_MID_H

#define exp2_bf16 exp2_bf16_exact
#include "common/exp2_bf16.h"
#undef exp2_bf16

static inline __attribute__((always_inline)) aie::vector<bfloat16, 32>
exp2_mid_32(aie::vector<float, 32> x) {
  using acc_t = aie::accum<accfloat, 32>;
  using bf_t = aie::vector<bfloat16, 32>;
  auto acc = [](float c) { return acc_t(aie::broadcast<float, 32>(c)); };
  const auto xi = x.cast_to<int32_t>();
  const auto xu = xi.cast_to<uint32_t>();
  // below 2^-125.5 the result is 0 (a masked score is the lowest bf16)
  const auto below =
      aie::gt(xu, aie::broadcast<float, 32>(-125.5f).cast_to<int32_t>().cast_to<uint32_t>());
  const auto magic = aie::broadcast<float, 32>(12582912.0f);
  const aie::vector<float, 32> xm = aie::add(x, magic);
  const bf_t one = bf16_splat(0x3f80);
  const bf_t k = acc_t(aie::sub(xm, magic)).to_vector<bfloat16>();  // round(x)
  const bf_t f = aie::msc(acc_t(x), k, one).to_vector<bfloat16>();  // x - k in [-0.5, 0.5]
  // 2^f = 1 + f (ln 2 + f (ln^2 2 / 2 + f c3)), the cubic IRON's bf16 path fits
  acc_t p = aie::mac(acc(0.2402265069f), f, aie::broadcast<bfloat16, 32>(0.0555041087f));
  p = aie::mac(acc(0.6931471805f), p.to_vector<bfloat16>(), f);
  p = aie::mac(acc(1.0f), p.to_vector<bfloat16>(), f);
  const auto y = aie::add(p.to_vector<float>().cast_to<int32_t>(),
                          aie::upshift(xm.cast_to<int32_t>(), 23));
  bf_t r = acc_t(y.cast_to<float>()).to_vector<bfloat16>();
  return aie::select(r, aie::zeros<bfloat16, 32>(), below);
}

template <unsigned N>
static inline aie::vector<bfloat16, N> exp2_bf16(aie::vector<float, N> x) {
  if constexpr (N < 32) {
    return exp2_mid_32(x.template grow<32>()).template extract<N>(0);
  } else {
    aie::vector<bfloat16, N> out;
    AIE_LOOP_UNROLL_FULL
    for (unsigned i = 0; i < N / 32; i++)
      out.insert(i, exp2_mid_32(x.template extract<32>(i)));
    return out;
  }
}

#endif
