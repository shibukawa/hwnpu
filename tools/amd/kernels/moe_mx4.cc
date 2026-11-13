//===- moe_mx4.cc - gpt-oss's experts for one token, on mx4 weights -*- C++ -*-===//
//
// The experts of a gpt-oss decode step (tools/amd/iron_gptoss_moe.py) on mx4, the
// checkpoint's MXFP4 kept exactly: a block holds 32 out-features by MX_KT in-features,
//
//   exps[MX_KT / 32][32]   uint8, each group's MXFP4 exponent E: the scale 2^(E - 128)
//   codes[32 / 16][MX_KT][16]  e2m1 nibbles (q4nx's strips: 16 out-features each,
//                              column-major, element 2 i in the low half of byte i)
//   bias[32]               bf16, the out-features' bias (the first k-block's; zero after)
//
// so w = scale * v, v = 2 * e2m1 = sign * max(m, 2 m - 4, 4 m - 16) for the magnitude
// code m (0 1 2 3 4 6 8 12): exact in bf16 and in the products. 4.3 bits a weight.
//
//   gate and up: acc = bias, acc += W x over the k-blocks; y = (clamp(u, -7, 7) + 1) *
//     g' sigmoid(1.702 g'), g' = min(g, 7) (mx4_glu)
//   down: per expert acc = bias + W a_e, then tot += w_e acc (mx4_fold); y = tot + r
//     (mx4_end_add), r the residual after the experts' activations in x's object
//
// The partial sums of 32 in-features are split into two bf16 parts before the scale (the
// codes reach 12, q6_mv.cc's reason). sigmoid is 1 / (1 + exp(-z)) in float32 (exp_f32):
// AIE2P's vtanh returns its argument for |x| <= 0.5.
//===----------------------------------------------------------------------===//

#include "utils.h"
#include "common/exp_f32.h"

#include <stdint.h>

#ifndef MX_K
#error "moe_mx4.cc needs -DMX_K"
#endif
#ifndef MX_KT
#define MX_KT 320
#endif
#ifndef MX_LIMIT
#define MX_LIMIT 7.0f
#endif

constexpr int K = MX_K;
constexpr int KT = MX_KT;
constexpr int MT = 32;
constexpr int PR = 16;
static_assert(KT % 32 == 0 && K % KT == 0, "K whole k-blocks of whole groups");

struct mx4_blk_t {
  uint8_t exps[MT * KT / 32]; // each group's MXFP4 exponent (exp_scales)
  uint8_t codes[MT * KT / 2];
  bf16 bias[MT];
};

// The 128 codes of unit u (8 in-features of a strip's 16 out-features) as v, -12..12.
// A group's scale 2^(E - 128) from its MXFP4 exponent byte E (the block's exps): bf16's exponent
// field E - 1, no mantissa. 16 at a time (a strip's out-features of a group).
static inline aie::vector<bf16, 16> exp_scales(const uint8_t *p) {
  aie::vector<uint8_t, 16> e = aie::load_v<16>(p);
  aie::vector<uint16_t, 16> w = aie::unpack(e);
  w = aie::upshift(aie::sub(w, uint16_t(1)), 7);
  return aie::vector_cast<bf16>(w);
}

static inline aie::vector<int8_t, 8 * PR> mx4_codes(const mx4_blk_t *A, int u) {
  aie::vector<uint4, 8 * PR> c4 =
      aie::load_v<8 * PR>(reinterpret_cast<const uint4 *>(A->codes + u * 4 * PR));
  aie::vector<uint8_t, 8 * PR> c = aie::unpack(c4);
  aie::vector<int8_t, 8 * PR> m = aie::vector_cast<int8_t>(aie::bit_and(uint8_t(7), c));
  aie::vector<int8_t, 8 * PR> m2 = aie::sub(aie::upshift(m, 1), int8_t(4));
  aie::vector<int8_t, 8 * PR> m4 = aie::sub(aie::upshift(m, 2), int8_t(16));
  aie::vector<int8_t, 8 * PR> v = aie::max(m, aie::max(m2, m4));
  aie::mask<8 * PR> neg = aie::ge(c, uint8_t(8));
  return aie::select(v, aie::neg(v), neg);
}

// c[0:MT] += dequantize(A) * B[0:KT].
static inline void mx4_accumulate(const mx4_blk_t *A, const bf16 *B, float *c) {
#ifdef MX_NOCOMPUTE
  return; // a measurement: the transfers alone
#endif
  int u = 0;
  AIE_LOOP_RANGE(MT / PR, MT / PR)
  for (int row = 0; row < MT; row += PR) {
    aie::accum<accfloat, PR> c_accum;
    c_accum.from_vector(aie::load_v<PR>(c + row));
    const bf16 *it_B = B;
    uint32_t scale_offset = row;
    AIE_LOOP_RANGE(KT / 32, KT / 32)
    for (int sub = 0; sub < KT / 32; sub++) {
      aie::vector<bf16, 32> b_col = aie::load_v<32>(it_B);
      it_B += 32;
      aie::vector<bf16, PR> a_scales = exp_scales(A->exps + scale_offset);
      scale_offset += MT;
      aie::vector<bf16, PR * 8> a_bf16[4];
#pragma clang loop unroll(full)
      for (int sv = 0; sv < 4; sv++) {
        aie::vector<int8_t, PR * 8> a = mx4_codes(A, u + sv);
        aie::accum<accfloat, PR * 8> acc;
        acc.from_vector(aie::to_float(a, 0));
        a_bf16[sv] = acc.template to_vector<bf16>();
      }
      u += 4;
      aie::accum<accfloat, PR> t = aie::mul(a_bf16[0].extract<PR>(0), b_col.get(0));
#pragma clang loop unroll(full)
      for (int sv = 0; sv < 4; sv++) {
#pragma clang loop unroll(full)
        for (int cc = 0; cc < 8; cc++) {
          if (sv == 0 && cc == 0)
            continue;
          t = aie::mac(t, a_bf16[sv].extract<PR>(cc), b_col.get(sv * 8 + cc));
        }
      }
      aie::vector<bf16, PR> hi = t.template to_vector<bf16>();
      aie::vector<bf16, PR> lo = aie::mac(t, hi, bf16(-1.0f)).template to_vector<bf16>();
      c_accum = aie::mac(c_accum, hi, a_scales);
      c_accum = aie::mac(c_accum, lo, a_scales);
    }
    aie::store_v(c + row, c_accum.template to_vector<float>());
  }
}

extern "C" {

// acc = the block's bias (its first k-block's), as floats.
void mx4_init(const uint8_t *block, float *acc) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const mx4_blk_t *A = reinterpret_cast<const mx4_blk_t *>(block);
  for (int i = 0; i < MT; i += 16) {
    aie::accum<accfloat, 16> b;
    b.from_vector(aie::load_v<16>(A->bias + i));
    aie::store_v(acc + i, b.template to_vector<float>());
  }
}

// acc += the block's 32 rows times x[KT kb .. KT kb + KT - 1].
void mx4_block(const uint8_t *block, const bf16 *x, float *acc, int32_t kb) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  mx4_accumulate(reinterpret_cast<const mx4_blk_t *>(block), x + KT * kb, acc);
}

// The same, x at expert e's activations (MX_XE apart: a gate-up output per expert).
#ifndef MX_XE
#define MX_XE K
#endif
void mx4_block_e(const uint8_t *block, const bf16 *x, float *acc, int32_t e, int32_t kb) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  mx4_accumulate(reinterpret_cast<const mx4_blk_t *>(block), x + MX_XE * e + KT * kb, acc);
}

// y = (clamp(u, -L, L) + 1) * g' / (1 + exp(-1.702 g')), g' = min(g, L).
void mx4_glu(const float *g, const float *u, bf16 *y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  alignas(64) float out[MT];
  const auto lim = aie::broadcast<float, 16>(MX_LIMIT);
  const auto nlim = aie::broadcast<float, 16>(-MX_LIMIT);
  const auto one = aie::broadcast<float, 16>(1.0f);
  for (int i = 0; i < MT; i += 16) {
    aie::vector<float, 16> gv = aie::min(aie::load_v<16>(g + i), lim);
    aie::vector<float, 16> uv = aie::max(aie::min(aie::load_v<16>(u + i), lim), nlim);
    aie::accum<accfloat, 16> z;
    z.from_vector(aie::mul(gv, -1.702f).template to_vector<float>());
    aie::vector<float, 16> e = exp_f32<16>(z).template to_vector<float>();
    aie::vector<float, 16> s = aie::inv(aie::add(e, one));
    aie::vector<float, 16> p = aie::mul(gv, s).template to_vector<float>();
    aie::store_v(out + i, aie::mul(p, aie::add(uv, one)).template to_vector<float>());
  }
  narrow_to_bf16<MT>(y, out);
}

void mx4_zero(float *acc) { zero_256<float, MT>(acc); }

// tot += w[e] acc, w the routing weights at MX_W past x's start.
#ifndef MX_W
#define MX_W 0
#endif
void mx4_fold(const float *acc, float *tot, const bf16 *x, int32_t e) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float w = float(x[MX_W + e]);
  for (int i = 0; i < MT; i += 16) {
    aie::vector<float, 16> a = aie::load_v<16>(acc + i);
    aie::vector<float, 16> t = aie::load_v<16>(tot + i);
    aie::store_v(tot + i, aie::add(t, aie::mul(a, w).template to_vector<float>()));
  }
}

// y = tot + r, r the residual's out-features at off past x's start, then tot back to zero.
void mx4_end_add(float *tot, const bf16 *x, int32_t off, bf16 *y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const bf16 *r = x + off;
  for (int i = 0; i < MT; i += 16) {
    aie::vector<float, 16> a = aie::load_v<16>(tot + i);
    aie::accum<accfloat, 16> t;
    t.from_vector(aie::load_v<16>(r + i));
    aie::store_v(tot + i, aie::add(a, t.template to_vector<float>()));
  }
  narrow_to_bf16<MT>(y, tot);
  zero_256<float, MT>(tot);
}

} // extern "C"
