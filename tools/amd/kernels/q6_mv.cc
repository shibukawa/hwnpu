//===- q6_mv.cc - y = W x for q6 weights, and q6 to bf16 --------*- C++ -*-===//
//
// q6 is hwnpu's 6-bit block (tools/amd/iron_q4.py), q8's with 6-bit codes: 32
// out-features by 256 in-features, a bf16 scale per 32 in-features of an
// out-feature (indexed [k / 32, m]), then the codes in q8's order (strips of
// 16 out-features, each column-major: [m / 16, k, m % 16]) stored as
// u = code + 32 (code -31..31) in two planes: the low 4 bits as q4nx's
// nibbles (element 2i in the low half of byte i), then the high 2 bits, 32
// bytes for each 128 codes, code 32 j + i of the 128 in bits 2 j, 2 j + 1 of
// byte i. w = scale * code. 6.5 bits a weight. The blocks come in q4nx's
// order (column blocks of 64 out-features, the two halves alternating).
//
// Q6_K is the in-features, a multiple of 512; Q_OUT the out-features a core
// keeps (64 or 32, as q4_mv.cc's).
//===----------------------------------------------------------------------===//

#include "utils.h"

#include <stdint.h>

#ifndef Q6_K
#error "q6_mv.cc needs -DQ6_K"
#endif
#ifndef Q_OUT
#define Q_OUT 64
#endif

constexpr int K = Q6_K;
constexpr int OUT = Q_OUT;
constexpr int KT = 256;
constexpr int MT = 32;
constexpr int PR = 16; // out-features a strip
static_assert(K % 512 == 0, "K must be a multiple of 512");
static_assert(OUT == 32 || OUT == 64, "a core keeps one or two halves");

struct q6_block {
  bf16 scales[MT * KT / 32];
  uint8_t lo[MT * KT / 2];
  uint8_t hi[MT * KT / 4];
};

// The 128 codes of unit u (8 in-features of a strip's 16 out-features), -31..31.
static inline aie::vector<int8_t, 8 * PR> q6_codes(const q6_block *A, int u) {
  aie::vector<uint4, 8 * PR> l4 =
      aie::load_v<8 * PR>(reinterpret_cast<const uint4 *>(A->lo + u * 4 * PR));
  aie::vector<uint8_t, 8 * PR> lo = aie::unpack(l4);
  aie::vector<uint8_t, 2 * PR> hb = aie::load_v<2 * PR>(A->hi + u * 2 * PR);
  aie::vector<uint8_t, 2 * PR> h0 = aie::bit_and(uint8_t(3), hb);
  aie::vector<uint8_t, 2 * PR> h1 = aie::bit_and(uint8_t(3), aie::downshift(hb, 2));
  aie::vector<uint8_t, 2 * PR> h2 = aie::bit_and(uint8_t(3), aie::downshift(hb, 4));
  aie::vector<uint8_t, 2 * PR> h3 = aie::downshift(hb, 6);
  aie::vector<uint8_t, 8 * PR> hi = aie::concat(h0, h1, h2, h3);
  aie::vector<uint8_t, 8 * PR> u8 = aie::bit_or(lo, aie::upshift(hi, 4));
  return aie::sub(aie::vector_cast<int8_t>(u8), int8_t(32));
}

// c[0:MT] += dequantize(A) * B[0:KT], as q8_mv.cc's q8_accumulate.
static inline void q6_accumulate(const q6_block *A, const bf16 *B, float *c) {
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
      aie::vector<bf16, PR> a_scales = aie::load_v<PR>(A->scales + scale_offset);
      scale_offset += MT;
      aie::vector<bf16, PR * 8> a_bf16[4];
#pragma clang loop unroll(full)
      for (int sv = 0; sv < 4; sv++) {
        aie::vector<int8_t, PR * 8> a = q6_codes(A, u + sv);
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
      // the partial sum in two bf16 parts, as q8_accumulate (6 bits of code
      // over 32 in-features also pass bf16's 8)
      aie::vector<bf16, PR> hi = t.template to_vector<bf16>();
      aie::vector<bf16, PR> lo =
          aie::mac(t, hi, bf16(-1.0f)).template to_vector<bf16>();
      c_accum = aie::mac(c_accum, hi, a_scales);
      c_accum = aie::mac(c_accum, lo, a_scales);
    }
    aie::store_v(c + row, c_accum.template to_vector<float>());
  }
}

#include "mv_rows.h"

extern "C" {

// The GEMV of ROWS rows (mv_rows.h).
MV_ROWS_CALLS(q6, q6_block, q6_codes)

// Every call first sets round-to-nearest-even (q8_mv.cc).

void q6_mv_begin(const bf16 *x, bf16 *sums, float *acc) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  (void)x;
  (void)sums;
  zero_256<float, OUT>(acc);
}

void q6_mv_block(const uint8_t *block, const bf16 *x, const bf16 *sums,
                 float *acc, int32_t kb, int32_t h) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  (void)sums;
  q6_accumulate(reinterpret_cast<const q6_block *>(block), x + KT * kb,
                acc + MT * h);
}

void q6_mv_end(float *acc, bf16 *y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  narrow_to_bf16<OUT>(y, acc);
  zero_256<float, OUT>(acc);
}

// As q8_dequant: rows 32 h to 32 h + 31 of out (64 rows of 256 bf16) from the
// block, w = scale * code rounded to bf16.
void q6_dequant(const uint8_t *block, bf16 *out, int32_t h) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const q6_block *A = reinterpret_cast<const q6_block *>(block);
  bf16 *rows = out + MT * KT * h;
  for (int s = 0; s < MT / PR; s++) {
    for (int g = 0; g < KT / 32; g++) {
      aie::vector<bf16, PR> sc = aie::load_v<PR>(A->scales + g * MT + s * PR);
      aie::vector<bf16, 2 * PR> sc2 = aie::concat(sc, sc);
      aie::vector<bf16, 4 * PR> sc4 = aie::concat(sc2, sc2);
      aie::vector<bf16, 8 * PR> sc8 = aie::concat(sc4, sc4);
      for (int k0 = 32 * g; k0 < 32 * g + 32; k0 += 8) {
        aie::vector<int8_t, 8 * PR> c = q6_codes(A, s * (KT / 8) + k0 / 8);
        aie::accum<accfloat, 8 * PR> cf;
        cf.from_vector(aie::to_float(c, 0));
        aie::accum<accfloat, 8 * PR> w = aie::mul(cf.template to_vector<bf16>(), sc8);
        aie::vector<bf16, 8 * PR> wb = w.template to_vector<bf16>();
        aie::vector<bf16, 4 * PR> t0 = aie::transpose(wb.template extract<4 * PR>(0), 4, PR);
        aie::vector<bf16, 4 * PR> t1 = aie::transpose(wb.template extract<4 * PR>(1), 4, PR);
        auto [r0, r1] = aie::interleave_zip(t0, t1, 4);
#pragma clang loop unroll(full)
        for (int p = 0; p < PR / 2; p++) {
          aie::store_v(rows + (s * PR + p) * KT + k0, r0.template extract<8>(p));
          aie::store_v(rows + (s * PR + PR / 2 + p) * KT + k0, r1.template extract<8>(p));
        }
      }
    }
  }
}

} // extern "C"

#define MV_FMT q6
#include "mv_add.h"
