//===- q8_mv.cc - y = W x for q8 weights, and q8 to bf16 --------*- C++ -*-===//
//
// q8 is hwnpu's 8-bit block (tools/amd/iron_q4.py): 32 out-features by 256
// in-features, a bf16 scale per 32 in-features of an out-feature (indexed
// [k / 32, m], as q4nx's), then int8 codes in strips of 16 out-features, each
// strip column-major ([m / 16, k, m % 16], as q4nx's nibbles). w = scale * code.
// 8.5 bits a weight. The blocks come in q4nx's order (column blocks of 64
// out-features, the two halves alternating).
//
// Q8_K is the in-features, a multiple of 512; Q_OUT the out-features a core
// keeps (64 or 32, as q4_mv.cc's).
//===----------------------------------------------------------------------===//

#include "utils.h"

#include <stdint.h>

#ifndef Q8_K
#error "q8_mv.cc needs -DQ8_K"
#endif
#ifndef Q_OUT
#define Q_OUT 64
#endif

constexpr int K = Q8_K;
constexpr int OUT = Q_OUT;
constexpr int KT = 256;
constexpr int MT = 32;
constexpr int PR = 16; // out-features a strip
static_assert(K % 512 == 0, "K must be a multiple of 512");
static_assert(OUT == 32 || OUT == 64, "a core keeps one or two halves");

struct q8_block {
  bf16 scales[MT * KT / 32];
  int8_t qs[MT * KT];
};

// c[0:MT] += dequantize(A) * B[0:KT], as q4nx_accumulate without the minima:
// each 32 columns' codes times B summed (exact in the accumulator: an int8 is
// exact in bf16), times the scales.
static inline void q8_accumulate(const q8_block *A, const bf16 *B, float *c) {
  const int8_t *qs_ptr = A->qs;
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
        aie::vector<int8_t, PR * 8> a = aie::load_v<PR * 8>(qs_ptr);
        qs_ptr += PR * 8;
        aie::accum<accfloat, PR * 8> acc;
        acc.from_vector(aie::to_float(a, 0));
        a_bf16[sv] = acc.template to_vector<bf16>();
      }
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
      // t in two bf16 parts (high, then what high leaves), each times the
      // scales: narrowing t to one bf16, as q4nx_accumulate does, would round
      // away what 8 bits keep (cosine 0.9998 against the exact product)
      aie::vector<bf16, PR> hi = t.template to_vector<bf16>();
      aie::vector<bf16, PR> lo =
          aie::mac(t, hi, bf16(-1.0f)).template to_vector<bf16>();
      c_accum = aie::mac(c_accum, hi, a_scales);
      c_accum = aie::mac(c_accum, lo, a_scales);
    }
    aie::store_v(c + row, c_accum.template to_vector<float>());
  }
}

// The 128 codes of unit u, for mv_rows.h.
static inline aie::vector<int8_t, 8 * PR> q8_codes(const q8_block *A, int u) {
  return aie::load_v<8 * PR>(A->qs + u * 8 * PR);
}

#include "mv_rows.h"

extern "C" {

// Every call first sets round-to-nearest-even: a core starts in floor mode,
// where each narrowing to bf16 rounds toward negative infinity. That bias adds
// up over the in-features when the inputs have a mean (a DiT's do): 4 to 10
// times the error of the bf16 weights.

// The GEMV's (q4_mv.cc's calls; q8 has no minima, so no sums).
void q8_mv_begin(const bf16 *x, bf16 *sums, float *acc) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  (void)x;
  (void)sums;
  zero_256<float, OUT>(acc);
}

void q8_mv_block(const uint8_t *block, const bf16 *x, const bf16 *sums,
                 float *acc, int32_t kb, int32_t h) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  (void)sums;
  q8_accumulate(reinterpret_cast<const q8_block *>(block), x + KT * kb,
                acc + MT * h);
}

void q8_mv_end(float *acc, bf16 *y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  narrow_to_bf16<OUT>(y, acc);
  zero_256<float, OUT>(acc);
}

// out: 64 rows of 256 bf16 (a column block's slice of 256 in-features); writes
// rows 32 h to 32 h + 31 from the block, w = scale * code rounded to bf16. A
// strip's codes come 8 in-features of its 16 rows at a time ([k][m]), are
// scaled by the 16 rows' scales and transposed to [m][k] in two halves of 4
// in-features, 8 values a row.
void q8_dequant(const uint8_t *block, bf16 *out, int32_t h) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const q8_block *A = reinterpret_cast<const q8_block *>(block);
  bf16 *rows = out + MT * KT * h;
  for (int s = 0; s < MT / PR; s++) {
    const int8_t *strip = A->qs + s * KT * PR;
    for (int g = 0; g < KT / 32; g++) {
      aie::vector<bf16, PR> sc = aie::load_v<PR>(A->scales + g * MT + s * PR);
      aie::vector<bf16, 2 * PR> sc2 = aie::concat(sc, sc);
      aie::vector<bf16, 4 * PR> sc4 = aie::concat(sc2, sc2);
      aie::vector<bf16, 8 * PR> sc8 = aie::concat(sc4, sc4);
      for (int k0 = 32 * g; k0 < 32 * g + 32; k0 += 8) {
        aie::vector<int8_t, 8 * PR> c = aie::load_v<8 * PR>(strip + k0 * PR);
        aie::accum<accfloat, 8 * PR> cf;
        cf.from_vector(aie::to_float(c, 0));
        aie::accum<accfloat, 8 * PR> w = aie::mul(cf.template to_vector<bf16>(), sc8);
        // a 16-bit transpose takes at most 64 values: in-features 0-3 and 4-7
        aie::vector<bf16, 8 * PR> wb = w.template to_vector<bf16>();
        aie::vector<bf16, 4 * PR> t0 = aie::transpose(wb.template extract<4 * PR>(0), 4, PR);
        aie::vector<bf16, 4 * PR> t1 = aie::transpose(wb.template extract<4 * PR>(1), 4, PR);
        // rows [m][k 0-3] and [m][k 4-7] zipped 4 by 4: [m][k 0-7], rows 0-7 then 8-15
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

// The GEMV of ROWS rows (mv_rows.h).
MV_ROWS_CALLS(q8, q8_block, q8_codes)

} // extern "C"

#define MV_FMT q8
#include "mv_add.h"
