//===- mv_rows.h - the GEMV of a few rows, for q8 and q6 ----------*- C++ -*-===//
//
// y = W x for ROWS rows of x at once (a call checking drafts: the next token
// and the drafts after it). A block's codes are expanded to bf16 once and
// multiply every row; each row's sums and roundings are those of the one-row
// GEMV (q8_mv.cc's q8_accumulate) on it alone.
//
// Included by q8_mv.cc and q6_mv.cc after their block and their codes(A, u),
// the 128 codes of unit u (8 in-features of a strip's 16 out-features, the
// units in the order the one-row GEMV reads them). x comes 256 in-features of
// every row at a time (the rows KT apart), a core's accumulators 64 floats a
// row (ROWS * 64), y's rows OUT apart.
//===----------------------------------------------------------------------===//

#ifndef Q_ROWS
#define Q_ROWS 1
#endif
constexpr int ROWS = Q_ROWS;
constexpr int ACC_ROW = 2 * MT;

template <typename Block, typename Codes>
static inline void accumulate_rows(const Block *A, const bf16 *B, float *c, Codes codes) {
  int u = 0;
  AIE_LOOP_RANGE(MT / PR, MT / PR)
  for (int row = 0; row < MT; row += PR) {
    uint32_t scale_offset = row;
    aie::accum<accfloat, PR> c_accum[ROWS];
#pragma clang loop unroll(full)
    for (int r = 0; r < ROWS; r++)
      c_accum[r].from_vector(aie::load_v<PR>(c + r * ACC_ROW + row));
    AIE_LOOP_RANGE(KT / 32, KT / 32)
    for (int sub = 0; sub < KT / 32; sub++) {
      aie::vector<bf16, PR> a_scales = aie::load_v<PR>(A->scales + scale_offset);
      scale_offset += MT;
      aie::vector<bf16, PR * 8> a_bf16[4];
#pragma clang loop unroll(full)
      for (int sv = 0; sv < 4; sv++) {
        aie::vector<int8_t, PR * 8> a = codes(A, u + sv);
        aie::accum<accfloat, PR * 8> acc;
        acc.from_vector(aie::to_float(a, 0));
        a_bf16[sv] = acc.template to_vector<bf16>();
      }
      u += 4;
#pragma clang loop unroll(full)
      for (int r = 0; r < ROWS; r++) {
        aie::vector<bf16, 32> b_col = aie::load_v<32>(B + r * KT + 32 * sub);
        aie::accum<accfloat, PR> t = aie::mul(a_bf16[0].template extract<PR>(0), b_col.get(0));
#pragma clang loop unroll(full)
        for (int sv = 0; sv < 4; sv++) {
#pragma clang loop unroll(full)
          for (int cc = 0; cc < 8; cc++) {
            if (sv == 0 && cc == 0)
              continue;
            t = aie::mac(t, a_bf16[sv].template extract<PR>(cc), b_col.get(sv * 8 + cc));
          }
        }
        // the partial sum in two bf16 parts, as the one-row GEMV
        aie::vector<bf16, PR> hi = t.template to_vector<bf16>();
        aie::vector<bf16, PR> lo = aie::mac(t, hi, bf16(-1.0f)).template to_vector<bf16>();
        c_accum[r] = aie::mac(c_accum[r], hi, a_scales);
        c_accum[r] = aie::mac(c_accum[r], lo, a_scales);
      }
    }
#pragma clang loop unroll(full)
    for (int r = 0; r < ROWS; r++)
      aie::store_v(c + r * ACC_ROW + row, c_accum[r].template to_vector<float>());
  }
}

// The calls (begin, block, end) of a format's GEMV of ROWS rows, as its one-row ones.
#define MV_ROWS_CALLS(fmt, Block, codes)                                            \
  void fmt##_mvr_begin(const bf16 *x, bf16 *sums, float *acc) {                     \
    aie::set_rounding(aie::rounding_mode::conv_even);                               \
    (void)x;                                                                        \
    (void)sums;                                                                     \
    for (int r = 0; r < ROWS; r++)                                                  \
      zero_256<float, OUT>(acc + r * ACC_ROW);                                      \
  }                                                                                 \
  void fmt##_mvr_block(const uint8_t *block, const bf16 *x, const bf16 *sums,       \
                       float *acc, int32_t kb, int32_t h) {                         \
    aie::set_rounding(aie::rounding_mode::conv_even);                               \
    (void)sums;                                                                     \
    if (kb == 0 && h == 0) /* a column block's first: no begin call a block */      \
      for (int r = 0; r < ROWS; r++)                                                \
        zero_256<float, 2 * MT>(acc + r * ACC_ROW);                                 \
    accumulate_rows(reinterpret_cast<const Block *>(block), x, acc + MT * h, codes); \
  }                                                                                 \
  void fmt##_mvr_end(float *acc, bf16 *y) {                                         \
    aie::set_rounding(aie::rounding_mode::conv_even);                               \
    for (int r = 0; r < ROWS; r++) {                                                \
      narrow_to_bf16<OUT>(y + r * OUT, acc + r * ACC_ROW);                          \
      zero_256<float, OUT>(acc + r * ACC_ROW);                                      \
    }                                                                               \
  }
