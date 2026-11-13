//===- q4_mv.cc - y = W x for q4nx weights, one core's share ----*- C++ -*-===//
//
// The matrix-vector product of a decode step on q4nx weights (FastFlowLM's
// 4-bit format, mlir-aie's flm_gemma4/q4nx.h): 32 out-features by 256
// in-features a block, a bf16 scale and minimum per 32 in-features of an
// out-feature, w = min + scale * code. The blocks come in the order
// iron.operators.flm.DequantBFP reads (a column block of 64 out-features: for
// each 256 in-features, the block of out-features 0-31, then 32-63), so prefill
// can dequantize the same bytes for its GEMM.
//
// Q4_K is the in-features, a multiple of 512. Q_OUT is the out-features a core
// keeps (64: both halves of a column block, its blocks alternating; 32: one
// half, the memtile giving it every other block).
//===----------------------------------------------------------------------===//

#include "q4nx.h"

#ifndef Q4_K
#error "q4_mv.cc needs -DQ4_K"
#endif

constexpr int K = Q4_K;
#ifndef Q_OUT
#define Q_OUT 64
#endif
constexpr int OUT = Q_OUT;
static_assert(OUT == 32 || OUT == 64, "a core keeps one or two halves");
constexpr int KT = 256; // in-features a block
constexpr int MT = 32;  // out-features a block
static_assert(K % 512 == 0, "K must be a multiple of 512");

using block_t = q4nx_block<MT, KT>;

// q4nx to bf16 for a prompt chunk's GEMM, as q8_mv.cc's q8_dequant: out is 64
// rows of 256 bf16 (a column block's slice of 256 in-features); writes rows
// 32 h to 32 h + 31 from the block, w = min + scale * code rounded to bf16 (to
// nearest even). A strip's codes come 8 in-features of its 16 rows at a time
// ([k][m], 64 bytes of nibbles), are scaled, shifted by the minima and
// transposed to [m][k] in two halves of 4 in-features.
static inline void q4_dequant_block(const block_t *A, bf16 *rows) {
  constexpr int PR = 16;
  const aie::vector<bf16, 8 * PR> one = aie::broadcast<bf16, 8 * PR>(bf16(1.0f));
  for (int s = 0; s < MT / PR; s++) {
    const uint4 *strip = A->qs + s * KT * PR / 2;
    for (int g = 0; g < KT / 32; g++) {
      aie::vector<bf16, PR> sc = aie::load_v<PR>(A->scales + g * MT + s * PR);
      aie::vector<bf16, PR> mn = aie::load_v<PR>(A->mins + g * MT + s * PR);
      aie::vector<bf16, 2 * PR> sc2 = aie::concat(sc, sc);
      aie::vector<bf16, 4 * PR> sc4 = aie::concat(sc2, sc2);
      aie::vector<bf16, 8 * PR> sc8 = aie::concat(sc4, sc4);
      aie::vector<bf16, 2 * PR> mn2 = aie::concat(mn, mn);
      aie::vector<bf16, 4 * PR> mn4 = aie::concat(mn2, mn2);
      aie::vector<bf16, 8 * PR> mn8 = aie::concat(mn4, mn4);
      for (int k0 = 32 * g; k0 < 32 * g + 32; k0 += 8) {
        aie::vector<uint4, 8 * PR> c = aie::load_v<8 * PR>(strip + k0 * PR / 2);
        aie::accum<accfloat, 8 * PR> cf;
        cf.from_vector(aie::to_float(c, 0));
        aie::accum<accfloat, 8 * PR> w = aie::mul(cf.template to_vector<bf16>(), sc8);
        w = aie::mac(w, mn8, one);
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

extern "C" {

// Every call first sets round-to-nearest-even: a core starts in floor mode,
// where each narrowing to bf16 (q4nx_accumulate's partial sums, the group
// sums, the output) rounds toward negative infinity, a bias that adds up over
// the in-features when the inputs have a mean.

// sums[g] = x[32 g] + ... + x[32 g + 31]: each minimum becomes one MAC.
// acc (OUT floats) starts at zero.
void q4_mv_begin(const bf16 *x, bf16 *sums, float *acc) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  for (int l = 0; l < K / 32; l++) {
    sums[l] = bf16(aie::reduce_add(aie::load_v<32>(x + 32 * l)));
  }
  zero_256<float, OUT>(acc);
}

// acc[32 h .. 32 h + 31] += the block's rows times x[256 kb .. 256 kb + 255].
void q4_mv_block(const uint8_t *block, const bf16 *x, const bf16 *sums,
                 float *acc, int32_t kb, int32_t h) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  q4nx_accumulate<MT, KT>(reinterpret_cast<const block_t *>(block),
                          x + KT * kb, acc + MT * h, sums + (KT / 32) * kb);
}

// y = acc narrowed to bf16 (OUT out-features), and acc back to zero.
void q4_mv_end(float *acc, bf16 *y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  narrow_to_bf16<OUT>(y, acc);
  zero_256<float, OUT>(acc);
}

void q4_dequant(const uint8_t *block, bf16 *out, int32_t h) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  q4_dequant_block(reinterpret_cast<const block_t *>(block), out + MT * KT * h);
}

} // extern "C"

#define MV_FMT q4
#include "mv_add.h"
