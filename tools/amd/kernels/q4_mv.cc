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
// Q4_K is the in-features, a multiple of 512.
//===----------------------------------------------------------------------===//

#include "q4nx.h"

#ifndef Q4_K
#error "q4_mv.cc needs -DQ4_K"
#endif

constexpr int K = Q4_K;
constexpr int KT = 256; // in-features a block
constexpr int MT = 32;  // out-features a block
static_assert(K % 512 == 0, "K must be a multiple of 512");

using block_t = q4nx_block<MT, KT>;

extern "C" {

// sums[g] = x[32 g] + ... + x[32 g + 31]: each minimum becomes one MAC.
// acc (64 floats, a column block) starts at zero.
void q4_mv_begin(const bf16 *x, bf16 *sums, float *acc) {
  for (int l = 0; l < K / 32; l++) {
    sums[l] = bf16(aie::reduce_add(aie::load_v<32>(x + 32 * l)));
  }
  zero_256<float, 2 * MT>(acc);
}

// acc[32 h .. 32 h + 31] += the block's rows times x[256 kb .. 256 kb + 255].
void q4_mv_block(const uint8_t *block, const bf16 *x, const bf16 *sums,
                 float *acc, int32_t kb, int32_t h) {
  q4nx_accumulate<MT, KT>(reinterpret_cast<const block_t *>(block),
                          x + KT * kb, acc + MT * h, sums + (KT / 32) * kb);
}

// y = acc narrowed to bf16 (64 out-features), and acc back to zero.
void q4_mv_end(float *acc, bf16 *y) {
  narrow_to_bf16<2 * MT>(y, acc);
  zero_256<float, 2 * MT>(acc);
}

} // extern "C"
