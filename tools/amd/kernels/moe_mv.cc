//===- moe_mv.cc - a mixture of experts' decode GEMVs on q4nx ----*- C++ -*-===//
//
// The GEMVs of a mixture-of-experts layer for one token on q4nx weights
// (q4_mv.cc's blocks and q4nx_accumulate), with the element-wise work around
// them done as the outputs leave the core (tools/amd/iron_moeprobe.py):
//
//   gate and up: y = silu(gate x) * (up x) * w[expert], the gate's and the up's
//     column blocks alternating in the table, so a core holds both accumulators
//     of its 32 out-features before it writes them (moe_glu)
//   down: y = r + sum_e down_e x_e, the residual added as it is written
//     (moe_end_add)
//
// A core keeps 32 out-features (two cores a column, the memtile splitting each
// pair of blocks). A core has two input channels, the weights' and x's, so x's
// object carries what else the core reads after its MOE_K in-features (the
// routing weights; the residual of the column's out-features), joined in the
// memtile. MOE_K is the in-features, a multiple of 512;
// MOE_PAIRS the 256-wide k-blocks of one expert's down (its in-features / 256).
//===----------------------------------------------------------------------===//

#include "q4nx.h"
#include "common/activations.h"

#ifndef MOE_K
#error "moe_mv.cc needs -DMOE_K"
#endif
#ifndef MOE_PAIRS
#define MOE_PAIRS 2
#endif

constexpr int K = MOE_K;
constexpr int KT = 256; // in-features a block
constexpr int MT = 32;  // out-features a block, and a core's
static_assert(K % 512 == 0, "K must be a multiple of 512");

using block_t = q4nx_block<MT, KT>;

extern "C" {

// sums[g] = x[32 g] + ... + x[32 g + 31]: each minimum becomes one MAC.
void moe_sums(const bf16 *x, bf16 *sums) {
  for (int l = 0; l < K / 32; l++) {
    sums[l] = bf16(aie::reduce_add(aie::load_v<32>(x + 32 * l)));
  }
}

void moe_zero(float *acc) { zero_256<float, MT>(acc); }

// acc += the block's 32 rows times x[256 kb .. 256 kb + 255].
void moe_block(const uint8_t *block, const bf16 *x, const bf16 *sums,
               float *acc, int32_t kb) {
  q4nx_accumulate<MT, KT>(reinterpret_cast<const block_t *>(block),
                          x + KT * kb, acc, sums + (KT / 32) * kb);
}

// The same for expert e's k-block kh of a down whose experts sit side by side.
void moe_block_e(const uint8_t *block, const bf16 *x, const bf16 *sums,
                 float *acc, int32_t e, int32_t kh) {
  moe_block(block, x, sums, acc, e * MOE_PAIRS + kh);
}

// y = silu(g) * u * w[idx], w the routing weights after x, then g and u back
// to zero.
void moe_glu(float *g, float *u, const bf16 *xw, int32_t idx, bf16 *y) {
  const float s = float(xw[K + idx]);
  for (int i = 0; i < MT; i += 16) {
    aie::vector<float, 16> gv = aie::load_v<16>(g + i);
    aie::vector<float, 16> uv = aie::load_v<16>(u + i);
    aie::vector<float, 16> a = silu_vec<16>(gv);
    aie::vector<float, 16> p = aie::mul(a, uv).template to_vector<float>();
    p = aie::mul(p, s).template to_vector<float>();
    aie::store_v(g + i, p);
  }
  narrow_to_bf16<MT>(y, g);
  zero_256<float, MT>(g);
  zero_256<float, MT>(u);
}

// y = acc + r, r the residual's out-features at off past x, then acc back to
// zero.
void moe_end_add(float *acc, const bf16 *xr, int32_t off, bf16 *y) {
  const bf16 *r = xr + K + off;
  for (int i = 0; i < MT; i += 16) {
    aie::vector<float, 16> a = aie::load_v<16>(acc + i);
    aie::accum<accfloat, 16> t;
    t.from_vector(aie::load_v<16>(r + i));
    aie::vector<float, 16> rv = t.template to_vector<float>();
    aie::store_v(acc + i, aie::add(a, rv));
  }
  narrow_to_bf16<MT>(y, acc);
  zero_256<float, MT>(acc);
}

} // extern "C"
