//===- q8x_dequant.cc - q8x blocks to bf16 ----------------------*- C++ -*-===//
//
// A q8x block (tools/amd/iron_gptoss_layer.py's q8x_table, the decode step's
// table of a projection): 32 out-features by 160 in-features, a bf16 scale per
// 32 in-features of an out-feature ([group][out-feature]), int8 codes in strips
// of 16 out-features, each strip column-major, then the block's bias.
// w = scale * code, rounded to bf16 (to nearest even): what a prompt chunk's
// GEMM multiplies by, as q8_mv.cc's q8_dequant gives it from q8 blocks.
//===----------------------------------------------------------------------===//

#include "utils.h"

#include <stdint.h>

constexpr int KT = 160;
constexpr int MT = 32;
constexpr int PR = 16; // out-features a strip

struct q8x_block {
  bf16 scales[KT / 32 * MT];
  int8_t qs[MT * KT];
  bf16 bias[MT];
};

extern "C" {

// out: the block's 32 rows of its first width in-features (width a multiple of
// 32: 160, or what is left of the weight's in-features in its last k-block),
// row after row. A strip's codes come 8 in-features of its 16 rows at a time,
// are scaled by the rows' scales and transposed to [row][in-feature] in two
// halves of 4 in-features (q8_dequant's steps).
void q8x_dequant(const uint8_t *block, bf16 *out, int32_t width) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const q8x_block *A = reinterpret_cast<const q8x_block *>(block);
  for (int s = 0; s < MT / PR; s++) {
    const int8_t *strip = A->qs + s * KT * PR;
    for (int g = 0; g < width / 32; g++) {
      aie::vector<bf16, PR> sc = aie::load_v<PR>(A->scales + g * MT + s * PR);
      aie::vector<bf16, 2 * PR> sc2 = aie::concat(sc, sc);
      aie::vector<bf16, 4 * PR> sc4 = aie::concat(sc2, sc2);
      aie::vector<bf16, 8 * PR> sc8 = aie::concat(sc4, sc4);
      for (int k0 = 32 * g; k0 < 32 * g + 32; k0 += 8) {
        aie::vector<int8_t, 8 * PR> c = aie::load_v<8 * PR>(strip + k0 * PR);
        aie::accum<accfloat, 8 * PR> cf;
        cf.from_vector(aie::to_float(c, 0));
        aie::accum<accfloat, 8 * PR> w = aie::mul(cf.template to_vector<bf16>(), sc8);
        aie::vector<bf16, 8 * PR> wb = w.template to_vector<bf16>();
        aie::vector<bf16, 4 * PR> t0 = aie::transpose(wb.template extract<4 * PR>(0), 4, PR);
        aie::vector<bf16, 4 * PR> t1 = aie::transpose(wb.template extract<4 * PR>(1), 4, PR);
        auto [r0, r1] = aie::interleave_zip(t0, t1, 4);
#pragma clang loop unroll(full)
        for (int p = 0; p < PR / 2; p++) {
          aie::store_v(out + (s * PR + p) * width + k0, r0.template extract<8>(p));
          aie::store_v(out + (s * PR + PR / 2 + p) * width + k0, r1.template extract<8>(p));
        }
      }
    }
  }
}

} // extern "C"
