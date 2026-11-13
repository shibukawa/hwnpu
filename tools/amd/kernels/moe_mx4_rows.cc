//===- moe_mx4_rows.cc - gpt-oss's experts over a prompt chunk, mx4 -*- C++ -*-===//
//
// The experts of a gpt-oss prompt chunk (tools/amd/iron_gptoss_moe.py's MX4GroupGateUp and
// MX4GroupDown), on moe_mx4.cc's blocks: every expert's weights stream through the cores once a
// pass, and each expert's rows (the chunk's tokens the host routed to it, up to MX_CAP) go
// through each block as it passes, MX_SB rows an object of x. How many rows an expert has is
// data (the counts, the first object of the x channel); the streams are static, so a core takes
// every object and computes the rows below the count only.
//
// A block's codes are decoded once per object of rows (mx4_rows_mm, AIE2P's matrix unit) and
// multiply each of them, the products summed in float32 over the block. The accumulators hold MX_CAP rows of
// the block's 32 out-features (gate and up each, or down's).
//===----------------------------------------------------------------------===//

#include "utils.h"
#include "common/exp_f32.h"

#include <stdint.h>

#ifndef MX_KT
#define MX_KT 320
#endif
#ifndef MX_CAP
#define MX_CAP 64
#endif
#ifndef MX_SB
#define MX_SB 16
#endif
#ifndef MX_LIMIT
#define MX_LIMIT 7.0f
#endif

constexpr int KT = MX_KT;
constexpr int MT = 32;
constexpr int PR = 16;
constexpr int CAP = MX_CAP;
constexpr int SB = MX_SB;

struct mx4_blk_t {
  uint8_t exps[MT * KT / 32]; // each group's MXFP4 exponent (exp_scales)
  uint8_t codes[MT * KT / 2];
  bf16 bias[MT];
};

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

// acc[r][0:MT] += dequantize(A) * x[r][0:KT] for rows r in [lo, hi) of the object x (SB rows of
// KT, its row 0 the chunk's row base), acc rows MT apart.
static inline void mx4_rows(const mx4_blk_t *A, const bf16 *x, float *acc, int base, int lo, int hi) {
  int u = 0;
  for (int row = 0; row < MT; row += PR) {
    uint32_t scale_offset = row;
    for (int sub = 0; sub < KT / 32; sub++) {
      aie::vector<bf16, PR> a_scales = exp_scales(A->exps + scale_offset);
      scale_offset += MT;
      aie::vector<bf16, PR * 8> a_bf16[4];
#pragma clang loop unroll(full)
      for (int sv = 0; sv < 4; sv++) {
        aie::vector<int8_t, PR * 8> a = mx4_codes(A, u + sv);
        aie::accum<accfloat, PR * 8> f;
        f.from_vector(aie::to_float(a, 0));
        a_bf16[sv] = f.template to_vector<bf16>();
      }
      u += 4;
      for (int r = lo; r < hi; r++) {
        float *c = acc + r * MT + row;
        aie::vector<bf16, 32> b_col = aie::load_v<32>(x + (r - base) * KT + 32 * sub);
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
        aie::accum<accfloat, PR> c_accum;
        c_accum.from_vector(aie::load_v<PR>(c));
        aie::vector<bf16, PR> hi_ = t.template to_vector<bf16>();
        aie::vector<bf16, PR> lo_ = aie::mac(t, hi_, bf16(-1.0f)).template to_vector<bf16>();
        c_accum = aie::mac(c_accum, hi_, a_scales);
        c_accum = aie::mac(c_accum, lo_, a_scales);
        aie::store_v(c, c_accum.template to_vector<float>());
      }
    }
  }
}

// The same through AIE2P's bf16 matrix unit (mac_4x8_8x8: 4 rows by 8 in-features times 8 by 8
// out-features, 256 products an instruction against the vector mac's 16): rows [base, base + SB)
// of the object x, acc rows MT apart. A k-step of 8 in-features decodes the two strips' units
// once, scales them (the power of two: exact in bf16) and splits each into two 8 x 8 tiles of B;
// each 4 rows' A tile (assembled from the rows' 8 in-features) multiplies all four. The products
// sum in float32 over the whole k-block (no two-part partial sums: the scale is in the weights).
using MM = aie::mmul<4, 8, 8, bfloat16, bfloat16>;

static inline aie::vector<bf16, 64> scale_rep(aie::vector<bf16, PR> sc) {
  aie::vector<bf16, 32> s2 = aie::concat(sc, sc);
  aie::vector<bf16, 64> s4 = aie::concat(s2, s2);
  return s4;
}

// noinline: inlined in both of block2's calls the gate-up core's program passed its 16 KB
__attribute__((noinline)) static void mx4_rows_mm(const mx4_blk_t *A, const bf16 *x, float *acc) {
  static_assert(SB % 4 == 0, "rows in tiles of 4");
  constexpr int G = SB / 4; // row tiles an object
  MM c[G][4];
#pragma clang loop unroll(full)
  for (int g = 0; g < G; g++)
#pragma clang loop unroll(full)
    for (int t = 0; t < 4; t++)
      c[g][t] = MM();
  for (int j = 0; j < KT / 8; j++) {
    aie::vector<bf16, 64> b[4];
    for (int s = 0; s < 2; s++) {
      aie::vector<int8_t, 8 * PR> v = mx4_codes(A, s * (KT / 8) + j);
      aie::accum<accfloat, 8 * PR> f;
      f.from_vector(aie::to_float(v, 0));
      aie::vector<bf16, 8 * PR> w = f.template to_vector<bf16>();
      aie::vector<bf16, 64> sc = scale_rep(exp_scales(A->exps + (j / 4) * MT + s * PR));
      aie::vector<bf16, 64> w0 = aie::mul(w.template extract<64>(0), sc).template to_vector<bf16>();
      aie::vector<bf16, 64> w1 = aie::mul(w.template extract<64>(1), sc).template to_vector<bf16>();
      // [k][16 m] to two [k][8 m] tiles: the 8-wide chunks alternate between them
      auto [lo, hi] = aie::interleave_unzip(w0, w1, 8);
      b[2 * s] = lo;
      b[2 * s + 1] = hi;
    }
#pragma clang loop unroll(full)
    for (int g = 0; g < G; g++) {
      const bf16 *xr = x + (4 * g) * KT + 8 * j;
      aie::vector<bf16, 32> a = aie::concat(aie::load_v<8>(xr), aie::load_v<8>(xr + KT),
                                            aie::load_v<8>(xr + 2 * KT), aie::load_v<8>(xr + 3 * KT));
#pragma clang loop unroll(full)
      for (int t = 0; t < 4; t++)
        c[g][t].mac(a, b[t]);
    }
  }
  // C tile [4 rows][8 out-features] of out-features 8 t (t = 2 strip + half) into acc's rows
  // (rolled: unrolled, the gate-up core's program passed its 16 KB)
#pragma clang loop unroll(full)
  for (int g = 0; g < G; g++) {
#pragma clang loop unroll(full)
    for (int t = 0; t < 4; t++) {
      aie::vector<float, 32> v = c[g][t].template to_vector<float>();
      const int out = (t / 2) * PR + (t % 2) * 8;
#pragma clang loop unroll(disable)
      for (int r = 0; r < 4; r++) {
        float *dst = acc + (4 * g + r) * MT + out;
        aie::store_v(dst, aie::add(aie::load_v<8>(dst), v.template extract<8>(r)));
      }
    }
  }
}

extern "C" {

// counts[0..n) from the x channel's first object (int32 words).
void mx4_rows_counts(const bf16 *obj, int32_t *counts, int32_t n) {
  const int32_t *p = reinterpret_cast<const int32_t *>(obj);
  for (int i = 0; i < n; i++)
    counts[i] = p[i];
}

// acc's first counts[e] rows = the block's bias (its first k-block's).
void mx4_rows_init(const uint8_t *block, float *acc, const int32_t *counts, int32_t e) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const mx4_blk_t *A = reinterpret_cast<const mx4_blk_t *>(block);
  aie::accum<accfloat, 16> b0, b1;
  b0.from_vector(aie::load_v<16>(A->bias));
  b1.from_vector(aie::load_v<16>(A->bias + 16));
  const int n = counts[e];
  for (int r = 0; r < n; r++) {
    aie::store_v(acc + r * MT, b0.template to_vector<float>());
    aie::store_v(acc + r * MT + 16, b1.template to_vector<float>());
  }
}

// acc += the block times object b's rows (SB of the chunk's, those under counts[e]).
void mx4_rows_block(const uint8_t *block, const bf16 *x, float *acc, const int32_t *counts, int32_t e,
                    int32_t b) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const int n = counts[e];
  const int lo = b * SB;
  // every row of an object holding one under the count (the rows past it are never read)
  if (n > lo)
    mx4_rows_mm(reinterpret_cast<const mx4_blk_t *>(block), x, acc + lo * MT);
}

// Two blocks (gate and up) times the same rows: the object read once.
void mx4_rows_block2(const uint8_t *g, const uint8_t *u, const bf16 *x, float *ag, float *au,
                     const int32_t *counts, int32_t e, int32_t b) {
  mx4_rows_block(g, x, ag, counts, e, b);
  mx4_rows_block(u, x, au, counts, e, b);
}

// gpt-oss's glu of 16 gate and up values: (clamp(u, -L, L) + 1) g' / (1 + exp(-1.702 g')).
static inline aie::vector<bf16, 16> glu16(aie::vector<float, 16> gv, aie::vector<float, 16> uv) {
  const auto lim = aie::broadcast<float, 16>(MX_LIMIT);
  const auto nlim = aie::broadcast<float, 16>(-MX_LIMIT);
  const auto one = aie::broadcast<float, 16>(1.0f);
  gv = aie::min(gv, lim);
  uv = aie::max(aie::min(uv, lim), nlim);
  aie::accum<accfloat, 16> z;
  z.from_vector(aie::mul(gv, -1.702f).template to_vector<float>());
  aie::vector<float, 16> ex = exp_f32<16>(z).template to_vector<float>();
  aie::vector<float, 16> sg = aie::inv(aie::add(ex, one));
  aie::vector<float, 16> p = aie::mul(gv, sg).template to_vector<float>();
  aie::accum<accfloat, 16> o;
  o.from_vector(aie::mul(p, aie::add(uv, one)).template to_vector<float>());
  return o.template to_vector<bf16>();
}

// y = gpt-oss's glu (moe_mx4.cc's mx4_glu) of row r's gate and up, zeros past counts[e]. Written
// without loops or an early return: as loops, llvm-aie's alignment pass failed on it
// ("LoopStart/LoopBody: insufficient padding").
void mx4_rows_glu(const float *g, const float *u, const int32_t *counts, int32_t e, int32_t r, bf16 *y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const bool live = r < counts[e];
  const int at = (live ? r : 0) * MT;
  aie::vector<bf16, 16> y0 = glu16(aie::load_v<16>(g + at), aie::load_v<16>(u + at));
  aie::vector<bf16, 16> y1 = glu16(aie::load_v<16>(g + at + 16), aie::load_v<16>(u + at + 16));
  const auto zero = aie::zeros<bf16, 16>();
  aie::store_v(y, live ? y0 : zero);
  aie::store_v(y + 16, live ? y1 : zero);
}

// y = row r of acc narrowed to bf16, zeros past counts[e].
void mx4_rows_out(const float *acc, const int32_t *counts, int32_t e, int32_t r, bf16 *y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  if (r >= counts[e]) {
    aie::store_v(y, aie::zeros<bf16, 16>());
    aie::store_v(y + 16, aie::zeros<bf16, 16>());
    return;
  }
  narrow_to_bf16<MT>(y, acc + r * MT);
}

} // extern "C"
