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
#ifndef MX_BFP
__attribute__((noinline)) static void mx4_rows_mm(const mx4_blk_t *A, const bf16 *x, float *acc) {
#ifdef MX_NOCOMPUTE
  return; // a measurement: the transfers alone
#endif
  static_assert(SB % 4 == 0, "rows in tiles of 4");
  constexpr int G = SB / 4; // row tiles an object
  // A strip (16 out-features: two 8-wide tiles) at a time, so that a unit of codes is decoded
  // once for all SB rows and only 2 G accumulators are live (with all four tiles at once, 8 rows
  // an object, decoding took about 5000 of a call's 5500 cycles and the tiles went to the stack)
  for (int s = 0; s < 2; s++) {
    MM c[G][2];
#pragma clang loop unroll(full)
    for (int g = 0; g < G; g++) {
      c[g][0] = MM();
      c[g][1] = MM();
    }
    for (int j = 0; j < KT / 8; j++) {
#ifdef MX_FASTDECODE // a measurement: the codes' bytes taken as bf16 as they are (wrong values), no decoding
      const bf16 *raw = reinterpret_cast<const bf16 *>(A->codes) + ((s * (KT / 8) + j) % 39) * 64;
      auto [b0, b1] = aie::interleave_unzip(aie::load_v<64>(raw), aie::load_v<64>(raw + 64), 8);
#else
      aie::vector<int8_t, 8 * PR> v = mx4_codes(A, s * (KT / 8) + j);
      aie::accum<accfloat, 8 * PR> f;
      f.from_vector(aie::to_float(v, 0));
      aie::vector<bf16, 8 * PR> w = f.template to_vector<bf16>();
      aie::vector<bf16, 64> sc = scale_rep(exp_scales(A->exps + (j / 4) * MT + s * PR));
      aie::vector<bf16, 64> w0 = aie::mul(w.template extract<64>(0), sc).template to_vector<bf16>();
      aie::vector<bf16, 64> w1 = aie::mul(w.template extract<64>(1), sc).template to_vector<bf16>();
      // [k][16 m] to two [k][8 m] tiles: the 8-wide chunks alternate between them
      auto [b0, b1] = aie::interleave_unzip(w0, w1, 8);
#endif
#pragma clang loop unroll(full)
      for (int g = 0; g < G; g++) {
        const bf16 *xr = x + (4 * g) * KT + 8 * j;
        aie::vector<bf16, 32> a = aie::concat(aie::load_v<8>(xr), aie::load_v<8>(xr + KT),
                                              aie::load_v<8>(xr + 2 * KT), aie::load_v<8>(xr + 3 * KT));
        c[g][0].mac(a, b0);
        c[g][1].mac(a, b1);
      }
    }
    // C tile [4 rows][8 out-features] of out-features 16 s + 8 t into acc's rows
#pragma clang loop unroll(full)
    for (int g = 0; g < G; g++) {
#pragma clang loop unroll(full)
      for (int t = 0; t < 2; t++) {
        aie::vector<float, 32> v = c[g][t].template to_vector<float>();
        const int out = s * PR + t * 8;
#pragma clang loop unroll(disable)
        for (int r = 0; r < 4; r++) {
          float *dst = acc + (4 * g + r) * MT + out;
          aie::store_v(dst, aie::add(aie::load_v<8>(dst), v.template extract<8>(r)));
        }
      }
    }
  }
}

#endif // MX_BFP

#ifdef MX_BFP
// The same on AIE2P's native float matrix instruction, bfp16 (mac_8x8_8x8T: 8 rows by 8 in-features
// by 8 out-features, 512 products; the bf16 mmul above is emulated with broadcast macs, about 32
// products a cycle). A k-step's 8 in-features of the 8 rows are converted to bfp16ebs8 (an exponent
// shared by each row's 8: the activations rounded to 8-bit mantissas, as AMD's own prefill GEMMs),
// and each 8 out-features' tile of the weights is built exactly: a tile's 8 values sharing an
// exponent are an out-feature's 8 in-features, inside one MXFP4 group (q4nx_dequant.cc's order:
// the nibbles' 8 k of 16 n, n 0-7 in the even words, transposed to n * 8 + k).
// A tile built directly (bfp16ebs8 is {v64int8 mantissa, v8int8 exponent}, a value mantissa *
// 2^(exponent - 133), aie.utils.bfp): the mantissas the codes' v = 2 e2m1 (-12 .. 12, exact), the
// exponent of out-feature n its group's E + 5 (v 2^(E - 128) = v 2^(E + 5 - 133)). No float work.
static inline v64bfp16ebs8 mx4_tile(aie::vector<uint32_t, 16> w, int half, uint64_t exps8) {
  aie::vector<uint32_t, 16> h = ::shuffle(w, w, T32_16x2_lo + half);
  aie::vector<uint4, 64> q = h.template extract<8>(0).template cast_to<uint4>();
  aie::vector<uint8_t, 64> c = aie::unpack(q); // [k][n]
  aie::vector<int8_t, 64> m = aie::vector_cast<int8_t>(aie::bit_and(uint8_t(7), c));
  // 2 m and 4 m by adds (an int8 upshift goes through the accumulators)
  aie::vector<int8_t, 64> d = aie::add(m, m);
  aie::vector<int8_t, 64> m2 = aie::sub(d, int8_t(4));
  aie::vector<int8_t, 64> m4 = aie::sub(aie::add(d, d), int8_t(16));
  aie::vector<int8_t, 64> v = aie::max(m, aie::max(m2, m4));
  v = aie::select(v, aie::neg(v), aie::ge(c, uint8_t(8)));
  aie::vector<int8_t, 64> vt = aie::transpose(v, 8, 8); // [n][k]: an exponent's 8 values one n's
  v64bfp16ebs8 t;
  t.mantissa = vt;
  t.exponent = __builtin_bit_cast(v8int8, exps8 + 0x0505050505050505ull);
  return t;
}

__attribute__((noinline)) static void mx4_rows_mm(const mx4_blk_t *A, const bf16 *x, float *acc) {
#ifdef MX_NOCOMPUTE
  return; // a measurement: the transfers alone
#endif
  static_assert(SB == 8, "a row tile of 8");
  aie::accum<accfloat, 64> c0 = aie::zeros<accfloat, 64>(), c1 = c0, c2 = c0, c3 = c0;
  const uint32_t *codes = reinterpret_cast<const uint32_t *>(A->codes);
  for (int j = 0; j < KT / 8; j++) {
    const bf16 *xr = x + 8 * j;
    aie::vector<bf16, 64> a = aie::concat(
        aie::concat(aie::load_v<8>(xr), aie::load_v<8>(xr + KT), aie::load_v<8>(xr + 2 * KT), aie::load_v<8>(xr + 3 * KT)),
        aie::concat(aie::load_v<8>(xr + 4 * KT), aie::load_v<8>(xr + 5 * KT), aie::load_v<8>(xr + 6 * KT),
                    aie::load_v<8>(xr + 7 * KT)));
    aie::accum<accfloat, 64> af;
    af.from_vector(a);
    aie::block_vector<bfp16ebs8, 64> ab = af.template to_vector<bfp16ebs8>();
    const uint64_t *ex = reinterpret_cast<const uint64_t *>(A->exps + (j / 4) * MT); // n 8 t .. 8 t + 7
    // strip 0 (n 0-15) and strip 1 (n 16-31): 8 k of 16 n a 512-bit load each
    aie::vector<uint32_t, 16> w0 = aie::load_v<16>(codes + j * 16);
    aie::vector<uint32_t, 16> w1 = aie::load_v<16>(codes + (KT / 8 + j) * 16);
    c0 = mac_8x8_8x8T(ab, mx4_tile(w0, 0, ex[0]), c0);
    c1 = mac_8x8_8x8T(ab, mx4_tile(w0, 1, ex[1]), c1);
    c2 = mac_8x8_8x8T(ab, mx4_tile(w1, 0, ex[2]), c2);
    c3 = mac_8x8_8x8T(ab, mx4_tile(w1, 1, ex[3]), c3);
  }
  // C lanes m * 8 + n: row m's 8 out-features of the tile
  aie::vector<float, 64> v[4] = {c0.template to_vector<float>(), c1.template to_vector<float>(),
                                 c2.template to_vector<float>(), c3.template to_vector<float>()};
#pragma clang loop unroll(disable)
  for (int r = 0; r < 8; r++) {
    float *dst = acc + r * MT;
#pragma clang loop unroll(full)
    for (int t = 0; t < 4; t++)
      aie::store_v(dst + 8 * t, aie::add(aie::load_v<8>(dst + 8 * t), v[t].template extract<8>(r)));
  }
}

// A block decoded once into tiles (DEC_BYTES: the 160 tiles' mantissas, then their exponents),
// then used by each of a slot's objects of rows (mx4_rows_mm_dec): a block's 160 tile builds once
// rather than once per 8 rows.
constexpr int TILES = (KT / 8) * 4;
constexpr int GROUPS = KT / 32;                     // a group's 4 k-steps share their tiles' exponents
constexpr int DEC_BYTES = TILES * 64 + GROUPS * 4 * 8; // mantissas, then each (group, tile)'s exponents

static inline void mx4_decode(const mx4_blk_t *A, int8_t *dec) {
  const uint32_t *codes = reinterpret_cast<const uint32_t *>(A->codes);
  uint64_t *ex = reinterpret_cast<uint64_t *>(dec + TILES * 64);
  for (int j = 0; j < KT / 8; j++) {
    const uint64_t *e8 = reinterpret_cast<const uint64_t *>(A->exps + (j / 4) * MT);
    aie::vector<uint32_t, 16> w0 = aie::load_v<16>(codes + j * 16);
    aie::vector<uint32_t, 16> w1 = aie::load_v<16>(codes + (KT / 8 + j) * 16);
    v64bfp16ebs8 t0 = mx4_tile(w0, 0, e8[0]), t1 = mx4_tile(w0, 1, e8[1]);
    v64bfp16ebs8 t2 = mx4_tile(w1, 0, e8[2]), t3 = mx4_tile(w1, 1, e8[3]);
    int8_t *m = dec + j * 4 * 64;
    aie::store_v(m, aie::vector<int8_t, 64>(t0.mantissa));
    aie::store_v(m + 64, aie::vector<int8_t, 64>(t1.mantissa));
    aie::store_v(m + 128, aie::vector<int8_t, 64>(t2.mantissa));
    aie::store_v(m + 192, aie::vector<int8_t, 64>(t3.mantissa));
    if (j % 4 == 0) {
      ex[j] = __builtin_bit_cast(uint64_t, t0.exponent); // (j / 4) * 4 + t
      ex[j + 1] = __builtin_bit_cast(uint64_t, t1.exponent);
      ex[j + 2] = __builtin_bit_cast(uint64_t, t2.exponent);
      ex[j + 3] = __builtin_bit_cast(uint64_t, t3.exponent);
    }
  }
}

static inline v64bfp16ebs8 dec_tile(const int8_t *dec, int j, int t) {
  v64bfp16ebs8 b;
  b.mantissa = aie::load_v<64>(dec + (4 * j + t) * 64);
  b.exponent = __builtin_bit_cast(v8int8, reinterpret_cast<const uint64_t *>(dec + TILES * 64)[(j / 4) * 4 + t]);
  return b;
}

__attribute__((noinline)) static void mx4_rows_mm_dec(const int8_t *dec, const bf16 *x, float *acc) {
#ifdef MX_NOCOMPUTE
  return;
#endif
  aie::accum<accfloat, 64> c0 = aie::zeros<accfloat, 64>(), c1 = c0, c2 = c0, c3 = c0;
  for (int j = 0; j < KT / 8; j++) {
    const bf16 *xr = x + 8 * j;
    aie::vector<bf16, 64> a = aie::concat(
        aie::concat(aie::load_v<8>(xr), aie::load_v<8>(xr + KT), aie::load_v<8>(xr + 2 * KT), aie::load_v<8>(xr + 3 * KT)),
        aie::concat(aie::load_v<8>(xr + 4 * KT), aie::load_v<8>(xr + 5 * KT), aie::load_v<8>(xr + 6 * KT),
                    aie::load_v<8>(xr + 7 * KT)));
    aie::accum<accfloat, 64> af;
    af.from_vector(a);
    aie::block_vector<bfp16ebs8, 64> ab = af.template to_vector<bfp16ebs8>();
    c0 = mac_8x8_8x8T(ab, dec_tile(dec, j, 0), c0);
    c1 = mac_8x8_8x8T(ab, dec_tile(dec, j, 1), c1);
    c2 = mac_8x8_8x8T(ab, dec_tile(dec, j, 2), c2);
    c3 = mac_8x8_8x8T(ab, dec_tile(dec, j, 3), c3);
  }
  aie::vector<float, 64> v[4] = {c0.template to_vector<float>(), c1.template to_vector<float>(),
                                 c2.template to_vector<float>(), c3.template to_vector<float>()};
#pragma clang loop unroll(disable)
  for (int r = 0; r < 8; r++) {
    float *dst = acc + r * MT;
#pragma clang loop unroll(full)
    for (int t = 0; t < 4; t++)
      aie::store_v(dst + 8 * t, aie::add(aie::load_v<8>(dst + 8 * t), v[t].template extract<8>(r)));
  }
}
#endif

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

#ifdef MX_BFP
// The decoded forms: dec (one block, DEC_BYTES) or dec2 (gate then up), skipped for an empty slot.
void mx4_rows_dec(const uint8_t *block, int8_t *dec, const int32_t *counts, int32_t e) {
  if (counts[e] > 0)
    mx4_decode(reinterpret_cast<const mx4_blk_t *>(block), dec);
}

void mx4_rows_dec2(const uint8_t *g, const uint8_t *u, int8_t *dec, const int32_t *counts, int32_t e) {
  if (counts[e] > 0) {
    mx4_decode(reinterpret_cast<const mx4_blk_t *>(g), dec);
    mx4_decode(reinterpret_cast<const mx4_blk_t *>(u), dec + DEC_BYTES);
  }
}

void mx4_rows_blockd(const int8_t *dec, const bf16 *x, float *acc, const int32_t *counts, int32_t e, int32_t b) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const int lo = b * SB;
  if (counts[e] > lo)
    mx4_rows_mm_dec(dec, x, acc + lo * MT);
}

void mx4_rows_block2d(const int8_t *dec, const bf16 *x, float *ag, float *au, const int32_t *counts, int32_t e,
                      int32_t b) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const int lo = b * SB;
  if (counts[e] > lo) {
    mx4_rows_mm_dec(dec, x, ag + lo * MT);
    mx4_rows_mm_dec(dec + DEC_BYTES, x, au + lo * MT);
  }
}
#endif

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
