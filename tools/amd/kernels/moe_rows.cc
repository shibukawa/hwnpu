//===- moe_rows.cc - a mixture of experts' prompt rows on q4nx ----*- C++ -*-===//
//
// The experts of a prompt chunk's mixture-of-experts layer (tools/amd/iron_moep.py):
// every expert's weights stream through the cores once, and each expert's rows
// (the chunk's tokens the host routed to it, sorted by expert) run through its
// blocks as they pass. How many rows an expert has is data (counts, the first
// objects on the x channel), so the cores loop over it; the streams themselves
// are static (8 rows a token, whatever the routing).
//
//   gate and up: a core keeps one strip (16 out-features) of a block's 32, of the
//     gate and of the up, for up to MOE_CAP rows of an expert, and writes
//     silu(g) u for each row once the expert's k-blocks have passed (moe_gu_glu);
//     two cores share each block, a strip each (moe_gu_*)
//   down: a core takes one 32-out-feature block of both k-blocks of an expert
//     at a time, and writes it for each of the expert's rows (moe_rows_down)
//===----------------------------------------------------------------------===//

#include "q4nx.h"
#include "common/activations.h"

#ifndef MOE_CAP
#define MOE_CAP 256
#endif

constexpr int KT = 256; // in-features a block
constexpr int MT = 32;  // out-features a block, and a core's
constexpr int CAP = MOE_CAP;

using block_t = q4nx_block<MT, KT>;

namespace {
// q4nx_accumulate (q4nx.h) for two rows of x at once: the codes are expanded to bf16 once and
// multiply both, in the same order as two calls would (the same sums, the same roundings).
template <int M, int N>
inline void q4nx_accumulate2(const q4nx_block<M, N> *A, const bf16 *B0, const bf16 *B1,
                             float *c0, float *c1, const bf16 *s0, const bf16 *s1) {
  constexpr int pr = 16;
  const uint4 *qs_ptr = A->qs;
  AIE_LOOP_RANGE(M / pr, M / pr)
  for (int row = 0; row < M; row += pr) {
    aie::accum<accfloat, pr> acc0, acc1;
    acc0.from_vector(aie::load_v<pr>(c0 + row));
    acc1.from_vector(aie::load_v<pr>(c1 + row));
    uint32_t sm = row;
    AIE_LOOP_RANGE(N / 32, N / 32)
    for (int sub = 0; sub < N / 32; sub++) {
      aie::vector<bf16, 32> b0 = aie::load_v<32>(B0 + 32 * sub);
      aie::vector<bf16, 32> b1 = aie::load_v<32>(B1 + 32 * sub);
      aie::vector<bf16, pr> scales = aie::load_v<pr>(A->scales + sm);
      aie::vector<bf16, pr> mins = aie::load_v<pr>(A->mins + sm);
      sm += M;
      aie::vector<bf16, pr * 8> cc[4];
#pragma clang loop unroll(full)
      for (int sv = 0; sv < 4; sv++) {
        aie::vector<uint4, pr * 8> q = aie::load_v<pr * 8>(qs_ptr);
        qs_ptr += pr * 4;
        aie::accum<accfloat, pr * 8> f;
        f.from_vector(aie::to_float(q, 0));
        cc[sv] = f.template to_vector<bf16>();
      }
      aie::accum<accfloat, pr> t0 = aie::mul(cc[0].extract<pr>(0), b0.get(0));
      aie::accum<accfloat, pr> t1 = aie::mul(cc[0].extract<pr>(0), b1.get(0));
#pragma clang loop unroll(full)
      for (int sv = 0; sv < 4; sv++) {
#pragma clang loop unroll(full)
        for (int k = 0; k < 8; k++) {
          if (sv == 0 && k == 0)
            continue;
          t0 = aie::mac(t0, cc[sv].extract<pr>(k), b0.get(sv * 8 + k));
          t1 = aie::mac(t1, cc[sv].extract<pr>(k), b1.get(sv * 8 + k));
        }
      }
      acc0 = aie::mac(acc0, t0.template to_vector<bf16>(), scales);
      acc0 = aie::mac(acc0, mins, s0[sub]);
      acc1 = aie::mac(acc1, t1.template to_vector<bf16>(), scales);
      acc1 = aie::mac(acc1, mins, s1[sub]);
    }
    aie::store_v(c0 + row, acc0.template to_vector<float>());
    aie::store_v(c1 + row, acc1.template to_vector<float>());
  }
}

// q4nx_accumulate (q4nx.h) on strip h of the block (its out-features 16 h .. 16 h + 15) for R
// rows of x (1 or 2) at once: the same sums and roundings as q4nx_accumulate on that strip.
template <int R, int M, int N>
inline void q4nx_strip(const q4nx_block<M, N> *A, int h, const bf16 *const *B, float *const *c,
                       const bf16 *const *s) {
  constexpr int pr = 16;
  const int row = h * pr;
  const uint4 *qs_ptr = A->qs + h * (N / 32) * 4 * (pr * 4);
  aie::accum<accfloat, pr> acc[R];
#pragma clang loop unroll(full)
  for (int i = 0; i < R; i++)
    acc[i].from_vector(aie::load_v<pr>(c[i] + row));
  uint32_t sm = row;
  AIE_LOOP_RANGE(N / 32, N / 32)
  for (int sub = 0; sub < N / 32; sub++) {
    aie::vector<bf16, 32> b[R];
#pragma clang loop unroll(full)
    for (int i = 0; i < R; i++)
      b[i] = aie::load_v<32>(B[i] + 32 * sub);
    aie::vector<bf16, pr> scales = aie::load_v<pr>(A->scales + sm);
    aie::vector<bf16, pr> mins = aie::load_v<pr>(A->mins + sm);
    sm += M;
    aie::vector<bf16, pr * 8> cc[4];
#pragma clang loop unroll(full)
    for (int sv = 0; sv < 4; sv++) {
      aie::vector<uint4, pr * 8> q = aie::load_v<pr * 8>(qs_ptr);
      qs_ptr += pr * 4;
      aie::accum<accfloat, pr * 8> f;
      f.from_vector(aie::to_float(q, 0));
      cc[sv] = f.template to_vector<bf16>();
    }
    aie::accum<accfloat, pr> t[R];
#pragma clang loop unroll(full)
    for (int i = 0; i < R; i++)
      t[i] = aie::mul(cc[0].extract<pr>(0), b[i].get(0));
#pragma clang loop unroll(full)
    for (int sv = 0; sv < 4; sv++) {
#pragma clang loop unroll(full)
      for (int k = 0; k < 8; k++) {
        if (sv == 0 && k == 0)
          continue;
#pragma clang loop unroll(full)
        for (int i = 0; i < R; i++)
          t[i] = aie::mac(t[i], cc[sv].extract<pr>(k), b[i].get(sv * 8 + k));
      }
    }
#pragma clang loop unroll(full)
    for (int i = 0; i < R; i++) {
      acc[i] = aie::mac(acc[i], t[i].template to_vector<bf16>(), scales);
      acc[i] = aie::mac(acc[i], mins, s[i][sub]);
    }
  }
#pragma clang loop unroll(full)
  for (int i = 0; i < R; i++)
    aie::store_v(c[i] + row, acc[i].template to_vector<float>());
}

// sums[g] = x[32 g] + ... + x[32 g + 31] for one k-block of x (q4nx's minimums).
inline void tile_sums(const bf16 *x, bf16 *sums) {
  for (int l = 0; l < KT / 32; l++)
    sums[l] = bf16(aie::reduce_add(aie::load_v<32>(x + 32 * l)));
}
} // namespace

extern "C" {

// The routing's row counts (each expert's, then the padding rows' at 256), 128 int32 an
// object of 256 bf16 (MoEGroupGateUp's x) or 256 of 512 (MoEGroupDown's), part of them.
void moe_rows_counts(const bf16 *a, int32_t *counts, int32_t part, int32_t words) {
  const int32_t *p = reinterpret_cast<const int32_t *>(a);
  for (int i = 0; i < words; i++)
    counts[part * words + i] = p[i];
}

// The gate and up of MoEGroupGateUp: a core's strip (16 out-features, ST) of each row.
constexpr int ST = 16;

void moe_gu_zero(float *g, float *u) {
  for (int i = 0; i < CAP * ST; i += 16) {
    aie::store_v(g + i, aie::zeros<float, 16>());
    aie::store_v(u + i, aie::zeros<float, 16>());
  }
}

// Row r of an expert: acc[r] += strip h of the block times one k-block x of the row.
void moe_gu_acc(const uint8_t *blk, const bf16 *x, float *acc, int32_t r, int32_t h) {
  alignas(32) bf16 sums[KT / 32];
  tile_sums(x, sums);
  const bf16 *xs[1] = {x}, *ss[1] = {sums};
  float *cs[1] = {acc + r * ST - h * ST};
  q4nx_strip<1>(reinterpret_cast<const block_t *>(blk), h, xs, cs, ss);
}

// Rows r and r + 1 at once (moe_gu_acc twice).
void moe_gu_acc2(const uint8_t *blk, const bf16 *x0, const bf16 *x1, float *acc, int32_t r,
                 int32_t h) {
  alignas(32) bf16 s0[KT / 32], s1[KT / 32];
  tile_sums(x0, s0);
  tile_sums(x1, s1);
  const bf16 *xs[2] = {x0, x1}, *ss[2] = {s0, s1};
  float *cs[2] = {acc + r * ST - h * ST, acc + (r + 1) * ST - h * ST};
  q4nx_strip<2>(reinterpret_cast<const block_t *>(blk), h, xs, cs, ss);
}

// y = silu(g[r]) u[r] (the strip's 16 out-features), then g[r] and u[r] back to zero.
void moe_gu_glu(float *g, float *u, int32_t r, bf16 *y) {
  float *gr = g + r * ST, *ur = u + r * ST;
  aie::vector<float, 16> gv = aie::load_v<16>(gr);
  aie::vector<float, 16> uv = aie::load_v<16>(ur);
  aie::store_v(gr, aie::mul(silu_vec<16>(gv), uv).template to_vector<float>());
  narrow_to_bf16<ST>(y, gr);
  aie::store_v(gr, aie::zeros<float, 16>());
  aie::store_v(ur, aie::zeros<float, 16>());
}

void moe_gu_pad(bf16 *y) {
  aie::store_v(y, aie::zeros<bf16, ST>());
}

// A padding row's output: zeros.
void moe_rows_pad(bf16 *y) {
  aie::store_v(y, aie::zeros<bf16, MT>());
}

// y = the block's 32 out-features of one row x of the expert's activation (two
// k-blocks, b0 for x[0..255] and b1 for x[256..511]).
void moe_rows_down(const uint8_t *b0, const uint8_t *b1, const bf16 *x, bf16 *y) {
  alignas(64) float acc[MT];
  alignas(32) bf16 sums[2 * KT / 32];
  for (int i = 0; i < MT; i += 16)
    aie::store_v(acc + i, aie::zeros<float, 16>());
  tile_sums(x, sums);
  tile_sums(x + KT, sums + KT / 32);
  q4nx_accumulate<MT, KT>(reinterpret_cast<const block_t *>(b0), x, acc, sums);
  q4nx_accumulate<MT, KT>(reinterpret_cast<const block_t *>(b1), x + KT, acc, sums + KT / 32);
  narrow_to_bf16<MT>(y, acc);
}

// moe_rows_down for two rows at once.
void moe_rows_down2(const uint8_t *b0, const uint8_t *b1, const bf16 *x0, const bf16 *x1,
                    bf16 *y0, bf16 *y1) {
  alignas(64) float a0[MT], a1[MT];
  alignas(32) bf16 s0[2 * KT / 32], s1[2 * KT / 32];
  for (int i = 0; i < MT; i += 16) {
    aie::store_v(a0 + i, aie::zeros<float, 16>());
    aie::store_v(a1 + i, aie::zeros<float, 16>());
  }
  tile_sums(x0, s0);
  tile_sums(x0 + KT, s0 + KT / 32);
  tile_sums(x1, s1);
  tile_sums(x1 + KT, s1 + KT / 32);
  q4nx_accumulate2<MT, KT>(reinterpret_cast<const block_t *>(b0), x0, x1, a0, a1, s0, s1);
  q4nx_accumulate2<MT, KT>(reinterpret_cast<const block_t *>(b1), x0 + KT, x1 + KT, a0, a1,
                           s0 + KT / 32, s1 + KT / 32);
  narrow_to_bf16<MT>(y0, a0);
  narrow_to_bf16<MT>(y1, a1);
}

} // extern "C"
