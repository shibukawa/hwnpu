//===- mv_add.h - a GEMV's output with vectors added as it leaves -*- C++ -*-===//
//
// The end of a quantized GEMV (q4_mv.cc, q6_mv.cc, q8_mv.cc) that adds Q_ADDS vectors to its
// output before narrowing it (iron_q4.py's QGEMV with adds: a projection's bias, a residual):
// one operator in place of the GEMV and an add a vector. The vectors ride on x's channel (a
// core has two input channels, the weights' and x's): x's object is x (K in-features), then each
// vector's Q_R values of the column's out-features. A core's OUT out-features of column block cb
// are at cb * 64 + h * OUT of them.
//
// Included after the kernel's K, OUT and narrow_to_bf16; defines NAME_mv_end_add.
//===----------------------------------------------------------------------===//

#ifdef Q_ADDS
#ifndef Q_R
#error "mv_add.h needs -DQ_R with -DQ_ADDS"
#endif

#define MV_ADD_CAT(a, b) a##b
#define MV_ADD_NAME(f) MV_ADD_CAT(f, _mv_end_add)

extern "C" void MV_ADD_NAME(MV_FMT)(float *acc, const bf16 *xa, int32_t cb, int32_t h, bf16 *y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const int off = cb * 64 + h * OUT;
  for (int i = 0; i < OUT; i += 16) {
    aie::vector<float, 16> a = aie::load_v<16>(acc + i);
#pragma clang loop unroll(full)
    for (int j = 0; j < Q_ADDS; j++) {
      aie::accum<accfloat, 16> t;
      t.from_vector(aie::load_v<16>(xa + K + j * Q_R + off + i));
      a = aie::add(a, t.template to_vector<float>());
    }
    aie::store_v(acc + i, a);
  }
  narrow_to_bf16<OUT>(y, acc);
  zero_256<float, OUT>(acc);
}
#endif

// A fused q, k, v projection's begin and end (iron_q4.py's QGEMVQKV): the layer's input RMSNorm
// done by each core on its x at the begin (x = x g / sqrt(mean(x^2) + eps), rounded to bf16 as
// the norm's output; the weight g after the angles on x's channel), the bias added, and the q
// and k rows (the first Q_ROPE_ROWS) turned by RoPE as they leave: the table's q and k rows are
// in interleaved pairs (out-features d and d + head_dim / 2 next to each other), the angles (cos
// and sin interleaved, 64 of a head) after the bias. A core's OUT = 32 out-features are within
// one head. (The weight folded into the table's columns instead took the q8 codes' error up.)
#ifdef Q_QKV
#ifndef Q_EPS
#error "mv_add.h needs -DQ_EPS with -DQ_QKV"
#endif

#define MV_QKV_CAT(a, b) a##b
#define MV_QKV_BEGIN(f) MV_QKV_CAT(f, _mv_begin_norm)
#define MV_QKV_END(f) MV_QKV_CAT(f, _mv_end_qkv)
#define MV_QKV_BASE(f) MV_QKV_CAT(f, _mv_begin)

extern "C" void MV_QKV_BEGIN(MV_FMT)(bf16 *xa, bf16 *sums, float *acc) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  aie::accum<accfloat, 16> ss = aie::zeros<accfloat, 16>();
  for (int i = 0; i < K; i += 16) {
    aie::vector<bf16, 16> v = aie::load_v<16>(xa + i);
    ss = aie::mac(ss, v, v);
  }
  const float mean = aie::reduce_add(ss.template to_vector<float>()) * (1.0f / K);
  aie::vector<float, 16> m = aie::broadcast<float, 16>(mean + Q_EPS);
  const float sc = aie::invsqrt(m).get(0);
  const bf16 *g = xa + K + Q_R + 64;
  for (int i = 0; i < K; i += 16) {
    aie::accum<accfloat, 16> a = aie::mul(aie::load_v<16>(xa + i), aie::load_v<16>(g + i));
    aie::vector<float, 16> f = aie::mul(a.template to_vector<float>(), sc).template to_vector<float>();
    aie::accum<accfloat, 16> b;
    b.from_vector(f);
    aie::store_v(xa + i, b.template to_vector<bf16>());
  }
  MV_QKV_BASE(MV_FMT)(xa, sums, acc);
}

extern "C" void MV_QKV_END(MV_FMT)(float *acc, const bf16 *xa, int32_t cb, int32_t h, int32_t lane, bf16 *y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const int off = cb * 64 + h * OUT;
  static_assert(OUT == 32, "a core's out-features 32 of a head");
  aie::vector<float, 16> v[2];
#pragma clang loop unroll(full)
  for (int i = 0; i < 2; i++) {
    aie::accum<accfloat, 16> b;
    b.from_vector(aie::load_v<16>(xa + K + off + 16 * i));
    v[i] = aie::add(aie::load_v<16>(acc + 16 * i), b.template to_vector<float>());
  }
  if (lane * Q_R + off < Q_ROPE_ROWS) {
    // pairs (2 p, 2 p + 1) of the core's 32 rows: frequency p of the head's half h
    aie::accum<accfloat, 32> ta;
    ta.from_vector(aie::load_v<32>(xa + K + Q_R + h * OUT));
    aie::vector<float, 32> t = ta.template to_vector<float>();
    auto [c, sn] = aie::interleave_unzip(t.template extract<16>(0), t.template extract<16>(1), 1);
    auto [a, b] = aie::interleave_unzip(v[0], v[1], 1);
    aie::vector<float, 16> a2 = aie::sub(aie::mul(a, c).template to_vector<float>(), aie::mul(b, sn).template to_vector<float>());
    aie::vector<float, 16> b2 = aie::add(aie::mul(a, sn).template to_vector<float>(), aie::mul(b, c).template to_vector<float>());
    auto [z0, z1] = aie::interleave_zip(a2, b2, 1);
    v[0] = z0;
    v[1] = z1;
  }
  aie::store_v(acc, v[0]);
  aie::store_v(acc + 16, v[1]);
  narrow_to_bf16<OUT>(y, acc);
  zero_256<float, OUT>(acc);
}
#endif

// The end of a gated GEMV (iron_q4.py's QGEMVGlu): the gate's and the up's accumulators of the
// same out-features (a column block of each, read in turn from the interleaved table), y =
// silu(g) u, then both back to zero. silu by exp in float32 (exp_f32): AIE2P's vtanh returns
// its argument for |x| <= 0.5.
#ifdef Q_GLU
#include "common/exp_f32.h"
#define MV_GLU_CAT(a, b) a##b
#define MV_GLU_NAME(f) MV_GLU_CAT(f, _mv_end_glu)

extern "C" void MV_GLU_NAME(MV_FMT)(float *g, float *u, bf16 *y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const auto one = aie::broadcast<float, 16>(1.0f);
  for (int i = 0; i < OUT; i += 16) {
    aie::vector<float, 16> gv = aie::load_v<16>(g + i);
#ifdef Q_GLU_GELU
    // gelu (tanh form, Gemma's gelu_pytorch_tanh): 0.5 g (1 + tanh(t)), t = sqrt(2 / pi) (g +
    // 0.044715 g^3), tanh(t) = 1 - 2 / (exp(2 t) + 1) in float32 (2 t clamped to +-30)
    aie::vector<float, 16> g3 = aie::mul(aie::mul(gv, gv).template to_vector<float>(), gv).template to_vector<float>();
    aie::vector<float, 16> t2 = aie::mul(aie::add(gv, aie::mul(g3, 0.044715f).template to_vector<float>()),
                                         2.0f * 0.7978845608f).template to_vector<float>();
    t2 = aie::max(aie::min(t2, aie::broadcast<float, 16>(30.0f)), aie::broadcast<float, 16>(-30.0f));
    aie::accum<accfloat, 16> z;
    z.from_vector(t2);
    aie::vector<float, 16> e = exp_f32<16>(z).template to_vector<float>();
    aie::vector<float, 16> th = aie::sub(one, aie::mul(aie::inv(aie::add(e, one)), 2.0f).template to_vector<float>());
    aie::vector<float, 16> a = aie::mul(aie::mul(gv, 0.5f).template to_vector<float>(), aie::add(one, th))
                                   .template to_vector<float>();
#else
    aie::accum<accfloat, 16> z;
    z.from_vector(aie::mul(gv, -1.0f).template to_vector<float>());  // aie::neg of floats does not lower
    aie::vector<float, 16> e = exp_f32<16>(z).template to_vector<float>();
    aie::vector<float, 16> s = aie::inv(aie::add(e, one));
    aie::vector<float, 16> a = aie::mul(gv, s).template to_vector<float>();
#endif
    aie::store_v(g + i, aie::mul(a, aie::load_v<16>(u + i)).template to_vector<float>());
  }
  narrow_to_bf16<OUT>(y, g);
  zero_256<float, OUT>(g);
  zero_256<float, OUT>(u);
}
#endif

// A GEMV whose x is a sandwich-normed layer's next input (iron_q4.py's sandwich forms of QGEMV and
// QGEMVGlu): the block's output h, the residual r, the post norm's weight and the next norm's
// (with Q_SANDWICH_SCALE then a scale row, Gemma 4's layer_scalar) ride on x's channel, K each.
// With Q_SANDWICH_PLAIN (a pre-norm decoder's residual and next norm) there is no post norm:
// y = r + h, and w_pre follows r on x's channel.
// The begin makes, on every core, y = r + rms(h) w_post (rounded; with the scale, rounded again
// times it) into r's place and n = rms(y) w_pre into h's, as kernels/sandwich_add.cc does (the
// same sums, lanes and roundings: the same bits), then starts the GEMV on n. One operator in place
// of SandwichAdd and the GEMV (docs/design/amd.md). MV_FMT_mv_put copies y or n out (a core of
// column 0 each).
#ifdef Q_SANDWICH
#ifndef Q_EPS
#error "mv_add.h needs -DQ_EPS with -DQ_SANDWICH"
#endif
#include "common/f32_split.h"
#include "common/scalar_f32.h"

#define MV_SW_CAT(a, b) a##b
#define MV_SW_BEGIN(f) MV_SW_CAT(f, _mv_begin_sandwich)
#define MV_SW_PUT(f) MV_SW_CAT(f, _mv_put)
#define MV_SW_BASE(f) MV_SW_CAT(f, _mv_begin)

namespace mv_sandwich {
constexpr int L = 32;
using bf_t = aie::vector<bf16, L>;
using acc_t = aie::accum<accfloat, L>;

inline float inv_rms(const bf16 *v) {
  acc_t sq = aie::zeros<accfloat, L>();
  for (int i = 0; i < K; i += L) {
    const bf_t x = aie::load_v<L>(v + i);
    sq = aie::mac_square(sq, x);
  }
  const float ms = scalar_mul(aie::reduce_add(sq.template to_vector<float>()), 1.0f / K);
  return scalar_invsqrt(ms + Q_EPS);
}

inline bf_t scaled(bf_t x, bf_t w, float s) {
  const Limbs<L> sl = bf16_limbs(acc_t(aie::broadcast<float, L>(s)));
  return mul_limbs(bf16_limbs(mul_limbs(sl, x)), w).template to_vector<bf16>();
}
} // namespace mv_sandwich

extern "C" void MV_SW_BEGIN(MV_FMT)(bf16 *xa, bf16 *sums, float *acc) {
  using namespace mv_sandwich;
  aie::set_rounding(aie::rounding_mode::conv_even);
  bf16 *h = xa, *r = xa + K;
  const bf_t one = aie::broadcast<bf16, L>(1.0f);
#ifdef Q_SANDWICH_PLAIN
  // A pre-norm decoder's (Q_SANDWICH_PLAIN): no post norm, y = r + h; the next norm's weight
  // third on x's channel
  const bf16 *wpre = xa + 2 * K;
  for (int i = 0; i < K; i += L) {
    bf_t y = aie::mac(aie::mul(aie::load_v<L>(h + i), one), aie::load_v<L>(r + i), one).template to_vector<bf16>();
#else
  const bf16 *wpost = xa + 2 * K, *wpre = xa + 3 * K;
  const float sh = inv_rms(h);
  for (int i = 0; i < K; i += L) {
    const bf_t p = scaled(aie::load_v<L>(h + i), aie::load_v<L>(wpost + i), sh);
    bf_t y = aie::mac(aie::mul(p, one), aie::load_v<L>(r + i), one).template to_vector<bf16>();
#endif
#ifdef Q_SANDWICH_SCALE
    y = aie::mul(y, aie::load_v<L>(xa + 4 * K + i)).template to_vector<bf16>();
#endif
    aie::store_v(r + i, y);
  }
  const float sy = inv_rms(r);
  for (int i = 0; i < K; i += L)
    aie::store_v(h + i, scaled(aie::load_v<L>(r + i), aie::load_v<L>(wpre + i), sy));
  MV_SW_BASE(MV_FMT)(xa, sums, acc);
}

// The plain begin on xa's first K (n, made by the begin above): QGEMVGlu's second accumulator.
#define MV_SW_AGAIN(f) MV_SW_CAT(f, _mv_begin_again)
extern "C" void MV_SW_AGAIN(MV_FMT)(bf16 *xa, bf16 *sums, float *acc) { MV_SW_BASE(MV_FMT)(xa, sums, acc); }

// out = xa's which-th K (0: n, 1: y).
extern "C" void MV_SW_PUT(MV_FMT)(const bf16 *xa, bf16 *out, int32_t which) {
  const bf16 *src = xa + which * K;
  for (int i = 0; i < K; i += 32)
    aie::store_v(out + i, aie::load_v<32>(src + i));
}
#endif
