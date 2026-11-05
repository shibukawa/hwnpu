// One decode step of an LFM3 convolution layer on one core of the AMD NPU (tools/iron/lfm3conv.py).
// Every core owns CH channels of the layer: the matching rows of in_proj (B, C and X), out_proj,
// w1 and w3 (FI each) and w2. The core keeps the full residual vector and runs the norms on
// the full vectors it is sent, so that everything but the four weight products stays local.
// The functions share the small buffers below (one object holds them all: object_file.bind);
// the large ones are the core's IRON buffers, passed in: R (the residual stream, f32 D), X (the
// input of the weight product, bf16 I, which also holds a full f32 vector to normalize) and N
// (the norm weight in use, bf16 D). Globals go to one contiguous region of the core's memory,
// which the ObjectFifo buffers leave small.
//
// The other arguments are ObjectFifo buffers: chunks of 1024 bytes of a broadcast vector, weight
// blocks of 5120 bytes (q4nx_block<32, 256>, or a parameter block) and output pieces of 64 bytes.
#include "common/activations.h" // tanh_bf16_v16 (aie::tanh on AIE2P)
#include "common/scalar_f32.h"  // scalar_mul, scalar_invsqrt
#include "flm_gemma4/q4nx.h"    // q4nx_accumulate, group_sums_32

#ifndef LYR_D
#define LYR_D 2048 // hidden size
#endif
#ifndef LYR_I
#define LYR_I 7680 // intermediate size
#endif
#ifndef LYR_CORES
#define LYR_CORES 16
#endif
#ifndef LYR_EPS
#define LYR_EPS 1e-5f
#endif
#define LYR_K 7 // convolution width
#define CH (LYR_D / LYR_CORES)
// Program memory is 16 KiB: loops that run once per stage stay rolled.
#define ROLLED _Pragma("clang loop unroll(disable)")
#define FI (LYR_I / LYR_CORES)

using blk = q4nx_block<32, 256>;

static_assert(LYR_I * sizeof(bf16) >= LYR_D * sizeof(float), "a f32 vector does not fit in X");
alignas(64) static float g_acc[2 * FI];      // outputs of the weight product
alignas(64) static bf16 g_convw[LYR_K * CH]; // convolution taps, [tap][channel]
alignas(64) static bf16 g_win[(LYR_K - 1) * CH]; // last K-1 B*x, [step][channel], oldest first
alignas(64) static unsigned char g_out[1792];    // the core's output of a stage

static inline aie::vector<float, 16> to_f32(aie::vector<bf16, 16> v) {
  return aie::mul(v, aie::broadcast<bf16, 16>(bf16(1.0f))).template to_vector<float>();
}

static inline aie::vector<bf16, 16> to_bf16(aie::vector<float, 16> v) {
  aie::accum<accfloat, 16> a;
  a.from_vector(v);
  return a.template to_vector<bf16>();
}

// 1 / rms(x) of n floats
static float inv_rms(const float *x, int n) {
  aie::accum<accfloat, 16> s = aie::zeros<accfloat, 16>();
  ROLLED
  for (int i = 0; i < n; i += 16) {
    aie::vector<float, 16> v = aie::load_v<16>(x + i);
    s = aie::mac(s, v, v);
  }
  float mean = scalar_mul(aie::reduce_add(s.template to_vector<float>()), 1.0f / n);
  return scalar_invsqrt(mean + LYR_EPS);
}

extern "C" {

// Chunk ch (1024 bytes) of a broadcast vector into R (dst 0), N (dst 1) or X (dst 2).
void lyr_ld(int32_t *chunk, int ch, int dst, float *R, bf16 *N, bf16 *X) {
  int32_t *d = (int32_t *)(dst == 0 ? (void *)R : dst == 1 ? (void *)N : (void *)X) + ch * 256;
  ROLLED
  for (int i = 0; i < 256; i += 16)
    aie::store_v(d + i, aie::load_v<16>(chunk + i));
}

// A parameter block: the core's convolution taps ([tap][channel] bf16, which 0) or its
// convolution window ([step][channel] bf16, which 1). The taps come first in a step, so their
// call also sets the rounding of every conversion to bf16 to the nearest (ties to even): the
// core starts out rounding toward minus infinity.
void lyr_ld_blk(int32_t *block, int which) {
  const bf16 *s = (const bf16 *)block;
  bf16 *d = which == 0 ? g_convw : g_win;
  int n = which == 0 ? LYR_K * CH : (LYR_K - 1) * CH;
  if (which == 0)
    aie::set_rounding(aie::rounding_mode::conv_even);
  ROLLED
  for (int i = 0; i < n; i += 32)
    aie::store_v(d + i, aie::load_v<32>(s + i));
}

// X = rms_norm(R) * N, in bf16: the input of in_proj (operator_norm) or w1, w3 (ffn_norm).
void lyr_rms_x(float *R, bf16 *N, bf16 *X) {
  float k = inv_rms(R, LYR_D);
  ROLLED
  for (int i = 0; i < LYR_D; i += 16) {
    aie::vector<float, 16> v = aie::mul(aie::load_v<16>(R + i), k).template to_vector<float>();
    v = aie::mul(v, to_f32(aie::load_v<16>(N + i))).template to_vector<float>();
    aie::store_v(X + i, to_bf16(v));
  }
}

// R += rms_norm(T) * N, T the f32 vector in X: a sandwich norm and its residual.
void lyr_resid(float *R, bf16 *N, bf16 *X) {
  const float *T = (const float *)X;
  float k = inv_rms(T, LYR_D);
  ROLLED
  for (int i = 0; i < LYR_D; i += 16) {
    aie::vector<float, 16> v = aie::mul(aie::load_v<16>(T + i), k).template to_vector<float>();
    aie::accum<accfloat, 16> a;
    a.from_vector(aie::load_v<16>(R + i));
    a = aie::mac(a, v, to_f32(aie::load_v<16>(N + i)));
    aie::store_v(R + i, a.template to_vector<float>());
  }
}

void lyr_zero(int n) {
  ROLLED
  for (int i = 0; i < n; i += 16)
    aie::store_v(g_acc + i, aie::zeros<float, 16>());
}

// g_acc[rb*32 ..] += dequantize(block) * X[cb*256 ..]
void lyr_mv(int32_t *block, int cb, int rb, bf16 *X) {
  bf16 *xs = X + cb * 256;
  alignas(32) bf16 sums[8];
  ROLLED
  for (int l = 0; l < 8; l++) // group_sums_32, rolled
    sums[l] = bf16(aie::reduce_add(aie::load_v<32>(xs + 32 * l)));
  q4nx_accumulate<32, 256>((blk *)block, xs, g_acc + rb * 32, sums);
}

// After in_proj: g_acc holds B, C and X of the core's channels. y = C * conv(B * X) in bf16
// to g_out[0..2*CH), then the new window to g_out[2*CH..).
void lyr_conv() {
  bf16 *y = (bf16 *)g_out;
  bf16 *win = (bf16 *)(g_out + 2 * CH);
  ROLLED
  for (int c = 0; c < CH; c += 16) {
    aie::vector<float, 16> b = aie::load_v<16>(g_acc + c);
    aie::vector<float, 16> gate = aie::load_v<16>(g_acc + CH + c);
    aie::vector<float, 16> x = aie::load_v<16>(g_acc + 2 * CH + c);
    aie::vector<bf16, 16> bx = to_bf16(aie::mul(b, x).template to_vector<float>());
    aie::accum<accfloat, 16> a = aie::mul(bx, aie::load_v<16>(g_convw + (LYR_K - 1) * CH + c));
    ROLLED
    for (int t = 0; t < LYR_K - 1; t++)
      a = aie::mac(a, aie::load_v<16>(g_win + t * CH + c), aie::load_v<16>(g_convw + t * CH + c));
    aie::store_v(y + c, to_bf16(aie::mul(gate, a.template to_vector<float>()).template to_vector<float>()));
    ROLLED
    for (int t = 0; t < LYR_K - 2; t++)
      aie::store_v(win + t * CH + c, aie::load_v<16>(g_win + (t + 1) * CH + c));
    aie::store_v(win + (LYR_K - 2) * CH + c, bx);
  }
}

// After w1 and w3: g_acc holds a = w1 h and b = w3 h of the core's FI rows. silu(a) * b in
// bf16 to g_out, with silu(a) = a * (0.5 + 0.5 tanh(a / 2)) in float32 but for the tanh itself
// (the native one, a bf16 result: aie::tanh<float> gives NaN).
void lyr_glu() {
  bf16 *g = (bf16 *)g_out;
  ROLLED
  for (int i = 0; i < FI; i += 16) {
    aie::vector<float, 16> a = aie::load_v<16>(g_acc + i);
    aie::accum<accfloat, 16> half_a;
    half_a.from_vector(aie::mul(a, 0.5f).template to_vector<float>());
    aie::vector<float, 16> t = to_f32(tanh_bf16_v16(half_a));
    aie::accum<accfloat, 16> s;
    s.from_vector(aie::broadcast<float, 16>(0.5f));
    aie::vector<float, 16> sig = aie::mac(s, t, 0.5f).template to_vector<float>();
    aie::vector<float, 16> silu = aie::mul(a, sig).template to_vector<float>();
    aie::store_v(g + i, to_bf16(aie::mul(silu, aie::load_v<16>(g_acc + FI + i)).template to_vector<float>()));
  }
}

// g_out = CH f32 in: the core's rows of out_proj or w2 (g_acc, core < 0), or its channels of
// the residual stream (R), the layer's output.
void lyr_out(int core, float *R) {
  float *o = (float *)g_out;
  const float *s = core < 0 ? g_acc : R + core * CH;
  ROLLED
  for (int i = 0; i < CH; i += 16)
    aie::store_v(o + i, aie::load_v<16>(s + i));
}

// Piece i (64 bytes) of g_out to an output buffer.
void lyr_st(int32_t *piece, int i) {
  aie::store_v(piece, aie::load_v<16>((const int32_t *)(g_out + 64 * i)));
}
}
