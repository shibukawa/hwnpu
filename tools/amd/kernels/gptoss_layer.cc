//===- gptoss_layer.cc - gpt-oss's decode step on one array -*- C++ -*-===//
//
// The kernels of tools/amd/iron_gptoss_layer.py's GptOssStep: every operator of a gpt-oss
// decode layer as a phase of one core program, so that the array is configured once (an
// operator of its own is loaded again each time it follows another: about 45 us and 1.2 us a KB
// of its programs). The core's loop is one frame for every phase (a fifo's acquire costs code:
// a frame a phase did not fit the 16 KB), the phase's kind choosing the work here:
//
//   plan    the phase's counts into cnt: [0] kind, [1] out-blocks, [2] weight blocks an
//           out-block, [3] objects out an out-block, [4] objects of x after the first, [5]
//           objects out at the end; [6] the core's column and [7] its place in it are the build's
//   begin   on x's first object (GL_XO bf16)
//   block   on each weight block (5504 bytes: an mx4 block, or q8x's)
//   emit    an object out (32 bf16) after an out-block's blocks
//   tail    an object out at the phase's end (x's object released: what it needs is in st)
//
//   E1  the experts' gate and up (moe_mx4.cc's blocks, a gate's then an up's each k-block) on
//       hn = RMSNorm(h) made here (x: h, the norm's weight at GL_NW), glu out
//   E2  their down on the activations (x: the four experts', the column's residual, the
//       routing weights): tot += w_e (D_e a_e + d_e), out tot + residual
//   Q   q, k and v: RMSNorm, q8x blocks (iron_q4's q8: an int8 code and a bf16 scale per 32
//       in-features of an out-feature, in blocks of 32 out-features by 160 in-features with the
//       block's bias: 5504 bytes too), the q and k rows turned by RoPE (interleaved pairs)
//   O   o_proj: q8x blocks on the heads' outputs, out acc + residual
//===----------------------------------------------------------------------===//

#define MX_NO_ENTRIES 1
#include "moe_mx4.cc"

#ifndef GL_EPS
#error "gptoss_layer.cc needs -DGL_EPS"
#endif
#ifndef GL_ROPE_ROWS
#define GL_ROPE_ROWS 0
#endif

constexpr int GL_H = 3072;    // the residual stream (padded)
constexpr int GL_NW = 6144;   // the norm's weight in x's object
constexpr int GL_ANG = 9216;  // the call's angles (cos and sin interleaved, 64 of a head)
constexpr int GL_QKT = 160;   // q8x's in-features a block
constexpr int GL_QK = 3200;   // q, k, v's in-features in whole blocks (x zero past GL_H)
constexpr int GL_OK = 4096;   // o_proj's in-features (zero to 4160)
constexpr int GL_RES = 12288; // o_proj: the residual's slice in x's object
constexpr int GL_R = 384;     // a column's out-features of the residual

struct q8x_blk_t {
  bf16 scales[GL_QKT / 32 * MT]; // [group][out-feature]
  int8_t qs[MT * GL_QKT];        // [strip of 16 out-features][in-feature][16]
  bf16 bias[MT];                 // the first k-block's
};
static_assert(sizeof(q8x_blk_t) == sizeof(mx4_blk_t), "one weight object for every phase");

// q8_mv.cc's q8_accumulate at 160 in-features a block.
__attribute__((noinline)) static void q8x_accumulate(const q8x_blk_t *A, const bf16 *B, float *c) {
  const int8_t *qs_ptr = A->qs;
  for (int row = 0; row < MT; row += PR) {
    aie::accum<accfloat, PR> c_accum;
    c_accum.from_vector(aie::load_v<PR>(c + row));
    const bf16 *it_B = B;
    uint32_t scale_offset = row;
    for (int sub = 0; sub < GL_QKT / 32; sub++) {
      aie::vector<bf16, 32> b_col = aie::load_v<32>(it_B);
      it_B += 32;
      aie::vector<bf16, PR> a_scales = aie::load_v<PR>(A->scales + scale_offset);
      scale_offset += MT;
      aie::vector<bf16, PR * 8> a_bf16[4];
#pragma clang loop unroll(full)
      for (int sv = 0; sv < 4; sv++) {
        aie::vector<int8_t, PR * 8> a = aie::load_v<PR * 8>(qs_ptr);
        qs_ptr += PR * 8;
        aie::accum<accfloat, PR * 8> acc;
        acc.from_vector(aie::to_float(a, 0));
        a_bf16[sv] = acc.template to_vector<bf16>();
      }
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
      aie::vector<bf16, PR> hi = t.template to_vector<bf16>();
      aie::vector<bf16, PR> lo = aie::mac(t, hi, bf16(-1.0f)).template to_vector<bf16>();
      c_accum = aie::mac(c_accum, hi, a_scales);
      c_accum = aie::mac(c_accum, lo, a_scales);
    }
    aie::store_v(c + row, c_accum.template to_vector<float>());
  }
}

static inline void gl_norm(bf16 *x) {
  aie::accum<accfloat, 16> ss = aie::zeros<accfloat, 16>();
  for (int i = 0; i < GL_H; i += 16) {
    aie::vector<bf16, 16> v = aie::load_v<16>(x + i);
    ss = aie::mac(ss, v, v);
  }
  const float mean = aie::reduce_add(ss.template to_vector<float>()) * (1.0f / GL_H);
  const float sc = aie::invsqrt(aie::broadcast<float, 16>(mean + GL_EPS)).get(0);
  const bf16 *g = x + GL_NW;
  for (int i = 0; i < GL_H; i += 16) {
    aie::accum<accfloat, 16> a = aie::mul(aie::load_v<16>(x + i), aie::load_v<16>(g + i));
    aie::accum<accfloat, 16> b;
    b.from_vector(aie::mul(a.template to_vector<float>(), sc).template to_vector<float>());
    aie::store_v(x + i, b.template to_vector<bf16>());
  }
  for (int i = GL_H; i < GL_QK; i += 16)
    aie::store_v(x + i, aie::zeros<bf16, 16>());
}

static inline void gl_bias(const bf16 *bias, float *acc) {
  for (int i = 0; i < MT; i += 16) {
    aie::accum<accfloat, 16> b;
    b.from_vector(aie::load_v<16>(bias + i));
    aie::store_v(acc + i, b.template to_vector<float>());
  }
}

// y = (clamp(u, -L, L) + 1) * g' / (1 + exp(-1.702 g')), g' = min(g, L) (moe_mx4.cc's mx4_glu).
static inline void gl_glu(const float *g, const float *u, bf16 *y) {
  alignas(64) float out[MT];
  const auto lim = aie::broadcast<float, 16>(MX_LIMIT);
  const auto nlim = aie::broadcast<float, 16>(-MX_LIMIT);
  const auto one = aie::broadcast<float, 16>(1.0f);
#pragma clang loop unroll(disable)
  for (int i = 0; i < MT; i += 16) {
    aie::vector<float, 16> gv = aie::min(aie::load_v<16>(g + i), lim);
    aie::vector<float, 16> uv = aie::max(aie::min(aie::load_v<16>(u + i), lim), nlim);
    aie::accum<accfloat, 16> z;
    z.from_vector(aie::mul(gv, -1.702f).template to_vector<float>());
    aie::vector<float, 16> e = exp_f32<16>(z).template to_vector<float>();
    aie::vector<float, 16> s = aie::inv(aie::add(e, one));
    aie::vector<float, 16> p = aie::mul(gv, s).template to_vector<float>();
    aie::store_v(out + i, aie::mul(p, aie::add(uv, one)).template to_vector<float>());
  }
  narrow_to_bf16<MT>(y, out);
}

// y = acc + r (32 values).
static inline void gl_add_out(float *acc, const bf16 *r, bf16 *y) {
  for (int i = 0; i < MT; i += 16) {
    aie::accum<accfloat, 16> t;
    t.from_vector(aie::load_v<16>(r + i));
    aie::store_v(acc + i, aie::add(aie::load_v<16>(acc + i), t.template to_vector<float>()));
  }
  narrow_to_bf16<MT>(y, acc);
}

enum { K_E1 = 1, K_E2 = 2, K_Q = 3, K_O = 4 };
constexpr int GL_OUTS = 3;       // a core's out-blocks of the residual (3072 / 32 / 8 columns / 4 cores)
constexpr int GL_CORES = 4;
constexpr int GL_XN = 4 * 3072;  // E2: the residual's slice after the four experts' activations
constexpr int GL_KBE = K / KT;   // the experts' k-blocks (K is their in-features)

struct gl_state {
  float g[MT], u[MT], acc[MT];
  float tot[GL_OUTS][MT];
  bf16 res[GL_R]; // the column's slice of the residual
  bf16 w[16];     // E2: the routing weights
};

extern "C" {

// The counts of the step's phase ph (kinds: 4 bits a phase).
void gl_plan(int32_t *cnt, int32_t ph, int32_t kinds) {
  const int kind = (kinds >> (4 * ph)) & 15;
  int outs = 0, blocks = 0, ys = 0, tail = 0;
  if (kind == K_E1) {
    outs = 3072 / MT / (2 * GL_CORES); // gate-up pairs a core
    blocks = 2 * GL_KBE;
    ys = 1;
  } else if (kind == K_E2) {
    outs = 4 * GL_OUTS; // each expert's out-blocks
    blocks = GL_KBE;
    tail = GL_OUTS;
  } else if (kind == K_Q) {
    outs = 5120 / MT / 8 / GL_CORES;
    blocks = GL_QK / GL_QKT;
    ys = 1;
  } else if (kind == K_O) {
    outs = GL_OUTS;
    blocks = (GL_OK + GL_QKT - 1) / GL_QKT;
    ys = 1;
  }
  cnt[0] = kind;
  cnt[1] = outs;
  cnt[2] = blocks;
  cnt[3] = ys;
  cnt[4] = 0;
  cnt[5] = tail;
}

void gl_begin(bf16 *x, float *state, const int32_t *cnt) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  gl_state *st = reinterpret_cast<gl_state *>(state);
  const int kind = cnt[0];
  if (kind == K_E1 || kind == K_Q) {
    gl_norm(x);
  } else if (kind == K_E2 || kind == K_O) {
    const bf16 *r = x + (kind == K_E2 ? GL_XN : GL_RES);
    for (int i = 0; i < GL_R; i += 32)
      aie::store_v(st->res + i, aie::load_v<32>(r + i));
    if (kind == K_E2) {
      aie::store_v(st->w, aie::load_v<16>(r + GL_R));
      for (int t = 0; t < GL_OUTS; t++)
        zero_256<float, MT>(st->tot[t]);
    } else {
      for (int i = GL_OK; i < GL_OK + 64; i += 16)
        aie::store_v(x + i, aie::zeros<bf16, 16>());
    }
  }
}

void gl_block(const uint8_t *block, const bf16 *x, float *state, const int32_t *cnt, int32_t o, int32_t b) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  gl_state *st = reinterpret_cast<gl_state *>(state);
  const int kind = cnt[0];
  if (kind == K_E1 || kind == K_E2) {
    const mx4_blk_t *A = reinterpret_cast<const mx4_blk_t *>(block);
    const bool up = kind == K_E1 && (b & 1);
    const int kb = kind == K_E1 ? b >> 1 : b;
    const int e = o / GL_OUTS; // E2: the expert
    float *acc = kind == K_E2 ? st->acc : up ? st->u : st->g;
    if (kb == 0)
      gl_bias(A->bias, acc);
    mx4_accumulate(A, x + (kind == K_E2 ? 3072 * e : 0) + KT * kb, acc);
    if (kind == K_E2 && kb == GL_KBE - 1) { // tot += w_e acc
      const float w = float(st->w[e]);
      float *tot = st->tot[o - GL_OUTS * e];
      for (int i = 0; i < MT; i += 16)
        aie::store_v(tot + i, aie::add(aie::load_v<16>(tot + i),
                                       aie::mul(aie::load_v<16>(acc + i), w).template to_vector<float>()));
    }
    return;
  }
  const q8x_blk_t *A = reinterpret_cast<const q8x_blk_t *>(block);
  if (b == 0)
    gl_bias(A->bias, st->acc);
  q8x_accumulate(A, x + GL_QKT * b, st->acc);
}

void gl_emit(float *state, const bf16 *x, const int32_t *cnt, int32_t o, int32_t q, bf16 *y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  gl_state *st = reinterpret_cast<gl_state *>(state);
  const int kind = cnt[0];
  if (kind == K_E1) {
    gl_glu(st->g, st->u, y);
  } else if (kind == K_O) {
    gl_add_out(st->acc, st->res + (GL_CORES * o + cnt[7]) * MT, y);
  } else { // Q: out-features row0 .. row0 + 31; below GL_ROPE_ROWS q's or k's, in pairs (2 p, 2 p + 1)
    // of frequency p of the head's half (row0 / 32) % 2
    float *acc = st->acc;
    const int row0 = cnt[6] * (5120 / 8) + cnt[7] * MT + 128 * o;
    if (row0 < GL_ROPE_ROWS) {
      aie::accum<accfloat, 32> ta;
      ta.from_vector(aie::load_v<32>(x + GL_ANG + ((row0 >> 5) & 1) * 32));
      aie::vector<float, 32> t = ta.template to_vector<float>();
      auto [c, sn] = aie::interleave_unzip(t.template extract<16>(0), t.template extract<16>(1), 1);
      auto [a, b] = aie::interleave_unzip(aie::load_v<16>(acc), aie::load_v<16>(acc + 16), 1);
      aie::vector<float, 16> a2 = aie::sub(aie::mul(a, c).template to_vector<float>(), aie::mul(b, sn).template to_vector<float>());
      aie::vector<float, 16> b2 = aie::add(aie::mul(a, sn).template to_vector<float>(), aie::mul(b, c).template to_vector<float>());
      auto [z0, z1] = aie::interleave_zip(a2, b2, 1);
      aie::store_v(acc, z0);
      aie::store_v(acc + 16, z1);
    }
    narrow_to_bf16<MT>(y, acc);
  }
}

// E2: the core's out-block q of the layer's output, tot + residual.
void gl_tail(float *state, const int32_t *cnt, int32_t q, bf16 *y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  gl_state *st = reinterpret_cast<gl_state *>(state);
  gl_add_out(st->tot[q], st->res + (GL_CORES * q + cnt[7]) * MT, y);
}

} // extern "C"
