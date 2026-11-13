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
//           objects out at the end, [8] whether the core takes part, [10] the tokens attended;
//           [6] the core's column and [7] its place in it are the build's
//   xblock  on each further object of x
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
//   A   the attention of the column's 8 heads (one key-value group) over the cache's keys and
//       values (one array: a row its 8 groups' keys, then their values): x's first object the
//       heads' q and sinks, then objects of 128 rows' key and value of the group; a softmax
//       kept as a running maximum, sum and weighted values (GL_ATT cores: 4 heads each on
//       cores 0 and 1)
//   O   o_proj: q8x blocks on the heads' outputs, out acc + residual
//   R   the router's logits (GL_ROUTER cores: column 0's last two): bf16 blocks of 80
//       in-features on hn, each core its k-blocks' partial sums, out as float32 (the column's
//       other cores take blocks of nothing and send zeros)
//   T   the routing (GL_ROUTE core): the partial sums added, the top 4 and their softmax
//       (moe_route.cc's), out the next command's scratchpad words and the weights
//===----------------------------------------------------------------------===//

#define MX_NO_ENTRIES 1
#include "moe_mx4.cc"

#ifndef GL_EPS
#error "gptoss_layer.cc needs -DGL_EPS"
#endif
#ifndef GL_ROPE_ROWS
#define GL_ROPE_ROWS 0
#endif

// what runs once a phase or an object out is built for size: a core's program is 16 KB
#define GL_COLD __attribute__((minsize))

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

// exp of 16 values (below -80 as -80): one copy for the glu, the softmax and the routing.
__attribute__((noinline)) static aie::vector<float, 16> gl_exp(aie::vector<float, 16> v) {
  aie::accum<accfloat, 16> z;
  z.from_vector(aie::max(v, aie::broadcast<float, 16>(-80.0f)));
  return exp_f32<16>(z).template to_vector<float>();
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
    aie::vector<float, 16> e = gl_exp(aie::mul(gv, -1.702f).template to_vector<float>());
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

enum { K_E1 = 1, K_E2 = 2, K_Q = 3, K_A = 4, K_O = 5, K_R = 6, K_T = 7 };
constexpr int GL_OUTS = 3;       // a core's out-blocks of the residual (3072 / 32 / 8 columns / 4 cores)
constexpr int GL_CORES = 4;
constexpr int GL_XN = 4 * 3072;  // E2: the residual's slice after the four experts' activations
constexpr int GL_KBE = K / KT;   // the experts' k-blocks (K is their in-features)
constexpr int GL_TOK = 128;      // A: tokens an object (a key and a value of 64 each)
constexpr int GL_D = 64;         // the head dimension
constexpr int GL_AH = 4;         // A: heads a core
constexpr int GL_SINK = 13312;   // A: the column's sinks in x's first object (after its q, repeated)
constexpr int GL_TS = 2 * GL_D;  // A: a token's key and value
constexpr int GL_RKT = 80;       // R: in-features a block
constexpr int GL_RKB = 20;       // R: blocks a core (39 k-blocks and one of zeros on two cores)
constexpr int GL_PROG = 256;     // T: the program's words in x's object (after the partial sums)
constexpr int GL_FED = 8;        // T: the scratchpad words routed (the experts' offsets)

// Every array a vector is loaded from starts on 64 bytes (a load takes the address's upper bits).
struct gl_state {
  alignas(64) float g[MT];
  alignas(64) float u[MT];
  alignas(64) float acc[MT];
  alignas(64) float tot[GL_OUTS][MT];
  alignas(64) bf16 res[GL_R]; // the column's slice of the residual; T: the words and the weights out
  alignas(64) bf16 w[32];     // E2: the routing weights
  // A: the core's 4 heads' q (times 1 / sqrt(D)) as tiles of 8 in-features ([tile][k][head]), the
  // weighted values as tiles of 8 of a head's 64 ([tile][head][8]); a head's sink, running
  // maximum and sum as 16 lanes of period 4 (lane 4 t + head: no scalar float work, which is
  // long code on a core), and 1 / the sum for the output ([head][8])
  alignas(64) bf16 qt[GL_D / 8][GL_AH * 8];
  alignas(64) float o2[GL_D / 8][GL_AH * 8];
  alignas(64) float sink[16];
  alignas(64) float m[16];
  alignas(64) float l[16];
  alignas(64) float inv[GL_AH * 8];
};
static_assert(sizeof(gl_state) <= 4096, "iron_gptoss_layer.py's st_t");

#ifdef GL_ATT
// 8 rows of 8 values (rows stride elements apart) as one vector, [row][8].
static inline aie::vector<bf16, 64> gl_tile(const bf16 *p, int stride) {
  return aie::concat(aie::concat(aie::load_v<8>(p), aie::load_v<8>(p + stride), aie::load_v<8>(p + 2 * stride),
                                 aie::load_v<8>(p + 3 * stride)),
                     aie::concat(aie::load_v<8>(p + 4 * stride), aie::load_v<8>(p + 5 * stride),
                                 aie::load_v<8>(p + 6 * stride), aie::load_v<8>(p + 7 * stride)));
}

// The lanes 4 t + head a token of an 8-token tile takes part with: t below n as they are
// (min with a large number), the others out of the softmax.
struct gl_mask_t {
  float v[9][GL_AH * 8];
};
constexpr gl_mask_t gl_masks() {
  gl_mask_t m{};
  for (int n = 0; n <= 8; n++)
    for (int i = 0; i < GL_AH * 8; i++)
      m.v[n][i] = i / GL_AH < n ? 3.0e38f : -1.0e30f;
  return m;
}
alignas(64) static const gl_mask_t GL_MASK = gl_masks();

// 16 lanes of period 4 from v's lanes folded by 4: each head's largest (or sum) of its lanes.
template <bool Max>
static inline aie::vector<float, 16> gl_heads(aie::vector<float, 16> v) {
  aie::vector<float, 4> r = v.template extract<4>(0);
#pragma clang loop unroll(full)
  for (int i = 1; i < 4; i++)
    r = Max ? aie::max(r, v.template extract<4>(i)) : aie::add(r, v.template extract<4>(i));
  aie::vector<float, 8> r8 = aie::concat(r, r);
  return aie::concat(r8, r8);
}

// Lanes of period 4 (4 t + head) as [head][8].
static inline aie::vector<float, 32> gl_rows(aie::vector<float, 16> v) {
  return aie::transpose(aie::concat(v, v), 8, 4);
}

// out [8 tokens][4 heads] = the scores of 8 tokens: their keys' tiles [token][8 k] (kv, 8 k
// further each step) times q's [8 k][head].
__attribute__((noinline)) static void gl_scores(const bf16 *qt, const bf16 *kv, float *out) {
  aie::mmul<8, 8, 4, bfloat16, bfloat16> c;
#pragma clang loop unroll(disable)
  for (int ks = 0; ks < GL_D / 8; ks++)
    c.mac(gl_tile(kv + 8 * ks, GL_TS), aie::load_v<GL_AH * 8>(qt + GL_AH * 8 * ks));
  aie::store_v(out, c.template to_vector<float>());
}

// out [4 heads][8 d] = the sum over the object's tiles of p's [head][8 tokens] times the values'
// [token][8 d] (kv, 8 tokens further each step).
__attribute__((noinline)) static void gl_values(const bf16 *p, const bf16 *kv, float *out) {
  aie::mmul<4, 8, 8, bfloat16, bfloat16> c;
#pragma clang loop unroll(disable)
  for (int tt = 0; tt < GL_TOK / 8; tt++)
    c.mac(aie::load_v<GL_AH * 8>(p + GL_AH * 8 * tt), gl_tile(kv + 8 * tt * GL_TS, GL_TS));
  aie::store_v(out, c.template to_vector<float>());
}

// An object's first n (of 128) tokens' keys and values (kv: [token][key 64, value 64]) into the
// core's 4 heads' softmax, 8 tokens at a time as matrix products (the bf16 mmul; no sum across
// a vector's lanes a token): a dot product a token and a head with its sum took 62 us an
// object, 12 ms a token at 2000 tokens of context.
static void gl_attend(const bf16 *kv, gl_state *st, int n) {
  constexpr int TILES = GL_TOK / 8, L = GL_AH * 8;
  alignas(64) float s[TILES][L];
  alignas(64) bf16 pb[TILES][L];
  alignas(64) float add[L];
  aie::vector<float, L> top = aie::broadcast<float, L>(-1.0e30f);
#pragma clang loop unroll(disable)
  for (int tt = 0; tt < TILES; tt++) {
    gl_scores(st->qt[0], kv + 8 * tt * GL_TS, s[tt]);
    const int left = n - 8 * tt;
    aie::vector<float, L> v = aie::min(aie::load_v<L>(s[tt]), aie::load_v<L>(GL_MASK.v[left < 0 ? 0 : left > 8 ? 8 : left]));
    aie::store_v(s[tt], v);
    top = aie::max(top, v);
  }
  const aie::vector<float, 16> old = aie::load_v<16>(st->m);
  const aie::vector<float, 16> m =
      aie::max(old, gl_heads<true>(aie::max(top.template extract<16>(0), top.template extract<16>(1))));
  const aie::vector<float, 16> d = gl_exp(aie::sub(old, m)); // the old terms at the new maximum
  aie::store_v(st->m, m);
  aie::vector<float, 16> sum = aie::zeros<float, 16>();
#pragma clang loop unroll(disable)
  for (int tt = 0; tt < TILES; tt++) {
    aie::vector<float, 16> e0 = gl_exp(aie::sub(aie::load_v<16>(s[tt]), m));
    aie::vector<float, 16> e1 = gl_exp(aie::sub(aie::load_v<16>(s[tt] + 16), m));
    sum = aie::add(sum, aie::add(e0, e1));
    aie::accum<accfloat, L> a;
    a.from_vector(aie::concat(e0, e1));
    aie::store_v(pb[tt], aie::transpose(a.template to_vector<bf16>(), 8, 4)); // [token][head] to [head][token]
  }
  aie::store_v(st->l, aie::add(aie::mul(aie::load_v<16>(st->l), d).template to_vector<float>(), gl_heads<false>(sum)));
  const aie::vector<float, L> drow = gl_rows(d);
#pragma clang loop unroll(disable)
  for (int ds = 0; ds < GL_D / 8; ds++) {
    gl_values(pb[0], kv + GL_D + 8 * ds, add);
    aie::vector<float, L> o = aie::mul(aie::load_v<L>(st->o2[ds]), drow).template to_vector<float>();
    aie::store_v(st->o2[ds], aie::add(o, aie::load_v<L>(add)));
  }
}
#endif

#ifdef GL_ROUTER
struct rt_blk_t {
  bf16 w[GL_RKT][MT]; // [in-feature][out-feature]
  uint8_t pad[sizeof(mx4_blk_t) - GL_RKT * MT * 2 - MT * 2];
  bf16 bias[MT]; // k-block 0's (zeros in the others)
};
#endif

#ifdef GL_ROUTE
// moe_route.cc's routing of 32 logits to the top 4: their softmax into w, and the scratchpad
// words the program gives (word s: ids[p0] * p1 + p2, shifted by p3) into image.
static inline void gl_route(const float *logit, const int32_t *program, bf16 *w, int32_t *image) {
  alignas(64) bf16 lv[32];
  alignas(64) float lane[16], e[16];
  int32_t id[4];
  float v[4];
  for (int i = 0; i < 32; i += 16) {
    aie::accum<accfloat, 16> a;
    a.from_vector(aie::load_v<16>(logit + i));
    aie::store_v(lv + i, a.template to_vector<bf16>());
  }
  for (int k = 0; k < 4; k++) {
    const aie::vector<bf16, 32> m = aie::load_v<32>(lv);
    const bf16 best = aie::reduce_max(m);
    const uint32_t hit = aie::eq(m, aie::broadcast<bf16, 32>(best)).to_uint32();
    const int at = __builtin_ctz(hit);
    id[k] = at;
    v[k] = (float)best;
    lv[at] = (bf16)(-__builtin_inff());
  }
  for (int i = 0; i < 16; i++)
    lane[i] = -100.0f;
  for (int k = 0; k < 4; k++)
    lane[k] = v[k] - v[0];
  aie::store_v(e, gl_exp(aie::load_v<16>(lane)));
  const float sum = e[0] + e[1] + e[2] + e[3];
  for (int i = 0; i < 16; i++)
    w[i] = (bf16)0.0f;
  for (int k = 0; k < 4; k++)
    w[k] = (bf16)(e[k] / sum);
  for (int s = 0; s < GL_FED; s++) {
    const int32_t *p = program + 4 * s;
    const int32_t x = p[0] < 0 ? p[2] : id[p[0]] * p[1] + p[2];
    image[s] = x << p[3];
  }
}
#endif

// The entries are named for the core's variant (GL_TAG: its GL_ATT, GL_ROUTER, GL_ROUTE): a
// design's kernels of one name are one object's.
#ifndef GL_TAG
#define GL_TAG base
#endif
#define GL_CAT2(a, b) a##_##b
#define GL_CAT(a, b) GL_CAT2(a, b)
#define GL_FN(n) GL_CAT(GL_TAG, n)

extern "C" {

// The counts of the step's phase ph (kinds: 4 bits a phase, bit 28 a sliding layer's attention;
// ntok the tokens a full layer's attention reads this call, nslide a sliding one's).
GL_COLD void GL_FN(gl_plan)(int32_t *cnt, int32_t ph, int32_t kinds, int32_t ntok, int32_t nslide) {
  // by kind: out-blocks, weight blocks an out-block, objects out an out-block, objects out at
  // the end, whether column 0's alone (a table: an if a kind costs the program more)
  static const uint8_t plan[8][5] = {
      {0, 0, 0, 0, 0},
      {3072 / MT / (2 * GL_CORES), 2 * GL_KBE, 1, 0, 0},                     // E1: gate-up pairs a core
      {4 * GL_OUTS, GL_KBE, 0, GL_OUTS, 0},                                   // E2: each expert's out-blocks
      {5120 / MT / 8 / GL_CORES, GL_QK / GL_QKT, 1, 0, 0},                    // Q
      {0, 0, 0, 2 * GL_AH, 0},                                                // A: a head's 64 in two objects
      {GL_OUTS, (GL_OK + GL_QKT - 1) / GL_QKT, 1, 0, 0},                      // O
      {1, GL_RKB, 2, 0, 1},                                                   // R: 32 partial sums as float32
      {0, 0, 0, 2, 1},                                                        // T: the words, the weights
  };
  const int kind = (kinds >> (4 * ph)) & 7;
  const uint8_t *p = plan[kind];
  const int n = (kinds >> 28) & 1 ? nslide : ntok;
  cnt[0] = kind;
  cnt[1] = p[0];
  cnt[2] = p[1];
  cnt[3] = p[2];
  cnt[4] = kind == K_A ? (n + GL_TOK - 1) / GL_TOK : 0;
  cnt[5] = p[3];
  cnt[8] = !p[4] || cnt[6] == 0;
  cnt[10] = n;
}

GL_COLD void GL_FN(gl_begin)(bf16 *x, float *state, const int32_t *cnt) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  gl_state *st = reinterpret_cast<gl_state *>(state);
  const int kind = cnt[0];
  if (kind == K_E1 || kind == K_Q || kind == K_R) {
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
#ifdef GL_ATT
  else if (kind == K_A) {
    const bf16 *q = x + GL_AH * cnt[7] * GL_D; // the core's first head of the column's 8
    for (int ks = 0; ks < GL_D / 8; ks++) { // q's tile [head][8 k] as [k][head]
      aie::vector<bf16, GL_AH * 8> t = aie::concat(aie::load_v<8>(q + 8 * ks), aie::load_v<8>(q + GL_D + 8 * ks),
                                                   aie::load_v<8>(q + 2 * GL_D + 8 * ks), aie::load_v<8>(q + 3 * GL_D + 8 * ks));
      aie::store_v(st->qt[ks], aie::transpose(aie::mul(t, bf16(0.125f)).template to_vector<bf16>(), 4, 8));
    }
    aie::accum<accfloat, 16> sk;
    sk.from_vector(aie::load_unaligned_v<16>(x + GL_SINK + GL_AH * cnt[7]));
    const aie::vector<float, 4> s4 = sk.template to_vector<float>().template extract<4>(0);
    const aie::vector<float, 8> s8 = aie::concat(s4, s4);
    aie::store_v(st->sink, aie::concat(s8, s8));
    aie::store_v(st->m, aie::broadcast<float, 16>(-1.0e30f));
    aie::store_v(st->l, aie::zeros<float, 16>());
    for (int ds = 0; ds < GL_D / 8; ds++)
      zero_256<float, GL_AH * 8>(st->o2[ds]);
  }
#endif
#ifdef GL_ROUTE
  else if (kind == K_T) {
    // each core's partial sums (two objects: [object][core][16 float32]) added, then routed
    alignas(64) float logit[32];
    const float *part = reinterpret_cast<const float *>(x);
    for (int q = 0; q < 2; q++) {
      aie::vector<float, 16> v = aie::load_v<16>(part + q * 64);
      for (int c = 1; c < GL_CORES; c++)
        v = aie::add(v, aie::load_v<16>(part + q * 64 + c * 16));
      aie::store_v(logit + 16 * q, v);
    }
    gl_route(logit, reinterpret_cast<const int32_t *>(x + GL_PROG), st->res + 16,
             reinterpret_cast<int32_t *>(st->res));
  }
#endif
}

void GL_FN(gl_block)(const uint8_t *block, const bf16 *x, float *state, const int32_t *cnt, int32_t o, int32_t b) {
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
#ifdef GL_ROUTER
  if (kind == K_R) { // the core's k-blocks 2 b + (core - 2)
    const rt_blk_t *A = reinterpret_cast<const rt_blk_t *>(block);
    if (b == 0)
      gl_bias(A->bias, st->acc);
    const bf16 *xk = x + GL_RKT * (2 * b + cnt[7] - 2);
    aie::accum<accfloat, 16> a0, a1;
    a0.from_vector(aie::load_v<16>(st->acc));
    a1.from_vector(aie::load_v<16>(st->acc + 16));
    for (int k = 0; k < GL_RKT; k++) {
      a0 = aie::mac(a0, aie::load_v<16>(A->w[k]), xk[k]);
      a1 = aie::mac(a1, aie::load_v<16>(A->w[k] + 16), xk[k]);
    }
    aie::store_v(st->acc, a0.template to_vector<float>());
    aie::store_v(st->acc + 16, a1.template to_vector<float>());
    return;
  }
#else
  if (kind == K_R)
    return;
#endif
  const q8x_blk_t *A = reinterpret_cast<const q8x_blk_t *>(block);
  if (b == 0)
    gl_bias(A->bias, st->acc);
  q8x_accumulate(A, x + GL_QKT * b, st->acc);
}

GL_COLD void GL_FN(gl_emit)(float *state, const bf16 *x, const int32_t *cnt, int32_t o, int32_t q, bf16 *y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  gl_state *st = reinterpret_cast<gl_state *>(state);
  const int kind = cnt[0];
  if (kind == K_E1) {
    gl_glu(st->g, st->u, y);
  } else if (kind == K_O) {
    gl_add_out(st->acc, st->res + (GL_CORES * o + cnt[7]) * MT, y);
  } else if (kind == K_R) { // 16 of the partial sums as they are (float32); zeros from a core without
#ifdef GL_ROUTER
    aie::store_v(reinterpret_cast<float *>(y), aie::load_v<16>(st->acc + 16 * q));
#else
    aie::store_v(y, aie::zeros<bf16, 32>());
#endif
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

// A: object b of the group's keys and values.
void GL_FN(gl_xblock)(const bf16 *xb, float *state, const int32_t *cnt, int32_t b) {
#ifdef GL_ATT
  aie::set_rounding(aie::rounding_mode::conv_even);
  const int left = cnt[10] - GL_TOK * b;
  gl_attend(xb, reinterpret_cast<gl_state *>(state), left < GL_TOK ? left : GL_TOK);
#endif
}

// An object out at the phase's end. E2: the core's out-block q of the layer's output, tot +
// residual. A: half a head's output, o / (l + exp(sink - m)) (the sink joins the softmax here).
// T: the routed words, then the weights.
GL_COLD void GL_FN(gl_tail)(float *state, const int32_t *cnt, int32_t q, bf16 *y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  gl_state *st = reinterpret_cast<gl_state *>(state);
  const int kind = cnt[0];
  if (kind == K_E2) {
    gl_add_out(st->tot[q], st->res + (GL_CORES * q + cnt[7]) * MT, y);
    return;
  }
#ifdef GL_ATT
  if (kind == K_A) {
    if (q == 0) { // 1 / (l + exp(sink - m)) at the larger of m and the sink, a head's 8 lanes
      const aie::vector<float, 16> m = aie::load_v<16>(st->m), sk = aie::load_v<16>(st->sink);
      const aie::vector<float, 16> top = aie::max(m, sk);
      const aie::vector<float, 16> t0 = gl_exp(aie::sub(m, top)), t1 = gl_exp(aie::sub(sk, top));
      const aie::vector<float, 16> sum = aie::add(aie::mul(aie::load_v<16>(st->l), t0).template to_vector<float>(), t1);
      aie::store_v(st->inv, gl_rows(aie::mul(t0, aie::inv(sum)).template to_vector<float>()));
    }
    const int j = q >> 1;
    alignas(64) float out[MT];
    for (int i = 0; i < 4; i++) // the half's 4 tiles of the head's values
      aie::store_v(out + 8 * i, aie::mul(aie::load_v<8>(st->o2[4 * (q & 1) + i] + 8 * j), aie::load_v<8>(st->inv + 8 * j))
                                    .template to_vector<float>());
    narrow_to_bf16<MT>(y, out);
    return;
  }
#endif
#ifdef GL_ROUTE
  if (kind == K_T) {
    aie::store_v(y, aie::load_v<16>(st->res + 16 * q));
    aie::store_v(y + 16, aie::zeros<bf16, 16>());
    return;
  }
#endif
  aie::store_v(y, aie::zeros<bf16, 32>());
}

} // extern "C"
