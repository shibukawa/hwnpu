//===- mha_sinks.cc - attention sinks in IRON's MHA, at the division -*- C++ -*-===//
//
// gpt-oss's attention adds a learned logit per head (a "sink") to each query's softmax and
// drops it from the output: o = sum_j e^{s_j} v_j / (sum_j e^{s_j} + e^{sink}). IRON's MHA
// keeps each query row's online softmax in its scale buffer (mha.cc, float32): the running
// maximum m of the raw scores (q . k, before the scale) at [0, B_q) and the running sum l at
// [2 B_q, 3 B_q), which rescale_O divides the output by; an exponent is (s - m) SINK_SCALE in
// log2 units, SINK_SCALE the kernel's own bf16 of scale log2 e. After the last key block,
// mha_sinks(state, table, h, p, B_q) adds each row's sink term to l: l += 2^(t - m SINK_SCALE),
// t the head's sink log2 e (table, float32). The scores' own maximum stays the softmax's m: seeding m
// with a large sink instead put every score's exponent far below zero, where the device's
// exp2 is up to twice too high (outputs 1.5 times too large at a sink of 6).
//
// h is the P*V core's head loop index, p its pipeline. A prompt block's rows are one head's
// queries (head h); a decode step packs a KV group's heads as a block's rows, repeated
// (head (p steps + h) group + row % group).
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "common/exp_f32.h"

#ifndef SINK_PACKED
#define SINK_PACKED 0
#endif
#ifndef SINK_GROUP
#define SINK_GROUP 8
#endif
#ifndef SINK_STEPS
#define SINK_STEPS 1
#endif
#ifndef SINK_SCALE
#error "mha_sinks.cc needs -DSINK_SCALE"
#endif

alignas(64) static float x_[16], e_[16];

extern "C" void mha_sinks(float *state, const float *table, int32_t h, int32_t p, int32_t B_q) {
  const aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  const float *m = state;
  float *l = state + 2 * B_q;
  for (int32_t r0 = 0; r0 < B_q; r0 += 16) {
    for (int32_t i = 0; i < 16; i++) {
      const int32_t r = r0 + i < B_q ? r0 + i : B_q - 1;
#if SINK_PACKED
      const int32_t head = (p * SINK_STEPS + h) * SINK_GROUP + r % SINK_GROUP;
#else
      const int32_t head = h;
      (void)p;
#endif
      // 2^(t - m) = e^((t - m) ln 2)
      x_[i] = (table[head] - m[r] * (float)SINK_SCALE) * 0.6931471805599453f;
    }
    aie::store_v(e_, exp_f32<16>(aie::accum<accfloat, 16>(aie::load_v<16>(x_))).to_vector<float>());
    for (int32_t i = 0; i < 16 && r0 + i < B_q; i++)
      l[r0 + i] = l[r0 + i] + e_[i];
  }
  aie::set_rounding(saved);
}
