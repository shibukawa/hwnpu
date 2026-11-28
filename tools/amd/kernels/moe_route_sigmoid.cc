//===- moe_route_sigmoid.cc - LFM2-MoE's routing, on the device --*- C++ -*-===//
//
// One token's router of LFM2-MoE (tools/amd/iron_lfm2moe.py's MoERouteSigmoid,
// transformers' Lfm2MoeTopKRouter): from the ROUTE_EXPERTS logits, each
// expert's score s = sigmoid(logit); the ROUTE_TOP largest of s + bias (the
// expert bias, bf16 after the program's words) are chosen; their weights are
// s / (the chosen's sum + 1e-6) times ROUTE_SCALE: w[0..TOP) the chosen's,
// zeros after (no shared expert).
//
// The chosen experts also become words of a scratchpad image (the next
// command's, which the experts' operators read their weights' byte offsets
// from), as kernels/moe_route.cc's: image[s] = (id[program[4 s]] *
// program[4 s + 1] + program[4 s + 2]) << program[4 s + 3], or program[4 s + 2]
// where program[4 s] is negative. Ties go to the lower index.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "common/exp_f32.h"

#ifndef ROUTE_EXPERTS
#define ROUTE_EXPERTS 32
#endif
#ifndef ROUTE_TOP
#define ROUTE_TOP 4
#endif
#ifndef ROUTE_SLOTS
#define ROUTE_SLOTS 32
#endif
#ifndef ROUTE_W
#define ROUTE_W 16
#endif
#ifndef ROUTE_SCALE
#define ROUTE_SCALE 1.0f
#endif

static_assert(ROUTE_TOP < 16 && ROUTE_W >= ROUTE_TOP, "moe_route_sigmoid: up to 15 chosen");
static_assert(ROUTE_EXPERTS % 16 == 0, "moe_route_sigmoid: experts in whole vectors of 16");

// Kept off the stack (a core's is 1 KiB).
alignas(64) static float sv[ROUTE_EXPERTS], pick[ROUTE_EXPERTS], lane[16];

extern "C" void moe_route_sigmoid(const bfloat16 *restrict logits,
                                  const int32_t *restrict program,
                                  bfloat16 *restrict w, int32_t *restrict image) {
  const aie::rounding_mode saved =
      aie::swap_rounding(aie::rounding_mode::conv_even);
  const bfloat16 *bias = reinterpret_cast<const bfloat16 *>(program + 4 * ROUTE_SLOTS);
  // s = 1 / (1 + exp(-logit)), 16 at a time
  for (int i = 0; i < ROUTE_EXPERTS; i += 16) {
    for (int j = 0; j < 16; j++)
      lane[j] = -(float)logits[i + j];
    aie::store_v(sv + i, exp_f32<16>(aie::accum<accfloat, 16>(aie::load_v<16>(lane)))
                             .to_vector<float>());
    for (int j = 0; j < 16; j++) {
      sv[i + j] = 1.0f / (1.0f + sv[i + j]);
      pick[i + j] = sv[i + j] + (float)bias[i + j];
    }
  }
  int32_t id[ROUTE_TOP];
  float sum = 0;
  for (int k = 0; k < ROUTE_TOP; k++) {
    int at = 0;
    for (int i = 1; i < ROUTE_EXPERTS; i++)
      if (pick[i] > pick[at])
        at = i;
    id[k] = at;
    sum += sv[at];
    pick[at] = -__builtin_inff();
  }
  for (int i = 0; i < ROUTE_W; i++)
    w[i] = (bfloat16)0.0f;
  const float norm = ROUTE_SCALE / (sum + 1e-6f);
  for (int k = 0; k < ROUTE_TOP; k++)
    w[k] = (bfloat16)(sv[id[k]] * norm);
  for (int s = 0; s < ROUTE_SLOTS; s++) {
    const int32_t *p = program + 4 * s;
    const int32_t x = p[0] < 0 ? p[2] : id[p[0]] * p[1] + p[2];
    image[s] = x << p[3];
  }
  aie::set_rounding(saved);
}
