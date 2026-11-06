//===- moe_route.cc - a mixture of experts' routing, on the device -*- C++ -*-===//
//
// One token's router (tools/amd/iron_moe.py's MoERoute): from its logits
// (ROUTE_EXPERTS routed experts', then the shared expert's gate logit), the
// ROUTE_TOP largest, their softmax weights over the chosen alone (the softmax
// over all, renormalized over the chosen, as transformers' Qwen3.5-MoE router),
// and sigmoid of the shared logit, as MoEGateUp's routing weights: w[0..TOP)
// the chosen's, w[TOP] the shared expert's, zeros after.
//
// The chosen experts also become words of a scratchpad image (the next
// command's, which the experts' operators read their weights' byte offsets
// from): image[s] = (id[program[4 s]] * program[4 s + 1] + program[4 s + 2])
// << program[4 s + 3], or program[4 s + 2] where program[4 s] is negative.
// Ties go to the lower index (torch.topk's order may differ on exact ties).
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "common/exp_f32.h"

#ifndef ROUTE_EXPERTS
#define ROUTE_EXPERTS 256
#endif
#ifndef ROUTE_TOP
#define ROUTE_TOP 8
#endif
#ifndef ROUTE_SLOTS
#define ROUTE_SLOTS 32
#endif
#ifndef ROUTE_W
#define ROUTE_W 16
#endif

static_assert(ROUTE_TOP < 16 && ROUTE_W >= ROUTE_TOP + 1, "moe_route: up to 15 chosen");

extern "C" void moe_route(const bfloat16 *restrict logits,
                          const int32_t *restrict program,
                          bfloat16 *restrict w, int32_t *restrict image) {
  const aie::rounding_mode saved =
      aie::swap_rounding(aie::rounding_mode::conv_even);
  int32_t id[ROUTE_TOP];
  float v[ROUTE_TOP];
  for (int k = 0; k < ROUTE_TOP; k++) {
    int best = -1;
    float bv = 0;
    for (int e = 0; e < ROUTE_EXPERTS; e++) {
      bool taken = false;
      for (int j = 0; j < k; j++)
        taken |= id[j] == e;
      const float x = (float)logits[e];
      if (!taken && (best < 0 || x > bv)) {
        best = e;
        bv = x;
      }
    }
    id[k] = best;
    v[k] = bv;
  }
  // exp of each chosen logit less the largest, and of minus the shared logit
  alignas(64) float lane[16];
  for (int i = 0; i < 16; i++)
    lane[i] = -100.0f;
  for (int k = 0; k < ROUTE_TOP; k++)
    lane[k] = v[k] - v[0];
  lane[ROUTE_TOP] = -(float)logits[ROUTE_EXPERTS];
  alignas(64) float e[16];
  aie::store_v(e, exp_f32<16>(aie::accum<accfloat, 16>(aie::load_v<16>(lane)))
                      .to_vector<float>());
  float sum = 0;
  for (int k = 0; k < ROUTE_TOP; k++)
    sum += e[k];
  for (int i = 0; i < ROUTE_W; i++)
    w[i] = (bfloat16)0.0f;
  for (int k = 0; k < ROUTE_TOP; k++)
    w[k] = (bfloat16)(e[k] / sum);
  w[ROUTE_TOP] = (bfloat16)(1.0f / (1.0f + e[ROUTE_TOP]));
  for (int s = 0; s < ROUTE_SLOTS; s++) {
    const int32_t *p = program + 4 * s;
    const int32_t x = p[0] < 0 ? p[2] : id[p[0]] * p[1] + p[2];
    image[s] = x << p[3];
  }
  aie::set_rounding(saved);
}
