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
#ifndef ROUTE_BIAS
#define ROUTE_BIAS 0
#endif

static_assert(ROUTE_TOP < 16 && ROUTE_W >= ROUTE_TOP + 1, "moe_route: up to 15 chosen");
static_assert(ROUTE_EXPERTS % 32 == 0, "moe_route: experts in whole vectors of 32");

// Kept off the stack (a core's is 1 KiB).
alignas(64) static bfloat16 lv[ROUTE_EXPERTS];
alignas(64) static float lane[16], e[16];

extern "C" void moe_route(const bfloat16 *restrict logits,
                          const int32_t *restrict program,
                          bfloat16 *restrict w, int32_t *restrict image) {
  const aie::rounding_mode saved =
      aie::swap_rounding(aie::rounding_mode::conv_even);
  int32_t id[ROUTE_TOP];
  float v[ROUTE_TOP];
  // A copy of the routed logits, each chosen one put to -inf: per pick, the
  // largest by a vector max, then the first lane holding it (ties to the
  // lower index).
  constexpr int N = 32;
  using vec = aie::vector<bfloat16, N>;
#if ROUTE_BIAS
  // the router's bias (gpt-oss's), as bf16 after the program's words: added here, not by an
  // operator of its own; the logits are the GEMV's bf16, the sum rounded to bf16 as an add's
  const bfloat16 *bias = reinterpret_cast<const bfloat16 *>(program + 4 * ROUTE_SLOTS);
  for (int i = 0; i < ROUTE_EXPERTS; i += N)
    aie::store_v(lv + i, aie::add(aie::load_v<N>(logits + i), aie::load_v<N>(bias + i)));
#else
  for (int i = 0; i < ROUTE_EXPERTS; i += N)
    aie::store_v(lv + i, aie::load_v<N>(logits + i));
#endif
  for (int k = 0; k < ROUTE_TOP; k++) {
    vec m = aie::load_v<N>(lv);
    for (int i = N; i < ROUTE_EXPERTS; i += N)
      m = aie::max(m, aie::load_v<N>(lv + i));
    const bfloat16 best = aie::reduce_max(m);
    const vec b = aie::broadcast<bfloat16, N>(best);
    int at = 0;
    for (int i = 0; i < ROUTE_EXPERTS; i += N) {
      const uint32_t hit = aie::eq(aie::load_v<N>(lv + i), b).to_uint32();
      if (hit) {
        at = i + __builtin_ctz(hit);
        break;
      }
    }
    id[k] = at;
    v[k] = (float)best;
    lv[at] = (bfloat16)(-__builtin_inff());
  }
  // exp of each chosen logit less the largest, and of minus the shared logit
  for (int i = 0; i < 16; i++)
    lane[i] = -100.0f;
  for (int k = 0; k < ROUTE_TOP; k++)
    lane[k] = v[k] - v[0];
  lane[ROUTE_TOP] = -(float)logits[ROUTE_EXPERTS];
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
