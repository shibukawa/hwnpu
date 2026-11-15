//===- mha_zero_keys.cc - keys of zeros counted, not attended, in IRON's MHA -*- C++ -*-===//
//
// A cross-attention's text rows past the prompt are zeros (Anima's conditioning: the host
// zeroes them, as the adapter's mask does), so their keys and values are zeros: each scores 0
// and adds nothing to the output, but e^{0 - m} to each query row's softmax sum. A call bounded
// to the prompt's keys (MHA's kv_valid) skips their blocks; mha_zero_keys puts their term back:
// after the last key block, l += n 2^(-m SCALE), n = KV_LEN - s_kv the keys left out. The state
// is mha.cc's (float32): the running maximum m of the raw scores at [0, B_q), the running sum l
// at [2 B_q, 3 B_q), which rescale_O divides the output by; SCALE is the kernel's own bf16 of
// scale log2 e (as kernels/mha_sinks.cc). The output is the unbounded call's, the 0 scores no
// longer in the maximum (the softmax does not change with the maximum it subtracts).
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "common/exp_f32.h"

#ifndef ZK_SCALE
#error "mha_zero_keys.cc needs -DZK_SCALE"
#endif
#ifndef ZK_KV_LEN
#error "mha_zero_keys.cc needs -DZK_KV_LEN"
#endif

alignas(64) static float x_[16], e_[16];

extern "C" void mha_zero_keys(float *state, int32_t s_kv, int32_t B_q) {
  const int32_t n = ZK_KV_LEN - s_kv;
  if (n <= 0)
    return;
  const aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  const float *m = state;
  float *l = state + 2 * B_q;
  const float count = (float)n;
  for (int32_t r0 = 0; r0 < B_q; r0 += 16) {
    for (int32_t i = 0; i < 16; i++) {
      const int32_t r = r0 + i < B_q ? r0 + i : B_q - 1;
      // 2^(-m SCALE) = e^(-m SCALE ln 2); a row no key reached (m the lowest bf16) is past
      // the queries, and its exponent is held at e^88 rather than overflowing
      float x = -m[r] * (float)ZK_SCALE * 0.6931471805599453f;
      x_[i] = x > 88.0f ? 88.0f : x;
    }
    aie::store_v(e_, exp_f32<16>(aie::accum<accfloat, 16>(aie::load_v<16>(x_))).to_vector<float>());
    for (int32_t i = 0; i < 16 && r0 + i < B_q; i++)
      l[r0 + i] = l[r0 + i] + count * e_[i];
  }
  aie::set_rounding(saved);
}
