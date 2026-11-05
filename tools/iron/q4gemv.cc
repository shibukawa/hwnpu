// q4nx matrix-vector kernels for the q4gemv benchmark: FastFlowLM's q4nx_accumulate
// (aie_kernels/flm_gemma4/q4nx.h) on one 32 x 256 block per call.
#include "q4nx.h"

using blk = q4nx_block<32, 256>;

extern "C" {

// Each symbol is built in its own object (one ExternalFunction each).
#ifdef Q4GEMV_ZERO
void q4_zero(float *y, int n) {
  const aie::vector<float, 8> z = aie::zeros<float, 8>();
  for (int i = 0; i < n; i += 8)
    aie::store_v(y + i, z);
}
#endif

#ifdef Q4GEMV_MV
// y[rb*32 .. rb*32+31] += dequantize(w) * x[cb*256 .. cb*256+255]
void q4_mv(blk *w, bf16 *x, float *y, int cb, int rb) {
  bf16 *xs = x + cb * 256;
  alignas(32) bf16 sums[8];
  group_sums_32<256>(sums, xs);
  q4nx_accumulate<32, 256>(w, xs, y + rb * 32, sums);
}
#endif
}
