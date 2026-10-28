# hwnpu power benchmark

2026-10-28T06:16:39Z, Mac15,13, macOS 27.0.1, 10s per phase, back-to-back calls

Idle (mean of 4): CPU 533 mW, GPU 5 mW, ANE 0 mW, combined 539 mW

| model | placement | rows (computed) | calls/s | latency µs | net CPU mW | net ANE mW | net total mW | mJ/call | µJ/computed row |
|---|---|---|---|---|---|---|---|---|---|
| mlp_small | 0 ops on the NPU, 6 elsewhere (runs on the CPU) | 1 (1) | 16796 | 59.5 | 5371 | -0 | 5370 | 0.3199 | 319.886 |
| mlp_small_ane [CPU only] | 0 ops on the NPU, 6 elsewhere (runs on the CPU) | 1 (64) | 16117 | 62.0 | 5273 | -0 | 5273 | 0.3272 | 5.112 |
| mlp_small_ane | 6 ops on the NPU, 0 elsewhere | 1 (64) | 6083 | 164.3 | 3277 | 1464 | 4740 | 0.7790 | 12.172 |

net = power while running - idle power. mJ/call = net total / calls per second.
µJ/computed row counts padded rows too: it is what each row would cost if the padding carried real requests.
