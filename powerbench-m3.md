# hwnpu power benchmark

2026-10-28T06:08:07Z, Mac15,13, macOS 27.0.1, 10s per phase, back-to-back calls

Idle (mean of 4): CPU 279 mW, GPU 6 mW, ANE 0 mW, combined 285 mW

| model | placement | rows (computed) | calls/s | latency µs | net CPU mW | net ANE mW | net total mW | mJ/call | µJ/computed row |
|---|---|---|---|---|---|---|---|---|---|
| mlp_small | 0 ops on the NPU, 6 elsewhere (runs on the CPU) | 1 (1) | 16693 | 59.9 | 5135 | -0 | 5134 | 0.3076 | 307.556 |
| mlp_small_ane | 6 ops on the NPU, 0 elsewhere | 1 (64) | 6029 | 165.8 | 3304 | 1466 | 4770 | 0.7912 | 12.362 |

net = power while running - idle power. mJ/call = net total / calls per second.
µJ/computed row counts padded rows too: it is what each row would cost if the padding carried real requests.
