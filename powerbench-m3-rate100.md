# hwnpu power benchmark

2026-10-28T06:13:29Z, Mac15,13, macOS 27.0.1, 10s per phase, 100 calls/s

Idle (mean of 4): CPU 587 mW, GPU 9 mW, ANE 0 mW, combined 596 mW

| model | placement | rows (computed) | calls/s | latency µs | net CPU mW | net ANE mW | net total mW | mJ/call | µJ/computed row |
|---|---|---|---|---|---|---|---|---|---|
| mlp_small | 0 ops on the NPU, 6 elsewhere (runs on the CPU) | 1 (1) | 100 | 621.7 | 9 | -0 | 5 | 0.0522 | 52.179 |
| mlp_small_ane [CPU only] | 0 ops on the NPU, 6 elsewhere (runs on the CPU) | 1 (64) | 100 | 689.7 | -56 | -0 | -61 | -0.6049 | -9.451 |
| mlp_small_ane | 6 ops on the NPU, 0 elsewhere | 1 (64) | 100 | 1117.8 | -89 | 59 | -34 | -0.3444 | -5.382 |

net = power while running - idle power. mJ/call = net total / calls per second.
µJ/computed row counts padded rows too: it is what each row would cost if the padding carried real requests.

mlp_small_ane [CPU only]: net power is not above idle; the difference is within noise (use a longer -d or more -rounds).

mlp_small_ane: net power is not above idle; the difference is within noise (use a longer -d or more -rounds).
