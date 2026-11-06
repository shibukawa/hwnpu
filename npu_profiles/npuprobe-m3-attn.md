# hwnpu npuprobe

2026-11-04T10:14:01+09:00

| | |
|---|---|
| CPU | Apple M3 |
| machine | Mac15,13 |
| OS | macOS 27.0.1 |
| RAM | 16 GiB |
| NPU | Apple Neural Engine (16 cores) (coreml) |
| driver | Core ML / Version 27.0.1 (Build 26A434)  |
| hwnpu | (devel), go1.27.3 |

Models: one token through Llama 3.2 1B sized layers (61 M weights per layer, random values), so a call is the work of generating one token without the attention over the context.

## Attention per chunk of a KV cache, merged on the host

| form | Q | chunk | placement | ms/call alone | CPU use | compile s | error |
|---|---|---|---|---|---|---|---|
| grouped | 1 | 1024 | CPU 17 | 0.176 | 1.18 | 0.1 |  |
| grouped | 1 | 2048 | CPU 17 | 0.228 | 1.15 | 0.1 |  |
| grouped | 1 | 4096 | CPU 17 | 0.297 | 1.32 | 0.1 |  |
| grouped | 1 | 8192 | CPU 17 | 0.444 | 1.75 | 0.1 |  |
| grouped | 1 | 16384 | NPU | 2.083 | 0.47 | 0.3 |  |
| grouped | 8 | 1024 | CPU 17 | 0.215 | 1.20 | 0.1 |  |
| grouped | 8 | 2048 | CPU 17 | 0.282 | 1.28 | 0.1 |  |
| grouped | 8 | 4096 | NPU | 0.766 | 0.45 | 0.3 |  |
| grouped | 8 | 8192 | NPU | 1.376 | 0.42 | 0.3 |  |
| grouped | 8 | 16384 | NPU | 2.663 | 0.38 | 0.3 |  |
| grouped | 64 | 1024 | NPU | 0.697 | 0.37 | 0.3 |  |
| grouped | 64 | 2048 | NPU | 1.104 | 0.27 | 0.3 |  |
| grouped | 64 | 4096 | NPU | 1.968 | 0.23 | 0.3 |  |
| grouped | 64 | 8192 | NPU | 4.149 | 0.19 | 0.3 |  |
| grouped | 64 | 16384 | NPU | 9.180 | 0.15 | 0.3 |  |

| form | Q | chunk | placement | used | layers × chunks | ms/step | ms/call in the step | of it the merge, ms | ms per 1K used | KV | peak RSS | CPU use | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| grouped | 1 | 1024 | CPU 17 | 4096 | 6 × 4 | 4.69 | 0.195 | 0.11 | 1.173 | 48 MiB | 0.09 GiB | 1.19 |  |
| grouped | 1 | 1024 | CPU 17 | 16384 | 6 × 16 | 19.01 | 0.198 | 0.43 | 1.188 | 192 MiB | 0.23 GiB | 1.19 |  |
| grouped | 1 | 1024 | CPU 17 | 65536 | 6 × 64 | 77.58 | 0.202 | 1.70 | 1.212 | 768 MiB | 0.79 GiB | 1.18 |  |
| grouped | 1 | 1024 | CPU 17 | 262144 | 6 × 256 | 324.30 | 0.211 | 6.71 | 1.267 | 3072 MiB | 3.04 GiB | 1.17 |  |
| grouped | 1 | 2048 | CPU 17 | 4096 | 6 × 2 | 3.14 | 0.262 | 0.05 | 0.785 | 48 MiB | 0.09 GiB | 1.15 |  |
| grouped | 1 | 2048 | CPU 17 | 16384 | 6 × 8 | 12.51 | 0.261 | 0.21 | 0.782 | 192 MiB | 0.24 GiB | 1.15 |  |
| grouped | 1 | 2048 | CPU 17 | 65536 | 6 × 32 | 50.12 | 0.261 | 0.84 | 0.783 | 768 MiB | 0.80 GiB | 1.15 |  |
| grouped | 1 | 2048 | CPU 17 | 262144 | 6 × 128 | 202.99 | 0.264 | 3.34 | 0.793 | 3072 MiB | 3.05 GiB | 1.15 |  |
| grouped | 1 | 4096 | CPU 17 | 4096 | 6 × 1 | 2.26 | 0.376 | 0.03 | 0.564 | 48 MiB | 0.11 GiB | 1.30 |  |
| grouped | 1 | 4096 | CPU 17 | 16384 | 6 × 4 | 9.03 | 0.376 | 0.10 | 0.564 | 192 MiB | 0.25 GiB | 1.30 |  |
| grouped | 1 | 4096 | CPU 17 | 65536 | 6 × 16 | 36.16 | 0.377 | 0.41 | 0.565 | 768 MiB | 0.81 GiB | 1.30 |  |
| grouped | 1 | 4096 | CPU 17 | 262144 | 6 × 64 | 168.37 | 0.438 | 1.65 | 0.658 | 3072 MiB | 3.05 GiB | 1.28 |  |
| grouped | 1 | 8192 | CPU 17 | 4096 | 6 × 1 | 3.38 | 0.563 | 0.03 | 0.845 | 96 MiB | 0.18 GiB | 1.75 |  |
| grouped | 1 | 8192 | CPU 17 | 16384 | 6 × 2 | 6.59 | 0.550 | 0.05 | 0.412 | 192 MiB | 0.27 GiB | 1.77 |  |
| grouped | 1 | 8192 | CPU 17 | 65536 | 6 × 8 | 26.39 | 0.550 | 0.21 | 0.412 | 768 MiB | 0.83 GiB | 1.77 |  |
| grouped | 1 | 8192 | CPU 17 | 262144 | 6 × 32 | 115.77 | 0.603 | 0.84 | 0.452 | 3072 MiB | 3.08 GiB | 1.73 |  |
| grouped | 1 | 16384 | NPU | 4096 | 6 × 1 | 12.74 | 2.123 | 0.02 | 3.185 | 192 MiB | 0.35 GiB | 0.48 |  |
| grouped | 1 | 16384 | NPU | 16384 | 6 × 1 | 12.70 | 2.117 | 0.02 | 0.794 | 192 MiB | 0.35 GiB | 0.48 |  |
| grouped | 1 | 16384 | NPU | 65536 | 6 × 4 | 51.04 | 2.126 | 0.10 | 0.797 | 768 MiB | 0.91 GiB | 0.48 |  |
| grouped | 1 | 16384 | NPU | 262144 | 6 × 16 | 217.51 | 2.266 | 0.39 | 0.850 | 3072 MiB | 3.15 GiB | 0.50 |  |
| grouped | 8 | 1024 | CPU 17 | 4096 | 6 × 4 | 6.95 | 0.290 | 0.77 | 1.737 | 48 MiB | 0.09 GiB | 1.17 |  |
| grouped | 8 | 1024 | CPU 17 | 16384 | 6 × 16 | 27.53 | 0.287 | 3.10 | 1.721 | 192 MiB | 0.23 GiB | 1.17 |  |
| grouped | 8 | 1024 | CPU 17 | 65536 | 6 × 64 | 110.20 | 0.287 | 12.39 | 1.722 | 768 MiB | 0.79 GiB | 1.17 |  |
| grouped | 8 | 1024 | CPU 17 | 262144 | 6 × 256 | 546.45 | 0.356 | 49.31 | 2.135 | 3072 MiB | 3.04 GiB | 1.12 |  |
| grouped | 8 | 2048 | CPU 17 | 4096 | 6 × 2 | 4.76 | 0.397 | 0.39 | 1.190 | 48 MiB | 0.10 GiB | 1.22 |  |
| grouped | 8 | 2048 | CPU 17 | 16384 | 6 × 8 | 18.88 | 0.393 | 1.54 | 1.180 | 192 MiB | 0.24 GiB | 1.22 |  |
| grouped | 8 | 2048 | CPU 17 | 65536 | 6 × 32 | 75.19 | 0.392 | 6.17 | 1.175 | 768 MiB | 0.80 GiB | 1.22 |  |
| grouped | 8 | 2048 | CPU 17 | 262144 | 6 × 128 | 325.82 | 0.424 | 24.54 | 1.273 | 3072 MiB | 3.05 GiB | 1.19 |  |
| grouped | 8 | 4096 | NPU | 4096 | 6 × 1 | 5.07 | 0.845 | 0.18 | 1.268 | 48 MiB | 0.11 GiB | 0.52 |  |
| grouped | 8 | 4096 | NPU | 16384 | 6 × 4 | 20.12 | 0.838 | 0.71 | 1.257 | 192 MiB | 0.26 GiB | 0.51 |  |
| grouped | 8 | 4096 | NPU | 65536 | 6 × 16 | 80.70 | 0.841 | 2.82 | 1.261 | 768 MiB | 0.82 GiB | 0.51 |  |
| grouped | 8 | 4096 | NPU | 262144 | 6 × 64 | 392.19 | 1.021 | 11.98 | 1.532 | 3072 MiB | 3.07 GiB | 0.58 |  |
| grouped | 8 | 8192 | NPU | 4096 | 6 × 1 | 8.75 | 1.458 | 0.18 | 2.187 | 96 MiB | 0.19 GiB | 0.45 |  |
| grouped | 8 | 8192 | NPU | 16384 | 6 × 2 | 17.40 | 1.450 | 0.35 | 1.087 | 192 MiB | 0.29 GiB | 0.45 |  |
| grouped | 8 | 8192 | NPU | 65536 | 6 × 8 | 69.50 | 1.448 | 1.42 | 1.086 | 768 MiB | 0.85 GiB | 0.45 |  |
| grouped | 8 | 8192 | NPU | 262144 | 6 × 32 | 295.72 | 1.540 | 5.68 | 1.155 | 3072 MiB | 3.10 GiB | 0.47 |  |
| grouped | 8 | 16384 | NPU | 4096 | 6 × 1 | 24.51 | 4.085 | 0.25 | 6.128 | 192 MiB | 0.35 GiB | 0.37 |  |
| grouped | 8 | 16384 | NPU | 16384 | 6 × 1 | 20.49 | 3.415 | 0.24 | 1.281 | 192 MiB | 0.35 GiB | 0.43 |  |
| grouped | 8 | 16384 | NPU | 65536 | 6 × 4 | 69.70 | 2.904 | 0.75 | 1.089 | 768 MiB | 0.84 GiB | 0.42 |  |
| grouped | 8 | 16384 | NPU | 262144 | 6 × 16 | 321.06 | 3.344 | 3.32 | 1.254 | 3072 MiB | 3.08 GiB | 0.48 |  |
| grouped | 64 | 1024 | NPU | 4096 | 6 × 4 | 24.18 | 1.008 | 5.77 | 6.046 | 48 MiB | 0.09 GiB | 0.54 |  |
| grouped | 64 | 1024 | NPU | 16384 | 6 × 16 | 91.45 | 0.953 | 22.40 | 5.715 | 192 MiB | 0.23 GiB | 0.53 |  |
| grouped | 64 | 1024 | NPU | 65536 | 6 × 64 | 364.46 | 0.949 | 89.60 | 5.695 | 768 MiB | 0.80 GiB | 0.53 |  |
| grouped | 64 | 1024 | NPU | 262144 | 6 × 256 | 1546.69 | 1.007 | 360.34 | 6.042 | 3072 MiB | 3.05 GiB | 0.56 |  |
| grouped | 64 | 2048 | NPU | 4096 | 6 × 2 | 16.79 | 1.399 | 2.80 | 4.196 | 48 MiB | 0.10 GiB | 0.43 |  |
| grouped | 64 | 2048 | NPU | 16384 | 6 × 8 | 66.31 | 1.381 | 11.21 | 4.144 | 192 MiB | 0.24 GiB | 0.42 |  |
| grouped | 64 | 2048 | NPU | 65536 | 6 × 32 | 264.38 | 1.377 | 44.74 | 4.131 | 768 MiB | 0.80 GiB | 0.42 |  |
| grouped | 64 | 2048 | NPU | 262144 | 6 × 128 | 1044.69 | 1.360 | 180.55 | 4.081 | 3072 MiB | 3.05 GiB | 0.42 |  |
| grouped | 64 | 4096 | NPU | 4096 | 6 × 1 | 14.42 | 2.404 | 1.47 | 3.606 | 48 MiB | 0.12 GiB | 0.35 |  |
| grouped | 64 | 4096 | NPU | 16384 | 6 × 4 | 54.89 | 2.287 | 5.97 | 3.431 | 192 MiB | 0.26 GiB | 0.33 |  |
| grouped | 64 | 4096 | NPU | 65536 | 6 × 16 | 217.60 | 2.267 | 23.71 | 3.400 | 768 MiB | 0.82 GiB | 0.34 |  |
| grouped | 64 | 4096 | NPU | 262144 | 6 × 64 | 1369.09 | 3.565 | 124.45 | 5.348 | 3072 MiB | 3.04 GiB | 0.47 |  |
| grouped | 64 | 8192 | NPU | 4096 | 6 × 1 | 28.42 | 4.737 | 1.92 | 7.105 | 96 MiB | 0.20 GiB | 0.27 |  |
| grouped | 64 | 8192 | NPU | 16384 | 6 × 2 | 55.57 | 4.631 | 3.88 | 3.473 | 192 MiB | 0.29 GiB | 0.26 |  |
| grouped | 64 | 8192 | NPU | 65536 | 6 × 8 | 224.33 | 4.674 | 15.40 | 3.505 | 768 MiB | 0.85 GiB | 0.26 |  |
| grouped | 64 | 8192 | NPU | 262144 | 6 × 32 | 906.57 | 4.722 | 62.55 | 3.541 | 3072 MiB | 3.10 GiB | 0.27 |  |
| grouped | 64 | 16384 | NPU | 4096 | 6 × 1 | 58.38 | 9.730 | 1.58 | 14.595 | 192 MiB | 0.36 GiB | 0.19 |  |
| grouped | 64 | 16384 | NPU | 16384 | 6 × 1 | 56.82 | 9.470 | 1.47 | 3.551 | 192 MiB | 0.36 GiB | 0.18 |  |
| grouped | 64 | 16384 | NPU | 65536 | 6 × 4 | 223.04 | 9.293 | 5.78 | 3.485 | 768 MiB | 0.92 GiB | 0.16 |  |
| grouped | 64 | 16384 | NPU | 262144 | 6 × 16 | 1285.65 | 13.392 | 24.94 | 5.022 | 3072 MiB | 3.17 GiB | 0.32 |  |

The program is attention without weights over one chunk of the cache (keys and values of 4 kv heads × 128, 16 query heads, Q query rows), returning the chunk's normalized output, largest scores and sums of exps. A step calls it once per chunk holding the used tokens in each layer, every chunk bound from buffers of its own, and merges the outputs on the host in float32. ms/call alone binds one chunk; ms/call in the step is a step's time over its calls, the merge included, so a difference between the two is the cost of binding many buffers. Peak RSS is the measuring process's largest resident memory after the step (it leaves out Core ML's and the Neural Engine's own processes); a peak well over the KV bytes means the inputs were copied.

Placement: Apple, Core ML's compute plan (MLComputePlan) for every operation; Intel, NPU when the
driver compiler built the graph (it has no fallback). CPU use is the process CPU time divided by
the wall time while the model runs: near 0 when the NPU does the work, 1 or more when the CPU
does. weights GB/s is the weight bytes times calls per second. net W is the device's power meter
while running minus idle, when it has one.
