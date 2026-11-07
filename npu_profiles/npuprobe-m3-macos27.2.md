# hwnpu npuprobe

2026-11-07T12:43:50+09:00

| | |
|---|---|
| CPU | Apple M3 |
| machine | Mac15,13 |
| OS | macOS 27.2 |
| RAM | 16 GiB |
| NPU | Apple Neural Engine (16 cores) (coreml) |
| driver | Core ML / Version 27.2 (Build 26B123)  |
| hwnpu | v0.0.0-20261107004622-bbe1402d3b27 bbe1402d3b278cb00550cf9a7ef8f0ac05ea7dd7, go1.27.3 |

Models: one token through Llama 3.2 1B sized layers (61 M weights per layer, random values), so a call is the work of generating one token without the attention over the context.

## Speed

| weights | format | unit | placement | ms/token | tokens/s | weights GB/s | CPU use | net W | mJ/token | compile s | error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.49 B, 0.91 GiB | fp16 | NPU | NPU | 45.6 | 21.9 | 21 | 0.03 |  |  | 6.2 |  |
| 0.49 B, 0.91 GiB | fp16 | CPU | CPU 130 | 14.0 | 71.6 | 70 | 3.64 |  |  | 4.0 |  |
| 0.49 B, 0.91 GiB | fp16 | GPU | GPU 130 | 13.4 | 74.5 | 72 | 0.25 |  |  | 4.9 |  |
| 0.49 B, 0.45 GiB | int8 | NPU | NPU | 21.3 | 46.9 | 23 | 0.01 |  |  | 3.6 |  |
| 0.49 B, 0.45 GiB | int8 | CPU | CPU 130 | 13.9 | 72.2 | 35 | 3.68 |  |  | 2.3 |  |
| 0.49 B, 0.45 GiB | int8 | GPU | GPU 130 | 22.6 | 44.3 | 22 | 0.18 |  |  | 1.8 |  |
| 0.49 B, 0.23 GiB | lut4 | NPU | NPU | 3.9 | 254.7 | 62 | 0.04 |  |  | 3.1 |  |
| 0.49 B, 0.23 GiB | lut4 | CPU | CPU 130 | 13.8 | 72.4 | 18 | 3.61 |  |  | 3.6 |  |
| 0.49 B, 0.23 GiB | lut4 | GPU | GPU 130 | 28.3 | 35.4 | 9 | 0.21 |  |  | 1.1 |  |
| 0.49 B, 0.23 GiB | int4 | CPU | CPU 130 | 14.0 | 71.6 | 17 | 3.51 |  |  | 2.4 |  |
| 0.49 B, 0.23 GiB | int4 | GPU | GPU 130 | 6.7 | 150.0 | 36 | 0.39 |  |  | 1.2 |  |
| 0.97 B, 1.81 GiB | fp16 | NPU | CPU 258 | 27.3 | 36.7 | 71 | 3.70 |  |  | 9.2 |  |
| 0.97 B, 1.81 GiB | fp16 | CPU | CPU 258 | 27.0 | 37.0 | 72 | 3.74 |  |  | 9.5 |  |
| 0.97 B, 1.81 GiB | fp16 | GPU | GPU 258 | 25.3 | 39.6 | 77 | 0.37 |  |  | 10.0 |  |
| 0.97 B, 0.91 GiB | int8 | NPU | NPU | 42.2 | 23.7 | 23 | 0.01 |  |  | 6.8 |  |
| 0.97 B, 0.91 GiB | int8 | CPU | CPU 258 | 26.6 | 37.6 | 37 | 3.83 |  |  | 5.3 |  |
| 0.97 B, 0.91 GiB | int8 | GPU | GPU 258 | 43.7 | 22.9 | 22 | 0.23 |  |  | 4.2 |  |
| 0.97 B, 0.45 GiB | lut4 | NPU | NPU | 7.6 | 130.9 | 64 | 0.02 |  |  | 6.5 |  |
| 0.97 B, 0.45 GiB | lut4 | CPU | CPU 258 | 26.8 | 37.3 | 18 | 3.77 |  |  | 7.3 |  |
| 0.97 B, 0.45 GiB | lut4 | GPU | GPU 258 | 52.5 | 19.0 | 9 | 0.10 |  |  | 2.6 |  |
| 0.97 B, 0.45 GiB | int4 | CPU | CPU 258 | 26.9 | 37.2 | 18 | 3.80 |  |  | 9.5 |  |
| 0.97 B, 0.45 GiB | int4 | GPU | GPU 258 | 9.4 | 106.1 | 52 | 0.45 |  |  | 2.3 |  |

Placement: Apple, Core ML's compute plan (MLComputePlan) for every operation; Intel, NPU when the
driver compiler built the graph (it has no fallback). CPU use is the process CPU time divided by
the wall time while the model runs: near 0 when the NPU does the work, 1 or more when the CPU
does. weights GB/s is the weight bytes times calls per second. net W is the device's power meter
while running minus idle, when it has one.
