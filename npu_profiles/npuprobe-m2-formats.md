# hwnpu-dev probe

2026-11-20T00:22:27+09:00

| | |
|---|---|
| CPU | Apple M2 |
| machine | Mac14,2 |
| OS | macOS 27.2.1 |
| RAM | 16 GiB |
| NPU | Apple Neural Engine (16 cores) (coreml) |
| driver | Core ML / Version 27.2.1 (Build 26B131)  |
| hwnpu | (devel), go1.27.4 |

Models: one token through Llama 3.2 1B sized layers (61 M weights per layer, random values), so a call is the work of generating one token without the attention over the context.

## Speed

| weights | format | unit | placement | ms/token | tokens/s | weights GB/s | G weights/s | CPU use | net W | mJ/token | compile s | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.49 B, 0.91 GiB | fp16 | NPU | NPU | 15.9 | 62.7 | 61 | 31 | 0.03 |  |  | 10.9 |  |
| 0.49 B, 0.91 GiB | fp16 | CPU | CPU 130 | 19.1 | 52.3 | 51 | 25 | 3.63 |  |  | 5.2 |  |
| 0.49 B, 0.91 GiB | fp16 | GPU | GPU 130 | 15.0 | 66.7 | 65 | 32 | 0.37 |  |  | 6.9 |  |
| 0.49 B, 0.45 GiB | int8 | NPU | NPU | 8.0 | 124.8 | 61 | 61 | 0.04 |  |  | 3.6 |  |
| 0.49 B, 0.45 GiB | int8 | CPU | CPU 130 | 18.7 | 53.5 | 26 | 26 | 3.71 |  |  | 2.7 |  |
| 0.49 B, 0.45 GiB | int8 | GPU | GPU 130 | 28.1 | 35.6 | 17 | 17 | 0.05 |  |  | 2.5 |  |
| 0.49 B, 0.23 GiB | lut4 | NPU | unverified | 4.1 | 243.3 | 59 | 118 | 0.05 |  |  | 4.5 |  |
| 0.49 B, 0.23 GiB | lut4 | CPU | CPU 130 | 19.8 | 50.6 | 12 | 25 | 3.42 |  |  | 3.7 |  |
| 0.49 B, 0.23 GiB | lut4 | GPU | GPU 130 | 28.1 | 35.6 | 9 | 17 | 0.13 |  |  | 1.1 |  |
| 0.49 B, 0.34 GiB | lut6 | NPU | NPU | 5.8 | 173.9 | 63 | 85 | 0.04 |  |  | 14.2 |  |
| 0.49 B, 0.34 GiB | lut6 | CPU | CPU 130 | 18.3 | 54.5 | 20 | 27 | 3.75 |  |  | 2.1 |  |
| 0.49 B, 0.34 GiB | lut6 | GPU | GPU 130 | 12.7 | 78.9 | 29 | 38 | 0.11 |  |  | 1.7 |  |
| 0.49 B, 0.45 GiB | lut8 | NPU | NPU | 7.7 | 130.3 | 63 | 63 | 0.03 |  |  | 6.7 |  |
| 0.49 B, 0.45 GiB | lut8 | CPU | CPU 130 | 18.0 | 55.5 | 27 | 27 | 3.89 |  |  | 4.1 |  |
| 0.49 B, 0.45 GiB | lut8 | GPU | GPU 130 | 29.6 | 33.8 | 16 | 16 | 0.11 |  |  | 2.2 |  |
| 0.49 B, 0.23 GiB | int4 | CPU | CPU 130 | 18.3 | 54.7 | 13 | 27 | 3.82 |  |  | 2.3 |  |
| 0.49 B, 0.23 GiB | int4 | GPU | GPU 130 | 9.8 | 102.0 | 25 | 50 | 0.39 |  |  | 1.2 |  |
| 0.97 B, 1.81 GiB | fp16 | NPU | CPU 258 | 34.7 | 28.8 | 56 | 28 | 4.14 |  |  | 14.7 |  |
| 0.97 B, 1.81 GiB | fp16 | CPU | CPU 258 | 37.3 | 26.8 | 52 | 26 | 3.88 |  |  | 12.8 |  |
| 0.97 B, 1.81 GiB | fp16 | GPU | GPU 258 | 23.0 | 43.6 | 85 | 42 | 0.11 |  |  | 16.3 |  |
| 0.97 B, 0.91 GiB | int8 | NPU | NPU | 15.4 | 64.8 | 63 | 63 | 0.02 |  |  | 7.3 |  |
| 0.97 B, 0.91 GiB | int8 | CPU | CPU 258 | 35.8 | 27.9 | 27 | 27 | 3.98 |  |  | 7.5 |  |
| 0.97 B, 0.91 GiB | int8 | GPU | error |  |  |  |  |  |  |  | 0.0 | measurement process failed: signal: abort trap: 12 hwnpu-dev 0x0000000100f19d2c hwnpu-dev + 564524 / ) / libc++abi: terminating due to uncaught exception of type NSException |
| 0.97 B, 0.45 GiB | lut4 | NPU | NPU | 8.0 | 125.8 | 61 | 122 | 0.03 |  |  | 11.7 |  |
| 0.97 B, 0.45 GiB | lut4 | CPU | CPU 258 | 36.9 | 27.1 | 13 | 26 | 3.74 |  |  | 8.2 |  |
| 0.97 B, 0.45 GiB | lut4 | GPU | GPU 258 | 55.1 | 18.1 | 9 | 18 | 0.11 |  |  | 3.6 |  |
| 0.97 B, 0.68 GiB | lut6 | NPU | NPU | 11.4 | 87.8 | 64 | 85 | 0.03 |  |  | 29.0 |  |
| 0.97 B, 0.68 GiB | lut6 | CPU | CPU 258 | 39.8 | 25.1 | 18 | 24 | 3.43 |  |  | 5.2 |  |
| 0.97 B, 0.68 GiB | lut6 | GPU | GPU 258 | 27.3 | 36.6 | 27 | 36 | 0.13 |  |  | 4.4 |  |
| 0.97 B, 0.91 GiB | lut8 | NPU | NPU | 15.3 | 65.6 | 64 | 64 | 0.03 |  |  | 13.6 |  |
| 0.97 B, 0.91 GiB | lut8 | CPU | CPU 258 | 36.8 | 27.2 | 26 | 26 | 3.90 |  |  | 9.4 |  |
| 0.97 B, 0.91 GiB | lut8 | GPU | GPU 258 | 64.0 | 15.6 | 15 | 15 | 0.14 |  |  | 4.6 |  |
| 0.97 B, 0.45 GiB | int4 | CPU | CPU 258 | 41.3 | 24.2 | 12 | 24 | 3.23 |  |  | 6.3 |  |
| 0.97 B, 0.45 GiB | int4 | GPU | GPU 258 | 12.8 | 78.1 | 38 | 76 | 0.61 |  |  | 3.3 |  |

Placement: Apple, Core ML's compute plan (MLComputePlan) for every operation; Intel, NPU when the
driver compiler built the graph (it has no fallback); AMD, IRON's full ELF, all on the NPU. CPU use is the process CPU time divided by
the wall time while the model runs: near 0 when the NPU does the work, 1 or more when the CPU
does. weights GB/s is the weight bytes times calls per second; G weights/s the weights times calls
per second, the same in every format (a model of N billion weights decodes at that over N tokens/s). net W is the device's power meter
while running minus idle, when it has one.
