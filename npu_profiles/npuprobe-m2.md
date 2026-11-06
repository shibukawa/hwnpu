# hwnpu npuprobe

2026-11-06T23:17:40+09:00

| | |
|---|---|
| CPU | Apple M2 |
| machine | Mac14,2 |
| OS | macOS 27.2 |
| RAM | 16 GiB |
| NPU | Apple Neural Engine (16 cores) (coreml) |
| driver | Core ML / Version 27.2 (Build 26B123)  |
| hwnpu | (devel), go1.27.3 |

Models: one token through Llama 3.2 1B sized layers (61 M weights per layer, random values), so a call is the work of generating one token without the attention over the context.

## Speed

| weights | format | unit | placement | ms/token | tokens/s | weights GB/s | CPU use | net W | mJ/token | compile s | error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.49 B, 0.91 GiB | fp16 | NPU | NPU | 15.9 | 63.0 | 61 | 0.02 |  |  | 7.6 |  |
| 0.49 B, 0.91 GiB | fp16 | CPU | CPU 130 | 17.9 | 55.8 | 54 | 3.65 |  |  | 4.3 |  |
| 0.49 B, 0.91 GiB | fp16 | GPU | GPU 130 | 12.0 | 83.0 | 81 | 0.10 |  |  | 6.1 |  |
| 0.49 B, 0.45 GiB | int8 | NPU | NPU | 8.1 | 122.8 | 60 | 0.03 |  |  | 5.7 |  |
| 0.49 B, 0.45 GiB | int8 | CPU | CPU 130 | 17.9 | 56.0 | 27 | 3.69 |  |  | 5.9 |  |
| 0.49 B, 0.45 GiB | int8 | GPU | GPU 130 | 27.8 | 36.0 | 18 | 0.05 |  |  | 4.8 |  |
| 0.49 B, 0.23 GiB | lut4 | NPU | NPU | 4.1 | 246.7 | 60 | 0.05 |  |  | 3.9 |  |
| 0.49 B, 0.23 GiB | lut4 | CPU | CPU 130 | 17.4 | 57.5 | 14 | 4.24 |  |  | 4.8 |  |
| 0.49 B, 0.23 GiB | lut4 | GPU | GPU 130 | 27.0 | 37.1 | 9 | 0.11 |  |  | 1.6 |  |
| 0.49 B, 0.23 GiB | int4 | CPU | CPU 130 | 18.0 | 55.5 | 14 | 4.07 |  |  | 5.5 |  |
| 0.49 B, 0.23 GiB | int4 | GPU | GPU 130 | 4.6 | 217.1 | 53 | 0.25 |  |  | 2.8 |  |
| 0.97 B, 1.81 GiB | fp16 | NPU | CPU 258 | 34.7 | 28.8 | 56 | 4.29 |  |  | 22.2 |  |
| 0.97 B, 1.81 GiB | fp16 | CPU | CPU 258 | 35.0 | 28.6 | 56 | 4.13 |  |  | 16.5 |  |
| 0.97 B, 1.81 GiB | fp16 | GPU | unverified | 25.5 | 39.2 | 76 | 0.43 |  |  | 17.9 |  |
| 0.97 B, 0.91 GiB | int8 | NPU | NPU | 15.6 | 64.2 | 62 | 0.02 |  |  | 11.6 |  |
| 0.97 B, 0.91 GiB | int8 | CPU | CPU 258 | 37.1 | 26.9 | 26 | 3.95 |  |  | 17.0 |  |
| 0.97 B, 0.91 GiB | int8 | GPU | GPU 258 | 55.2 | 18.1 | 18 | 0.06 |  |  | 12.0 |  |
| 0.97 B, 0.45 GiB | lut4 | NPU | NPU | 7.8 | 128.8 | 63 | 0.03 |  |  | 22.9 |  |
| 0.97 B, 0.45 GiB | lut4 | CPU | CPU 258 | 34.7 | 28.9 | 14 | 4.17 |  |  | 33.2 |  |
| 0.97 B, 0.45 GiB | lut4 | GPU | GPU 258 | 55.4 | 18.0 | 9 | 0.11 |  |  | 6.4 |  |
| 0.97 B, 0.45 GiB | int4 | CPU | CPU 258 | 35.4 | 28.2 | 14 | 4.02 |  |  | 14.2 |  |
| 0.97 B, 0.45 GiB | int4 | GPU | GPU 258 | 8.5 | 117.8 | 57 | 0.35 |  |  | 2.9 |  |

## Size limit (one model at a time, up to 4.0 GiB)

| format | layers | weights | placement | ms/call | weights GB/s | CPU use | compile s | error |
|---|---|---|---|---|---|---|---|---|
| fp16 | 2 | 0.23 GiB | NPU | 4.3 | 57 | 0.04 | 1.5 |  |
| fp16 | 4 | 0.45 GiB | NPU | 8.4 | 58 | 0.02 | 3.9 |  |
| fp16 | 8 | 0.91 GiB | NPU | 15.3 | 63 | 0.02 | 15.8 |  |
| fp16 | 9 | 1.02 GiB | CPU 146 | 55.8 | 20 | 2.91 | 6.5 |  |
| fp16 | 10 | 1.13 GiB | CPU 162 | 61.3 | 20 | 2.93 | 6.9 |  |
| fp16 | 12 | 1.36 GiB | CPU 194 | 76.7 | 19 | 2.82 | 9.5 |  |
| int8 | 2 | 0.11 GiB | NPU | 2.2 | 55 | 0.07 | 0.9 |  |
| int8 | 4 | 0.23 GiB | NPU | 4.1 | 59 | 0.04 | 1.8 |  |
| int8 | 8 | 0.45 GiB | NPU | 8.2 | 59 | 0.02 | 3.6 |  |
| int8 | 12 | 0.68 GiB | NPU | 12.1 | 60 | 0.02 | 5.6 |  |
| int8 | 16 | 0.91 GiB | NPU | 15.8 | 62 | 0.02 | 6.8 |  |
| int8 | 17 | 0.96 GiB | NPU | 17.1 | 60 | 0.02 | 10.3 |  |
| int8 | 18 | 1.02 GiB | CPU 290 | 142.0 | 8 | 2.24 | 8.8 |  |
| int8 | 20 | 1.13 GiB | CPU 322 | 136.7 | 9 | 2.69 | 18.6 |  |
| int8 | 24 | 1.36 GiB | CPU 386 | 3811.1 | 0 | 0.31 | 33.5 |  |
| lut4 | 2 | 0.06 GiB | NPU | 1.2 | 50 | 0.12 | 1.0 |  |
| lut4 | 4 | 0.11 GiB | NPU | 2.1 | 57 | 0.07 | 1.8 |  |
| lut4 | 8 | 0.23 GiB | NPU | 4.1 | 60 | 0.04 | 3.6 |  |
| lut4 | 12 | 0.34 GiB | NPU | 5.9 | 62 | 0.03 | 5.8 |  |
| lut4 | 16 | 0.45 GiB | NPU | 7.9 | 62 | 0.03 | 7.7 |  |
| lut4 | 24 | 0.68 GiB | NPU | 11.4 | 64 | 0.02 | 12.9 |  |
| lut4 | 32 | 0.91 GiB | NPU | 15.1 | 64 | 0.01 | 24.7 |  |
| lut4 | 34 | 0.96 GiB | NPU | 16.6 | 62 | 0.01 | 16.6 |  |
| lut4 | 35 | 0.99 GiB | NPU | 17.0 | 63 | 0.02 | 20.6 |  |
| lut4 | 36 | 1.02 GiB | CPU 578 | 237.3 | 5 | 2.88 | 51.0 |  |
| lut4 | 40 | 1.13 GiB | CPU 642 | 6302.2 | 0 | 0.23 | 60.9 |  |
| lut4 | 48 | 1.36 GiB | unverified | 17681.3 | 0 | 1.00 | 154.0 | placement not verified: MLComputePlan did not finish within 2 minutes |

- fp16: 8 layers (0.91 GiB) on the NPU, 9 layers (1.02 GiB) not.
- int8: 17 layers (0.96 GiB) on the NPU, 18 layers (1.02 GiB) not.
- lut4: 35 layers (0.99 GiB) on the NPU, 36 layers (1.02 GiB) not.

## Several models at once (fp16, 8 layers each, 2.72 GiB loaded in total)

| model | weights | placement | ms/call after loading it | CPU use | ms/call with all loaded | CPU use with all loaded | compile s | error |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.91 GiB | NPU | 16.2 | 0.02 | 16.2 | 0.01 | 8.7 |  |
| 2 | 0.91 GiB | NPU | 15.3 | 0.02 | 15.8 | 0.02 | 11.5 |  |
| 3 | 0.91 GiB | NPU | 16.3 | 0.01 | 15.2 | 0.01 | 51.9 |  |

Placement: Apple, Core ML's compute plan (MLComputePlan) for every operation; Intel, NPU when the
driver compiler built the graph (it has no fallback). CPU use is the process CPU time divided by
the wall time while the model runs: near 0 when the NPU does the work, 1 or more when the CPU
does. weights GB/s is the weight bytes times calls per second. net W is the device's power meter
while running minus idle, when it has one.
