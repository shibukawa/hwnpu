# hwnpu npuprobe

2026-11-07T13:31:54+09:00

| | |
|---|---|
| CPU | Apple M1 |
| machine | MacBookPro17,1 |
| OS | macOS 27.2 |
| RAM | 16 GiB |
| NPU | Apple Neural Engine (16 cores) (coreml) |
| driver | Core ML / Version 27.2 (Build 26B123)  |
| hwnpu | (devel), go1.27.3 |

Models: one token through Llama 3.2 1B sized layers (61 M weights per layer, random values), so a call is the work of generating one token without the attention over the context.

## Speed

| weights | format | unit | placement | ms/token | tokens/s | weights GB/s | CPU use | net W | mJ/token | compile s | error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.49 B, 0.91 GiB | fp16 | NPU | NPU | 15.6 | 64.3 | 63 | 0.01 |  |  | 6.2 |  |
| 0.49 B, 0.91 GiB | fp16 | CPU | CPU 130 | 19.3 | 51.9 | 51 | 5.00 |  |  | 3.0 |  |
| 0.49 B, 0.91 GiB | fp16 | GPU | GPU 130 | 19.0 | 52.5 | 51 | 0.29 |  |  | 4.2 |  |
| 0.49 B, 0.45 GiB | int8 | NPU | NPU | 8.0 | 125.4 | 61 | 0.03 |  |  | 3.3 |  |
| 0.49 B, 0.45 GiB | int8 | CPU | CPU 130 | 19.4 | 51.5 | 25 | 4.91 |  |  | 2.5 |  |
| 0.49 B, 0.45 GiB | int8 | GPU | GPU 130 | 38.6 | 25.9 | 13 | 0.20 |  |  | 2.2 |  |
| 0.49 B, 0.23 GiB | lut4 | NPU | NPU | 5.6 | 179.0 | 44 | 0.03 |  |  | 3.7 |  |
| 0.49 B, 0.23 GiB | lut4 | CPU | CPU 130 | 19.2 | 52.0 | 13 | 4.99 |  |  | 3.2 |  |
| 0.49 B, 0.23 GiB | lut4 | GPU | GPU 130 | 42.1 | 23.8 | 6 | 0.28 |  |  | 1.2 |  |
| 0.49 B, 0.23 GiB | int4 | CPU | CPU 130 | 19.3 | 51.7 | 13 | 4.95 |  |  | 2.4 |  |
| 0.49 B, 0.23 GiB | int4 | GPU | GPU 130 | 8.4 | 118.3 | 29 | 0.46 |  |  | 1.3 |  |
| 0.97 B, 1.81 GiB | fp16 | NPU | CPU 258 | 38.2 | 26.2 | 51 | 4.99 |  |  | 7.0 |  |
| 0.97 B, 1.81 GiB | fp16 | CPU | CPU 258 | 38.1 | 26.2 | 51 | 5.02 |  |  | 7.4 |  |
| 0.97 B, 1.81 GiB | fp16 | GPU | GPU 258 | 35.3 | 28.4 | 55 | 0.32 |  |  | 8.6 |  |
| 0.97 B, 0.91 GiB | int8 | NPU | NPU | 15.7 | 63.7 | 62 | 0.02 |  |  | 6.4 |  |
| 0.97 B, 0.91 GiB | int8 | CPU | CPU 258 | 38.2 | 26.2 | 25 | 5.01 |  |  | 5.0 |  |
| 0.97 B, 0.91 GiB | int8 | GPU | GPU 258 | 73.4 | 13.6 | 13 | 0.21 |  |  | 4.1 |  |
| 0.97 B, 0.45 GiB | lut4 | NPU | NPU | 11.0 | 91.3 | 44 | 0.02 |  |  | 7.6 |  |
| 0.97 B, 0.45 GiB | lut4 | CPU | CPU 258 | 38.3 | 26.1 | 13 | 4.96 |  |  | 6.3 |  |
| 0.97 B, 0.45 GiB | lut4 | GPU | GPU 258 | 83.2 | 12.0 | 6 | 0.28 |  |  | 2.3 |  |
| 0.97 B, 0.45 GiB | int4 | CPU | CPU 258 | 41.3 | 24.2 | 12 | 4.52 |  |  | 5.1 |  |
| 0.97 B, 0.45 GiB | int4 | GPU | GPU 258 | 14.8 | 67.7 | 33 | 0.58 |  |  | 2.8 |  |

## Size limit (one model at a time, up to 4.0 GiB)

| format | layers | weights | placement | ms/call | weights GB/s | CPU use | compile s | error |
|---|---|---|---|---|---|---|---|---|
| fp16 | 2 | 0.23 GiB | NPU | 4.9 | 50 | 0.08 | 1.5 |  |
| fp16 | 4 | 0.45 GiB | NPU | 8.5 | 57 | 0.04 | 2.9 |  |
| fp16 | 8 | 0.91 GiB | NPU | 16.7 | 58 | 0.03 | 5.7 |  |
| fp16 | 9 | 1.02 GiB | CPU 146 | 51.8 | 21 | 3.82 | 3.9 |  |
| fp16 | 10 | 1.13 GiB | CPU 162 | 62.2 | 20 | 3.56 | 4.8 |  |
| fp16 | 12 | 1.36 GiB | CPU 194 | 81.3 | 18 | 3.24 | 5.4 |  |
| int8 | 2 | 0.11 GiB | NPU | 2.2 | 54 | 0.08 | 1.0 |  |
| int8 | 4 | 0.23 GiB | NPU | 4.1 | 60 | 0.04 | 1.8 |  |
| int8 | 8 | 0.45 GiB | NPU | 8.0 | 61 | 0.03 | 3.4 |  |
| int8 | 12 | 0.68 GiB | NPU | 11.6 | 63 | 0.02 | 4.9 |  |
| int8 | 16 | 0.91 GiB | NPU | 15.9 | 61 | 0.02 | 6.5 |  |
| int8 | 17 | 0.96 GiB | NPU | 16.5 | 63 | 0.01 | 7.5 |  |
| int8 | 18 | 1.02 GiB | CPU 290 | 109.6 | 10 | 3.64 | 6.9 |  |
| int8 | 20 | 1.13 GiB | CPU 322 | 123.8 | 10 | 3.60 | 8.7 |  |
| int8 | 24 | 1.36 GiB | CPU 386 | 146.1 | 10 | 3.64 | 12.7 |  |
| lut4 | 2 | 0.06 GiB | NPU | 1.6 | 39 | 0.09 | 1.1 |  |
| lut4 | 4 | 0.11 GiB | NPU | 2.9 | 42 | 0.05 | 1.9 |  |
| lut4 | 8 | 0.23 GiB | NPU | 5.6 | 43 | 0.03 | 3.7 |  |
| lut4 | 12 | 0.34 GiB | NPU | 8.3 | 44 | 0.02 | 5.7 |  |
| lut4 | 16 | 0.45 GiB | NPU | 10.7 | 45 | 0.02 | 7.7 |  |
| lut4 | 24 | 0.68 GiB | NPU | 16.0 | 45 | 0.01 | 11.7 |  |
| lut4 | 32 | 0.91 GiB | NPU | 21.7 | 45 | 0.01 | 16.2 |  |
| lut4 | 34 | 0.96 GiB | NPU | 23.1 | 45 | 0.01 | 17.1 |  |
| lut4 | 35 | 0.99 GiB | NPU | 23.7 | 45 | 0.01 | 17.6 |  |
| lut4 | 36 | 1.02 GiB | CPU 578 | 2700.0 | 0 | 0.59 | 23.7 |  |
| lut4 | 40 | 1.13 GiB | CPU 642 | 3267.7 | 0 | 0.63 | 25.2 |  |
| lut4 | 48 | 1.36 GiB | CPU 770 | 6338.1 | 0 | 0.58 | 43.2 |  |

- fp16: 8 layers (0.91 GiB) on the NPU, 9 layers (1.02 GiB) not.
- int8: 17 layers (0.96 GiB) on the NPU, 18 layers (1.02 GiB) not.
- lut4: 35 layers (0.99 GiB) on the NPU, 36 layers (1.02 GiB) not.

## Several models at once (fp16, 8 layers each, 2.72 GiB loaded in total)

| model | weights | placement | ms/call after loading it | CPU use | ms/call with all loaded | CPU use with all loaded | compile s | error |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.91 GiB | NPU | 15.6 | 0.02 | 15.4 | 0.01 | 5.2 |  |
| 2 | 0.91 GiB | NPU | 15.4 | 0.01 | 15.4 | 0.01 | 5.5 |  |
| 3 | 0.91 GiB | NPU | 15.4 | 0.01 | 15.4 | 0.01 | 5.5 |  |

Placement: Apple, Core ML's compute plan (MLComputePlan) for every operation; Intel, NPU when the
driver compiler built the graph (it has no fallback). CPU use is the process CPU time divided by
the wall time while the model runs: near 0 when the NPU does the work, 1 or more when the CPU
does. weights GB/s is the weight bytes times calls per second. net W is the device's power meter
while running minus idle, when it has one.
