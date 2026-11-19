# hwnpu-dev probe

2026-11-19T22:16:36+09:00

| | |
|---|---|
| CPU | AMD Ryzen AI 9 365 w/ Radeon 880M |
| machine | AI Series |
| OS | Linux, kernel 7.3.0-10-generic |
| RAM | 59 GiB |
| NPU | AMD NPU npu4 (Strix Point), 8 columns (xdna) |
| driver | amdxdna, AIE 1.1, 4 core rows 1.1.2.64 |
| power meter | amdgpu PPT (integrated), idle 5.38 W |
| hwnpu | v0.0.0-20261116001404-75165f1c8b9c+dirty 75165f1c8b9c5528ae53eec19a57846460374c7e, go1.27.1 |

Models: one token through Llama 3.2 1B sized layers (61 M weights per layer, random values), so a call is the work of generating one token without the attention over the context.

## Speed of a decode step (AMD, IRON)

| weights | format | form | commands | ms/token | tokens/s | weights GB/s | G weights/s | CPU use | net W | mJ/token | export s | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.49 B (8 layers), 0.28 GiB | q4 | layers | 1 | 16.1 | 62.1 | 19 | 30 | 0.07 | 9.23 | 148.7 | 26 |  |
| 0.49 B (8 layers), 0.28 GiB | q4 | stream | 1 | 5.6 | 177.8 | 54 | 86 | 0.02 | 13.04 | 73.3 | 26 |  |
| 0.49 B (8 layers), 0.37 GiB | q6 | layers | 1 | 20.3 | 49.3 | 20 | 24 | 0.06 | 10.35 | 209.8 | 10 |  |
| 0.49 B (8 layers), 0.37 GiB | q6 | stream | 1 | 10.3 | 96.9 | 38 | 47 | 0.04 | 12.19 | 125.9 | 7 |  |
| 0.49 B (8 layers), 0.48 GiB | q8 | layers | 1 | 19.3 | 51.8 | 27 | 25 | 0.09 | 9.21 | 177.6 | 6 |  |
| 0.49 B (8 layers), 0.48 GiB | q8 | stream | 1 | 9.4 | 106.5 | 55 | 52 | 0.03 | 16.08 | 150.9 | 6 |  |
| 0.49 B (8 layers), 0.91 GiB | bf16 | layers | 1 | 20.8 | 48.1 | 47 | 23 | 0.07 | 9.18 | 190.8 | 5 |  |
| 0.49 B (8 layers), 0.91 GiB | bf16 | stream | 1 | 17.8 | 56.2 | 55 | 27 | 0.05 | 9.75 | 173.3 | 5 |  |
| 0.97 B (16 layers), 0.57 GiB | q4 | layers | 1 | 31.3 | 32.0 | 19 | 31 | 0.06 | 9.36 | 292.9 | 51 |  |
| 0.97 B (16 layers), 0.57 GiB | q4 | stream | 1 | 11.6 | 86.6 | 53 | 84 | 0.08 | 11.81 | 136.4 | 52 |  |
| 0.97 B (16 layers), 0.74 GiB | q6 | layers | 1 | 40.2 | 24.9 | 20 | 24 | 0.06 | 9.02 | 362.6 | 14 |  |
| 0.97 B (16 layers), 0.74 GiB | q6 | stream | 1 | 20.7 | 48.4 | 38 | 47 | 0.05 | 9.52 | 196.7 | 14 |  |
| 0.97 B (16 layers), 0.96 GiB | q8 | layers | 1 | 38.2 | 26.2 | 27 | 25 | 0.07 | 8.69 | 332.2 | 12 |  |
| 0.97 B (16 layers), 0.96 GiB | q8 | stream | 1 | 19.0 | 52.7 | 54 | 51 | 0.05 | 10.92 | 207.2 | 12 |  |
| 0.97 B (16 layers), 1.81 GiB | bf16 | layers | 1 | 41.0 | 24.4 | 47 | 24 | 0.06 | 9.11 | 373.5 | 10 |  |
| 0.97 B (16 layers), 1.81 GiB | bf16 | stream | 1 | 35.3 | 28.3 | 55 | 28 | 0.04 | 9.29 | 327.7 | 10 |  |

layers: a layer's seven linears one after another (QGEMV for q4, q6 and q8, IRON's GEMV for bf16) and the SwiGLU's product, as hwnpu's exports of the other models than gpt-oss run them: every operator that follows another is loaded onto the cores again. stream: the same bytes in GEMVs of one shape (8192 by 2048) back to back, not loaded again: the time is the weights' transfer, the ceiling of a decode step run as one resident array (gpt-oss-20b's GptOssStep, docs/design.md 16.179).

## K rows per call (AMD, IRON)

| weights | format | K | path | ms/call | × K=1 | ms/token | tokens/s | TFLOPS | export s | error |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.49 B (8 layers) | q8 | 1 | QGEMV (decode step) | 19.2 | 1.00 | 19.167 | 52 | 0.05 | 6 |  |
| 0.49 B (8 layers) | q8 | 2 | QGEMVRows (checking drafts) | 19.9 | 1.04 | 9.945 | 101 | 0.10 | 6 |  |
| 0.49 B (8 layers) | q8 | 4 | QGEMVRows (checking drafts) | 24.4 | 1.27 | 6.107 | 164 | 0.16 | 7 |  |
| 0.49 B (8 layers) | q8 | 8 | QGEMVRows (checking drafts) | 54.6 | 2.85 | 6.820 | 147 | 0.14 | 6 |  |
| 0.49 B (8 layers) | q8 | 256 | dequantize and GEMM (prompt chunk) | 88.0 | 4.59 | 0.344 | 2909 | 2.83 | 8 |  |
| 0.49 B (8 layers) | q8 | 1024 | dequantize and GEMM (prompt chunk) | 142.7 | 7.44 | 0.139 | 7177 | 6.98 | 8 |  |
| 0.49 B (8 layers) | q8 | 4096 | dequantize and GEMM (prompt chunk) | 361.9 | 18.88 | 0.088 | 11317 | 11.01 | 8 |  |
| 0.49 B (8 layers) | q6 | 1 | QGEMV (decode step) | 20.2 | 1.00 | 20.189 | 50 | 0.05 | 7 |  |
| 0.49 B (8 layers) | q6 | 2 | QGEMVRows (checking drafts) | 20.8 | 1.03 | 10.394 | 96 | 0.09 | 9 |  |
| 0.49 B (8 layers) | q6 | 4 | QGEMVRows (checking drafts) | 25.0 | 1.24 | 6.249 | 160 | 0.16 | 10 |  |
| 0.49 B (8 layers) | q6 | 8 | QGEMVRows (checking drafts) | 79.2 | 3.92 | 9.903 | 101 | 0.10 | 10 |  |
| 0.49 B (8 layers) | q6 | 256 | dequantize and GEMM (prompt chunk) | 92.3 | 4.57 | 0.360 | 2774 | 2.70 | 12 |  |
| 0.49 B (8 layers) | q6 | 1024 | dequantize and GEMM (prompt chunk) | 147.1 | 7.29 | 0.144 | 6960 | 6.77 | 11 |  |
| 0.49 B (8 layers) | q6 | 4096 | dequantize and GEMM (prompt chunk) | 366.8 | 18.17 | 0.090 | 11165 | 10.86 | 12 |  |
| 0.49 B (8 layers) | q4 | 1 | QGEMV (decode step) | 15.9 | 1.00 | 15.935 | 63 | 0.06 | 26 |  |
| 0.49 B (8 layers) | q4 | 256 | dequantize and GEMM (prompt chunk) | 85.9 | 5.39 | 0.336 | 2979 | 2.90 | 27 |  |
| 0.49 B (8 layers) | q4 | 1024 | dequantize and GEMM (prompt chunk) | 133.3 | 8.37 | 0.130 | 7679 | 7.47 | 27 |  |
| 0.49 B (8 layers) | q4 | 4096 | dequantize and GEMM (prompt chunk) | 320.5 | 20.11 | 0.078 | 12779 | 12.43 | 27 |  |
| 0.49 B (8 layers) | bfp16 | 256 | GEMM, weights packed on the host | 27.5 |  | 0.107 | 9303 | 9.05 | 26 |  |
| 0.49 B (8 layers) | bfp16 | 1024 | GEMM, weights packed on the host | 75.3 |  | 0.073 | 13607 | 13.24 | 26 |  |
| 0.49 B (8 layers) | bfp16 | 4096 | GEMM, weights packed on the host | 261.7 |  | 0.064 | 15652 | 15.23 | 26 |  |

× K=1 is the time per call over that of one row in the same format. A call checking 2 to 8 drafts streams the weights once through all its rows (q8, q6); a prompt chunk of q4 or q8 dequantizes its weights on the device at every call, bfp16 (the image DiTs' weights) is read by the GEMM as it is. TFLOPS counts a multiply and an add per weight and row.

## The image DiTs' attention (AMD, IRON)

| queries and keys | heads | ms/call | TFLOPS | export s | error |
|---|---|---|---|---|---|
| 1536 | 24 of 128 | 12.8 | 2.26 | 1 |  |
| 4608 | 24 of 128 | 81.8 | 3.19 | 1 |  |

iron_mha.ImageMHA with QKᵀ and P·V through bfp16 and the softmax's exponential on one bf16 limb (FLUX.2 klein's export, docs/design.md 16.207). TFLOPS counts the two products (4 × rows² × heads × dim).

Placement: Apple, Core ML's compute plan (MLComputePlan) for every operation; Intel, NPU when the
driver compiler built the graph (it has no fallback); AMD, IRON's full ELF, all on the NPU. CPU use is the process CPU time divided by
the wall time while the model runs: near 0 when the NPU does the work, 1 or more when the CPU
does. weights GB/s is the weight bytes times calls per second; G weights/s the weights times calls
per second, the same in every format (a model of N billion weights decodes at that over N tokens/s). net W is the device's power meter
while running minus idle, when it has one.
