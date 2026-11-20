# hwnpu-dev probe

2026-11-20T00:27:06+09:00

| | |
|---|---|
| CPU | AMD Ryzen AI 9 365 w/ Radeon 880M |
| machine | AI Series |
| OS | Linux, kernel 7.3.0-10-generic |
| RAM | 59 GiB |
| NPU | AMD NPU npu4 (Strix Point), 8 columns (xdna) |
| driver | amdxdna, AIE 1.1, 4 core rows 1.1.2.64 |
| power meter | amdgpu PPT (integrated), idle 5.28 W |
| hwnpu | v0.0.0-20261119145443-521c35d103f1+dirty 521c35d103f123eac176aa68d930aba8194ac66d, go1.27.1 |

Models: one token through Llama 3.2 1B sized layers (61 M weights per layer, random values), so a call is the work of generating one token without the attention over the context.

## Speed of a decode step (AMD, IRON)

| weights | format | form | commands | ms/token | tokens/s | weights GB/s | G weights/s | CPU use | net W | mJ/token | export s | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.49 B (8 layers), 0.28 GiB | q4 | layers | 1 | 15.9 | 62.9 | 19 | 31 | 0.06 | 9.26 | 147.2 | 26 |  |
| 0.49 B (8 layers), 0.28 GiB | q4 | stream | 1 | 5.6 | 177.3 | 54 | 86 | 0.04 | 12.96 | 73.1 | 26 |  |
| 0.49 B (8 layers), 0.37 GiB | q6 | layers | 1 | 20.3 | 49.2 | 19 | 24 | 0.08 | 8.25 | 167.8 | 7 |  |
| 0.49 B (8 layers), 0.37 GiB | q6 | stream | 1 | 10.2 | 98.1 | 39 | 48 | 0.03 | 10.61 | 108.2 | 7 |  |
| 0.49 B (8 layers), 0.48 GiB | q8 | layers | 1 | 19.4 | 51.5 | 27 | 25 | 0.10 | 8.56 | 166.1 | 6 |  |
| 0.49 B (8 layers), 0.48 GiB | q8 | stream | 1 | 9.4 | 106.8 | 55 | 52 | 0.02 | 11.37 | 106.4 | 6 |  |
| 0.49 B (8 layers), 0.91 GiB | bf16 | layers | 1 | 20.6 | 48.4 | 47 | 24 | 0.06 | 9.91 | 204.6 | 5 |  |
| 0.49 B (8 layers), 0.91 GiB | bf16 | stream | 1 | 17.8 | 56.1 | 55 | 27 | 0.05 | 9.75 | 173.6 | 5 |  |
| 0.97 B (16 layers), 0.57 GiB | q4 | layers | 1 | 31.3 | 32.0 | 19 | 31 | 0.08 | 9.10 | 284.4 | 51 |  |
| 0.97 B (16 layers), 0.57 GiB | q4 | stream | 1 | 11.3 | 88.2 | 54 | 86 | 0.05 | 12.43 | 140.9 | 52 |  |
| 0.97 B (16 layers), 0.74 GiB | q6 | layers | 1 | 40.5 | 24.7 | 20 | 24 | 0.08 | 8.42 | 341.0 | 13 |  |
| 0.97 B (16 layers), 0.74 GiB | q6 | stream | 1 | 20.6 | 48.7 | 38 | 47 | 0.06 | 10.39 | 213.4 | 13 |  |
| 0.97 B (16 layers), 0.96 GiB | q8 | layers | 1 | 38.0 | 26.3 | 27 | 26 | 0.08 | 8.76 | 332.7 | 12 |  |
| 0.97 B (16 layers), 0.96 GiB | q8 | stream | 1 | 18.9 | 52.8 | 55 | 51 | 0.06 | 11.20 | 212.1 | 12 |  |
| 0.97 B (16 layers), 1.81 GiB | bf16 | layers | 1 | 41.1 | 24.3 | 47 | 24 | 0.09 | 8.97 | 368.8 | 10 |  |
| 0.97 B (16 layers), 1.81 GiB | bf16 | stream | 1 | 35.5 | 28.2 | 55 | 27 | 0.07 | 9.68 | 343.2 | 10 |  |

layers: a layer's seven linears one after another (QGEMV for q4, q6 and q8, IRON's GEMV for bf16) and the SwiGLU's product, as hwnpu's exports of the other models than gpt-oss run them: every operator that follows another is loaded onto the cores again. stream: the same bytes in GEMVs of one shape (8192 by 2048) back to back, not loaded again: the time is the weights' transfer, the ceiling of a decode step run as one resident array (gpt-oss-20b's GptOssStep, docs/design.md 16.179).

## K rows per call (AMD, IRON)

| weights | format | K | path | ms/call | × K=1 | ms/token | tokens/s | TFLOPS | export s | error |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.49 B (8 layers) | q8 | 1 | QGEMV (decode step) | 19.2 | 1.00 | 19.205 | 52 | 0.05 | 6 |  |
| 0.49 B (8 layers) | q8 | 2 | QGEMVRows (checking drafts) | 19.8 | 1.03 | 9.923 | 101 | 0.10 | 6 |  |
| 0.49 B (8 layers) | q8 | 4 | QGEMVRows (checking drafts) | 24.5 | 1.28 | 6.125 | 163 | 0.16 | 6 |  |
| 0.49 B (8 layers) | q8 | 8 | QGEMVRows (checking drafts) | 54.5 | 2.84 | 6.816 | 147 | 0.14 | 7 |  |
| 0.49 B (8 layers) | q8 | 256 | dequantize and GEMM (prompt chunk) | 87.6 | 4.56 | 0.342 | 2922 | 2.84 | 8 |  |
| 0.49 B (8 layers) | q8 | 1024 | dequantize and GEMM (prompt chunk) | 142.3 | 7.41 | 0.139 | 7195 | 7.00 | 8 |  |
| 0.49 B (8 layers) | q8 | 4096 | dequantize and GEMM (prompt chunk) | 361.5 | 18.82 | 0.088 | 11330 | 11.02 | 8 |  |
| 0.49 B (8 layers) | q6 | 1 | QGEMV (decode step) | 20.5 | 1.00 | 20.489 | 49 | 0.05 | 7 |  |
| 0.49 B (8 layers) | q6 | 2 | QGEMVRows (checking drafts) | 20.7 | 1.01 | 10.359 | 97 | 0.09 | 7 |  |
| 0.49 B (8 layers) | q6 | 4 | QGEMVRows (checking drafts) | 25.0 | 1.22 | 6.254 | 160 | 0.16 | 7 |  |
| 0.49 B (8 layers) | q6 | 8 | QGEMVRows (checking drafts) | 79.2 | 3.87 | 9.899 | 101 | 0.10 | 7 |  |
| 0.49 B (8 layers) | q6 | 256 | dequantize and GEMM (prompt chunk) | 92.4 | 4.51 | 0.361 | 2770 | 2.69 | 8 |  |
| 0.49 B (8 layers) | q6 | 1024 | dequantize and GEMM (prompt chunk) | 147.1 | 7.18 | 0.144 | 6961 | 6.77 | 8 |  |
| 0.49 B (8 layers) | q6 | 4096 | dequantize and GEMM (prompt chunk) | 366.7 | 17.89 | 0.090 | 11171 | 10.87 | 8 |  |
| 0.49 B (8 layers) | q4 | 1 | QGEMV (decode step) | 15.8 | 1.00 | 15.826 | 63 | 0.06 | 26 |  |
| 0.49 B (8 layers) | q4 | 256 | dequantize and GEMM (prompt chunk) | 86.1 | 5.44 | 0.336 | 2973 | 2.89 | 27 |  |
| 0.49 B (8 layers) | q4 | 1024 | dequantize and GEMM (prompt chunk) | 133.4 | 8.43 | 0.130 | 7676 | 7.47 | 27 |  |
| 0.49 B (8 layers) | q4 | 4096 | dequantize and GEMM (prompt chunk) | 321.2 | 20.30 | 0.078 | 12752 | 12.41 | 27 |  |
| 0.49 B (8 layers) | bfp16 | 256 | GEMM, weights packed on the host | 27.6 |  | 0.108 | 9265 | 9.02 | 26 |  |
| 0.49 B (8 layers) | bfp16 | 1024 | GEMM, weights packed on the host | 75.1 |  | 0.073 | 13637 | 13.27 | 25 |  |
| 0.49 B (8 layers) | bfp16 | 4096 | GEMM, weights packed on the host | 262.0 |  | 0.064 | 15631 | 15.21 | 26 |  |

× K=1 is the time per call over that of one row in the same format. A call checking 2 to 8 drafts streams the weights once through all its rows (q8, q6); a prompt chunk of q4 or q8 dequantizes its weights on the device at every call, bfp16 (the image DiTs' weights) is read by the GEMM as it is. TFLOPS counts a multiply and an add per weight and row.

## The image DiTs' attention (AMD, IRON)

| queries and keys | heads | ms/call | TFLOPS | export s | error |
|---|---|---|---|---|---|
| 1536 | 24 of 128 | 12.8 | 2.27 | 1 |  |
| 4608 | 24 of 128 | 82.2 | 3.17 | 1 |  |

iron_mha.ImageMHA with QKᵀ and P·V through bfp16 and the softmax's exponential on one bf16 limb (FLUX.2 klein's export, docs/design.md 16.207). TFLOPS counts the two products (4 × rows² × heads × dim).

## Where the power goes (AMD, gpu_metrics)

At rest: package 4.38 W (NPU 0.00, CPU cores 0.12, GPU 0.00, the rest 4.26), DRAM 0.3 GB/s read and 0.0 written.

| run | ms/call | package W | above rest | NPU W | CPU cores | the rest | NPU's share | the rest's share | DRAM GB/s | NPU busy | NPU MHz |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bench layers-q4-l8-r1 | 15.9 | 13.51 | +9.13 | 1.47 | +1.71 | +5.95 | 16 % | 65 % | 18.6 | 80 % | 1659 |
| bench stream-q4-l8-r1 | 5.6 | 17.82 | +13.44 | 0.74 | +2.55 | +10.15 | 5 % | 76 % | 48.0 | 83 % | 1657 |
| bench layers-q6-l8-r1 | 20.3 | 12.76 | +8.38 | 1.30 | +1.33 | +5.75 | 16 % | 69 % | 19.2 | 81 % | 1663 |
| bench stream-q6-l8-r1 | 10.2 | 15.30 | +10.92 | 0.68 | +1.61 | +8.63 | 6 % | 79 % | 34.9 | 83 % | 1658 |
| bench layers-q8-l8-r1 | 19.4 | 13.05 | +8.67 | 1.31 | +1.34 | +6.03 | 15 % | 69 % | 25.5 | 81 % | 1662 |
| bench stream-q8-l8-r1 | 9.4 | 16.16 | +11.79 | 0.67 | +1.72 | +9.39 | 6 % | 80 % | 48.8 | 83 % | 1656 |
| bench layers-bf16-l8-r1 | 20.6 | 14.40 | +10.02 | 0.81 | +2.34 | +6.88 | 8 % | 69 % | 43.7 | 82 % | 1664 |
| bench stream-bf16-l8-r1 | 17.8 | 14.10 | +9.72 | 0.61 | +1.63 | +7.48 | 6 % | 77 % | 48.7 | 82 % | 1661 |
| bench layers-q4-l16-r1 | 31.3 | 13.34 | +8.96 | 1.53 | +1.69 | +5.73 | 17 % | 64 % | 19.5 | 82 % | 1666 |
| bench stream-q4-l16-r1 | 11.3 | 17.21 | +12.83 | 0.75 | +1.79 | +10.29 | 6 % | 80 % | 47.7 | 82 % | 1658 |
| bench layers-q6-l16-r1 | 40.5 | 12.82 | +8.44 | 1.33 | +1.39 | +5.72 | 16 % | 68 % | 19.7 | 83 % | 1670 |
| bench stream-q6-l16-r1 | 20.6 | 15.19 | +10.81 | 0.69 | +1.68 | +8.45 | 6 % | 78 % | 35.7 | 82 % | 1658 |
| bench layers-q8-l16-r1 | 38.0 | 13.22 | +8.84 | 1.35 | +1.37 | +6.11 | 15 % | 69 % | 26.3 | 83 % | 1669 |
| bench stream-q8-l16-r1 | 18.9 | 16.04 | +11.66 | 0.68 | +1.64 | +9.34 | 6 % | 80 % | 49.4 | 83 % | 1662 |
| bench layers-bf16-l16-r1 | 41.1 | 13.43 | +9.05 | 0.82 | +1.39 | +6.83 | 9 % | 76 % | 43.7 | 83 % | 1666 |
| bench stream-bf16-l16-r1 | 35.5 | 14.09 | +9.71 | 0.62 | +1.59 | +7.50 | 6 % | 77 % | 50.2 | 84 % | 1668 |
| verify layers-q8-l8-r1 | 19.2 | 13.43 | +9.05 | 1.32 | +1.52 | +6.21 | 15 % | 69 % | 25.5 | 82 % | 1662 |
| verify layers-q8-l8-r2 | 19.8 | 13.82 | +9.45 | 1.29 | +1.41 | +6.75 | 14 % | 71 % | 26.0 | 81 % | 1664 |
| verify layers-q8-l8-r4 | 24.5 | 14.37 | +10.00 | 1.22 | +1.43 | +7.35 | 12 % | 74 % | 22.5 | 83 % | 1661 |
| verify layers-q8-l8-r8 | 54.5 | 13.71 | +9.33 | 0.87 | +1.31 | +7.15 | 9 % | 77 % | 12.8 | 84 % | 1676 |
| verify layers-q8-l8-r256 | 87.6 | 14.82 | +10.44 | 1.79 | +1.19 | +7.46 | 17 % | 71 % | 34.0 | 86 % | 1689 |
| verify layers-q8-l8-r1024 | 142.3 | 18.39 | +14.01 | 1.74 | +1.05 | +11.23 | 12 % | 80 % | 40.3 | 88 % | 1706 |
| verify layers-q8-l8-r4096 | 361.5 | 21.84 | +17.46 | 1.79 | +0.54 | +15.14 | 10 % | 87 % | 47.7 | 94 % | 1754 |
| verify layers-q6-l8-r1 | 20.5 | 13.12 | +8.74 | 1.30 | +1.48 | +5.96 | 15 % | 68 % | 19.2 | 82 % | 1663 |
| verify layers-q6-l8-r2 | 20.7 | 13.43 | +9.05 | 1.25 | +1.55 | +6.26 | 14 % | 69 % | 20.2 | 81 % | 1661 |
| verify layers-q6-l8-r4 | 25.0 | 13.55 | +9.18 | 1.18 | +1.42 | +6.58 | 13 % | 72 % | 18.0 | 82 % | 1663 |
| verify layers-q6-l8-r8 | 79.2 | 12.79 | +8.41 | 0.74 | +1.26 | +6.41 | 9 % | 76 % | 7.8 | 86 % | 1684 |
| verify layers-q6-l8-r256 | 92.4 | 14.61 | +10.23 | 1.71 | +1.33 | +7.18 | 17 % | 70 % | 31.2 | 86 % | 1687 |
| verify layers-q6-l8-r1024 | 147.1 | 18.19 | +13.81 | 1.70 | +1.10 | +11.01 | 12 % | 80 % | 38.2 | 88 % | 1705 |
| verify layers-q6-l8-r4096 | 366.7 | 21.67 | +17.29 | 1.77 | +0.56 | +14.96 | 10 % | 87 % | 46.9 | 94 % | 1756 |
| verify layers-q4-l8-r1 | 15.8 | 13.70 | +9.32 | 1.49 | +1.86 | +5.96 | 16 % | 64 % | 18.8 | 81 % | 1657 |
| verify layers-q4-l8-r256 | 86.1 | 14.32 | +9.94 | 2.54 | +1.35 | +6.05 | 26 % | 61 % | 23.2 | 84 % | 1686 |
| verify layers-q4-l8-r1024 | 133.4 | 17.93 | +13.55 | 2.31 | +1.07 | +10.16 | 17 % | 75 % | 32.4 | 87 % | 1702 |
| verify layers-q4-l8-r4096 | 321.2 | 22.42 | +18.04 | 2.15 | +0.64 | +15.26 | 12 % | 85 % | 44.0 | 93 % | 1748 |
| verify layers-bfp16-l8-r256 | 27.6 | 19.39 | +15.02 | 2.23 | +1.30 | +11.49 | 15 % | 77 % | 40.5 | 82 % | 1666 |
| verify layers-bfp16-l8-r1024 | 75.1 | 22.80 | +18.42 | 1.97 | +1.13 | +15.33 | 11 % | 83 % | 45.5 | 85 % | 1683 |
| verify layers-bfp16-l8-r4096 | 262.0 | 24.66 | +20.28 | 1.98 | +0.70 | +17.60 | 10 % | 87 % | 50.1 | 92 % | 1736 |
| attn attn-r1536-h24-d128 | 12.8 | 15.10 | +10.72 | 0.76 | +0.44 | +9.52 | 7 % | 89 % | 17.3 | 87 % | 1734 |
| attn attn-r4608-h24-d128 | 82.2 | 15.47 | +11.09 | 0.60 | +0.43 | +10.06 | 5 % | 91 % | 16.7 | 89 % | 1735 |

The SMU's own split (amdgpu's gpu_metrics, the means of a sample every 50 ms while the run is timed). NPU W is the NPU's rail; the rest is the package less the NPU, the CPU cores and the GPU: the data fabric, the memory controller and its PHY, and I/O, the part of moving the weights that is on the package. The DRAM chips are outside it and not measured. The shares are of the package's power above rest.

Placement: Apple, Core ML's compute plan (MLComputePlan) for every operation; Intel, NPU when the
driver compiler built the graph (it has no fallback); AMD, IRON's full ELF, all on the NPU. CPU use is the process CPU time divided by
the wall time while the model runs: near 0 when the NPU does the work, 1 or more when the CPU
does. weights GB/s is the weight bytes times calls per second; G weights/s the weights times calls
per second, the same in every format (a model of N billion weights decodes at that over N tokens/s). net W is the device's power meter
while running minus idle, when it has one.
