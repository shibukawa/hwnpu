# hwnpu-dev probe

2026-11-20T00:29:16+09:00

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

## K tokens per call (checking K drafted tokens)

| weights | format | K | placement | ms/call | × K=1 | ms/token | tokens/s | weights GB/s | TFLOPS | CPU use | compile s | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.49 B, 0.91 GiB | fp16 | 1 | NPU | 15.6 | 1.00 | 15.6 | 63.9 | 62 | 0.06 | 0.03 | 8.0 |  |
| 0.49 B, 0.91 GiB | fp16 | 2 | NPU | 15.4 | 0.99 | 7.7 | 129.7 | 63 | 0.13 | 0.02 | 7.7 |  |
| 0.49 B, 0.91 GiB | fp16 | 4 | NPU | 15.7 | 1.00 | 3.9 | 254.8 | 62 | 0.25 | 0.02 | 6.3 |  |
| 0.49 B, 0.91 GiB | fp16 | 8 | NPU | 15.5 | 0.99 | 1.9 | 517.5 | 63 | 0.50 | 0.02 | 5.6 |  |
| 0.49 B, 0.91 GiB | fp16 | 16 | NPU | 15.4 | 0.99 | 0.965 | 1035.8 | 63 | 1.01 | 0.02 | 6.7 |  |
| 0.49 B, 0.91 GiB | fp16 | 32 | NPU | 15.5 | 0.99 | 0.484 | 2064.7 | 63 | 2.01 | 0.02 | 6.1 |  |
| 0.49 B, 0.91 GiB | fp16 | 64 | NPU | 15.9 | 1.02 | 0.248 | 4028.4 | 61 | 3.92 | 0.02 | 6.1 |  |
| 0.49 B, 0.91 GiB | fp16 | 128 | NPU | 21.2 | 1.35 | 0.166 | 6039.5 | 46 | 5.88 | 0.02 | 6.9 |  |
| 0.49 B, 0.45 GiB | int8 | 1 | NPU | 7.9 | 1.00 | 7.9 | 126.9 | 62 | 0.12 | 0.03 | 3.5 |  |
| 0.49 B, 0.45 GiB | int8 | 2 | NPU | 8.0 | 1.02 | 4.0 | 249.8 | 61 | 0.24 | 0.03 | 3.3 |  |
| 0.49 B, 0.45 GiB | int8 | 4 | NPU | 8.0 | 1.02 | 2.0 | 499.9 | 61 | 0.49 | 0.03 | 3.3 |  |
| 0.49 B, 0.45 GiB | int8 | 8 | NPU | 8.4 | 1.07 | 1.1 | 950.9 | 58 | 0.93 | 0.03 | 4.2 |  |
| 0.49 B, 0.45 GiB | int8 | 16 | NPU | 8.1 | 1.02 | 0.504 | 1983.6 | 60 | 1.93 | 0.03 | 3.5 |  |
| 0.49 B, 0.45 GiB | int8 | 32 | NPU | 8.3 | 1.05 | 0.259 | 3862.0 | 59 | 3.76 | 0.03 | 3.2 |  |
| 0.49 B, 0.45 GiB | int8 | 64 | NPU | 8.2 | 1.04 | 0.128 | 7783.7 | 59 | 7.57 | 0.03 | 3.4 |  |
| 0.49 B, 0.45 GiB | int8 | 128 | NPU | 11.9 | 1.51 | 0.093 | 10756.5 | 41 | 10.47 | 0.03 | 3.4 |  |
| 0.49 B, 0.23 GiB | lut4 | 1 | NPU | 4.1 | 1.00 | 4.1 | 243.9 | 59 | 0.24 | 0.05 | 3.5 |  |
| 0.49 B, 0.23 GiB | lut4 | 2 | NPU | 4.1 | 1.01 | 2.1 | 482.2 | 59 | 0.47 | 0.05 | 4.9 |  |
| 0.49 B, 0.23 GiB | lut4 | 4 | NPU | 4.2 | 1.03 | 1.1 | 950.4 | 58 | 0.92 | 0.05 | 5.0 |  |
| 0.49 B, 0.23 GiB | lut4 | 8 | NPU | 4.2 | 1.03 | 0.528 | 1893.6 | 58 | 1.84 | 0.05 | 5.0 |  |
| 0.49 B, 0.23 GiB | lut4 | 16 | NPU | 4.2 | 1.03 | 0.264 | 3787.6 | 58 | 3.69 | 0.05 | 4.9 |  |
| 0.49 B, 0.23 GiB | lut4 | 32 | NPU | 4.4 | 1.06 | 0.136 | 7343.6 | 56 | 7.15 | 0.05 | 4.9 |  |
| 0.49 B, 0.23 GiB | lut4 | 64 | NPU | 4.6 | 1.13 | 0.073 | 13766.7 | 52 | 13.40 | 0.05 | 5.2 |  |
| 0.49 B, 0.23 GiB | lut4 | 128 | NPU | 12.2 | 2.98 | 0.095 | 10481.6 | 20 | 10.20 | 0.02 | 6.4 |  |
| 0.49 B, 0.34 GiB | lut6 | 1 | NPU | 5.8 | 1.00 | 5.8 | 173.0 | 63 | 0.17 | 0.04 | 14.3 |  |
| 0.49 B, 0.34 GiB | lut6 | 2 | NPU | 6.0 | 1.04 | 3.0 | 334.1 | 61 | 0.33 | 0.04 | 15.3 |  |
| 0.49 B, 0.34 GiB | lut6 | 4 | NPU | 5.9 | 1.02 | 1.5 | 676.1 | 62 | 0.66 | 0.04 | 15.5 |  |
| 0.49 B, 0.34 GiB | lut6 | 8 | NPU | 5.9 | 1.01 | 0.732 | 1365.8 | 62 | 1.33 | 0.04 | 15.3 |  |
| 0.49 B, 0.34 GiB | lut6 | 16 | NPU | 6.1 | 1.05 | 0.381 | 2623.5 | 60 | 2.55 | 0.04 | 15.2 |  |
| 0.49 B, 0.34 GiB | lut6 | 32 | NPU | 5.9 | 1.02 | 0.184 | 5430.1 | 62 | 5.28 | 0.03 | 15.4 |  |
| 0.49 B, 0.34 GiB | lut6 | 64 | NPU | 5.9 | 1.02 | 0.092 | 10848.3 | 62 | 10.56 | 0.04 | 15.7 |  |
| 0.49 B, 0.34 GiB | lut6 | 128 | NPU | 12.8 | 2.21 | 0.100 | 10028.0 | 29 | 9.76 | 0.02 | [architecture](../docs/design/architecture.md) |  |
| 0.49 B, 0.45 GiB | lut8 | 1 | NPU | 7.7 | 1.00 | 7.7 | 130.7 | 64 | 0.13 | 0.03 | 4.9 |  |
| 0.49 B, 0.45 GiB | lut8 | 2 | NPU | 7.7 | 1.01 | 3.9 | 258.3 | 63 | 0.25 | 0.03 | 6.2 |  |
| 0.49 B, 0.45 GiB | lut8 | 4 | NPU | 7.9 | 1.03 | 2.0 | 507.7 | 62 | 0.49 | 0.03 | 6.2 |  |
| 0.49 B, 0.45 GiB | lut8 | 8 | NPU | 7.7 | 1.01 | 0.967 | 1033.6 | 63 | 1.01 | 0.03 | 6.4 |  |
| 0.49 B, 0.45 GiB | lut8 | 16 | NPU | 8.0 | 1.04 | 0.497 | 2011.4 | 61 | 1.96 | 0.03 | 6.1 |  |
| 0.49 B, 0.45 GiB | lut8 | 32 | NPU | 7.8 | 1.01 | 0.242 | 4125.7 | 63 | 4.01 | 0.03 | 6.3 |  |
| 0.49 B, 0.45 GiB | lut8 | 64 | NPU | 7.8 | 1.02 | 0.122 | 8197.1 | 62 | 7.98 | 0.03 | 6.4 |  |
| 0.49 B, 0.45 GiB | lut8 | 128 | NPU | 17.4 | 2.28 | 0.136 | 7342.8 | 28 | 7.15 | 0.02 | 7.7 |  |

× K=1 is the time per call over that of one token. Near 1, the call is bound by reading the weights and checking K drafted tokens costs about as much as generating one; tokens/s is then the most speculative decoding could reach if every draft were accepted. TFLOPS counts a multiply and an add per weight and token.

- fp16, 8 layers: still bound by reading the weights at K=128 (5.9 TFLOPS); a larger -tokens would show where the compute runs out.
- int8, 8 layers: still bound by reading the weights at K=128 (10.5 TFLOPS); a larger -tokens would show where the compute runs out.
- lut4, 8 layers: at most 13.4 TFLOPS. Reading the weights once (4.1 ms at K=1) takes as long as computing about 56 tokens at that rate: up to about that many tokens per call cost little more than one, and beyond it the time grows with K.
- lut6, 8 layers: at most 10.6 TFLOPS. Reading the weights once (5.8 ms at K=1) takes as long as computing about 63 tokens at that rate: up to about that many tokens per call cost little more than one, and beyond it the time grows with K.
- lut8, 8 layers: at most 8.0 TFLOPS. Reading the weights once (7.7 ms at K=1) takes as long as computing about 63 tokens at that rate: up to about that many tokens per call cost little more than one, and beyond it the time grows with K.

Placement: Apple, Core ML's compute plan (MLComputePlan) for every operation; Intel, NPU when the
driver compiler built the graph (it has no fallback); AMD, IRON's full ELF, all on the NPU. CPU use is the process CPU time divided by
the wall time while the model runs: near 0 when the NPU does the work, 1 or more when the CPU
does. weights GB/s is the weight bytes times calls per second; G weights/s the weights times calls
per second, the same in every format (a model of N billion weights decodes at that over N tokens/s). net W is the device's power meter
while running minus idle, when it has one.
