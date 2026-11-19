# hwnpu-dev probe

2026-11-19T23:47:21+09:00

| | |
|---|---|
| CPU | Apple M2 |
| machine | Mac14,2 |
| OS | macOS 27.2.1 |
| RAM | 16 GiB |
| NPU | Apple Neural Engine (16 cores) (coreml) |
| driver | Core ML / Version 27.2.1 (Build 26B131)  |
| hwnpu | v0.0.0-20261119144003-bcc3e1358aae, go1.27.4 |

Models: one token through Llama 3.2 1B sized layers (61 M weights per layer, random values), so a call is the work of generating one token without the attention over the context.

## K tokens per call (checking K drafted tokens)

| weights | format | K | placement | ms/call | × K=1 | ms/token | tokens/s | weights GB/s | TFLOPS | CPU use | compile s | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.49 B, 0.91 GiB | fp16 | 1 | NPU | 15.5 | 1.00 | 15.5 | 64.6 | 63 | 0.06 | 0.02 | 6.5 |  |
| 0.49 B, 0.91 GiB | fp16 | 2 | NPU | 15.4 | 0.99 | 7.7 | 130.2 | 63 | 0.13 | 0.02 | 8.2 |  |
| 0.49 B, 0.91 GiB | fp16 | 4 | NPU | 15.5 | 1.00 | 3.9 | 257.3 | 63 | 0.25 | 0.02 | 6.8 |  |
| 0.49 B, 0.91 GiB | fp16 | 8 | NPU | 15.5 | 1.00 | 1.9 | 516.3 | 63 | 0.50 | 0.02 | 7.1 |  |
| 0.49 B, 0.91 GiB | fp16 | 16 | NPU | 15.6 | 1.01 | 0.972 | 1028.6 | 63 | 1.00 | 0.02 | 7.0 |  |
| 0.49 B, 0.91 GiB | fp16 | 32 | NPU | 15.5 | 1.00 | 0.483 | 2070.8 | 63 | 2.02 | 0.02 | 5.5 |  |
| 0.49 B, 0.91 GiB | fp16 | 64 | NPU | 15.8 | 1.02 | 0.247 | 4052.9 | 62 | 3.94 | 0.02 | 5.5 |  |
| 0.49 B, 0.91 GiB | fp16 | 128 | NPU | 21.4 | 1.39 | 0.168 | 5969.7 | 45 | 5.81 | 0.02 | 7.3 |  |
| 0.49 B, 0.45 GiB | int8 | 1 | NPU | 8.0 | 1.00 | 8.0 | 125.6 | 61 | 0.12 | 0.03 | 3.0 |  |
| 0.49 B, 0.45 GiB | int8 | 2 | NPU | 7.9 | 0.99 | 3.9 | 254.3 | 62 | 0.25 | 0.03 | 3.6 |  |
| 0.49 B, 0.45 GiB | int8 | 4 | NPU | 8.0 | 1.00 | 2.0 | 501.9 | 61 | 0.49 | 0.04 | 3.2 |  |
| 0.49 B, 0.45 GiB | int8 | 8 | NPU | 8.0 | 1.01 | 1.0 | 995.7 | 61 | 0.97 | 0.04 | 3.4 |  |
| 0.49 B, 0.45 GiB | int8 | 16 | NPU | 7.9 | 1.00 | 0.497 | 2012.8 | 61 | 1.96 | 0.03 | 4.5 |  |
| 0.49 B, 0.45 GiB | int8 | 32 | NPU | 8.0 | 1.01 | 0.251 | 3977.8 | 60 | 3.87 | 0.04 | 3.2 |  |
| 0.49 B, 0.45 GiB | int8 | 64 | NPU | 8.4 | 1.06 | 0.131 | 7615.2 | 58 | 7.41 | 0.04 | 3.5 |  |
| 0.49 B, 0.45 GiB | int8 | 128 | NPU | 12.1 | 1.52 | 0.094 | 10607.7 | 40 | 10.32 | 0.03 | 3.9 |  |
| 0.49 B, 0.23 GiB | lut4 | 1 | NPU | 4.2 | 1.00 | 4.2 | 240.1 | 58 | 0.23 | 0.05 | 3.5 |  |
| 0.49 B, 0.23 GiB | lut4 | 2 | NPU | 4.2 | 1.00 | 2.1 | 479.2 | 58 | 0.47 | 0.05 | 4.9 |  |
| 0.49 B, 0.23 GiB | lut4 | 4 | NPU | 4.2 | 1.02 | 1.1 | 941.2 | 57 | 0.92 | 0.05 | 5.0 |  |
| 0.49 B, 0.23 GiB | lut4 | 8 | NPU | 4.3 | 1.02 | 0.531 | 1882.3 | 57 | 1.83 | 0.05 | 5.0 |  |
| 0.49 B, 0.23 GiB | lut4 | 16 | NPU | 4.3 | 1.03 | 0.269 | 3723.1 | 57 | 3.62 | 0.05 | 5.2 |  |
| 0.49 B, 0.23 GiB | lut4 | 32 | NPU | 4.4 | 1.04 | 0.136 | 7353.3 | 56 | 7.16 | 0.05 | 7.4 |  |
| 0.49 B, 0.23 GiB | lut4 | 64 | NPU | 4.7 | 1.13 | 0.073 | 13616.0 | 52 | 13.25 | 0.05 | 5.1 |  |
| 0.49 B, 0.23 GiB | lut4 | 128 | NPU | 12.3 | 2.95 | 0.096 | 10428.4 | 20 | 10.15 | 0.03 | 6.3 |  |

× K=1 is the time per call over that of one token. Near 1, the call is bound by reading the weights and checking K drafted tokens costs about as much as generating one; tokens/s is then the most speculative decoding could reach if every draft were accepted. TFLOPS counts a multiply and an add per weight and token.

- fp16, 8 layers: still bound by reading the weights at K=128 (5.8 TFLOPS); a larger -tokens would show where the compute runs out.
- int8, 8 layers: still bound by reading the weights at K=128 (10.3 TFLOPS); a larger -tokens would show where the compute runs out.
- lut4, 8 layers: at most 13.2 TFLOPS. Reading the weights once (4.2 ms at K=1) takes as long as computing about 57 tokens at that rate: up to about that many tokens per call cost little more than one, and beyond it the time grows with K.

Placement: Apple, Core ML's compute plan (MLComputePlan) for every operation; Intel, NPU when the
driver compiler built the graph (it has no fallback); AMD, IRON's full ELF, all on the NPU. CPU use is the process CPU time divided by
the wall time while the model runs: near 0 when the NPU does the work, 1 or more when the CPU
does. weights GB/s is the weight bytes times calls per second; G weights/s the weights times calls
per second, the same in every format (a model of N billion weights decodes at that over N tokens/s). net W is the device's power meter
while running minus idle, when it has one.
