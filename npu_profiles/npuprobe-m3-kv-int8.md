# hwnpu npuprobe

2026-10-29T23:43:09+09:00

| | |
|---|---|
| CPU | Apple M3 |
| machine | Mac15,13 |
| OS | macOS 27.0.1 |
| RAM | 16 GiB |
| NPU | Apple Neural Engine (16 cores) (coreml) |
| driver | Core ML / Version 27.0.1 (Build 26A434)  |
| hwnpu | v0.0.0-20261029134442-f30d16b64ae8+dirty f30d16b64ae8f2471d54e28f49426457f472e557, go1.27.0 |

Models: one token through Llama 3.2 1B sized layers (61 M weights per layer, random values), so a call is the work of generating one token without the attention over the context.

## KV cache: host buffers against Core ML states

| cache | KV bytes | mode | placement | ms/token | CPU use | error |
|---|---|---|---|---|---|---|
| 1024 | 22.0 MiB | host | NPU | 10.82 | 0.17 |  |
| 1024 | 11.2 MiB | host-int8 | NPU | 11.91 | 0.27 |  |
| 4096 | 88.0 MiB | host | NPU | 27.58 | 0.16 |  |
| 4096 | 44.7 MiB | host-int8 | NPU | 29.45 | 0.26 |  |
| 8192 | 176.0 MiB | host | NPU | 46.55 | 0.14 |  |
| 8192 | 89.4 MiB | host-int8 | NPU | 50.80 | 0.24 |  |

One token through lut4 layers of the bench size with attention over a cache of the given length. host binds the cache as inputs and outputs the token's keys and values; state keeps it in Core ML states (MLState) and writes the token in with a one-hot row.

Placement: Apple, Core ML's compute plan (MLComputePlan) for every operation; Intel, NPU when the
driver compiler built the graph (it has no fallback). CPU use is the process CPU time divided by
the wall time while the model runs: near 0 when the NPU does the work, 1 or more when the CPU
does. weights GB/s is the weight bytes times calls per second. net W is the device's power meter
while running minus idle, when it has one.
