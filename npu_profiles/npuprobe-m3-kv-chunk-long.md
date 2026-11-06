# hwnpu npuprobe

2026-10-30T07:29:08+09:00

| | |
|---|---|
| CPU | Apple M3 |
| machine | Mac15,13 |
| OS | macOS 27.0.1 |
| RAM | 16 GiB |
| NPU | Apple Neural Engine (16 cores) (coreml) |
| driver | Core ML / Version 27.0.1 (Build 26A434)  |
| hwnpu | v0.0.0-20261029214225-d75b75fcf086+dirty d75b75fcf0862f5b09216fc8deb6232880852523, go1.27.0 |

Models: one token through Llama 3.2 1B sized layers (61 M weights per layer, random values), so a call is the work of generating one token without the attention over the context.

## KV cache: host buffers against Core ML states

| cache | KV bytes | mode | placement | ms/token | CPU use | error |
|---|---|---|---|---|---|---|
| 32768 | 384.0 MiB | host-chunk | NPU | 84.01 | 0.14 |  |
| 65536 | 768.0 MiB | host-chunk | NPU | 201.63 | 0.29 |  |

One token through lut4 layers of the bench size with attention over a cache of the given length. host binds the cache as inputs and outputs the token's keys and values; state keeps it in Core ML states (MLState) and writes the token in with a one-hot row.

Placement: Apple, Core ML's compute plan (MLComputePlan) for every operation; Intel, NPU when the
driver compiler built the graph (it has no fallback). CPU use is the process CPU time divided by
the wall time while the model runs: near 0 when the NPU does the work, 1 or more when the CPU
does. weights GB/s is the weight bytes times calls per second. net W is the device's power meter
while running minus idle, when it has one.
