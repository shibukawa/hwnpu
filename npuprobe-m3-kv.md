# hwnpu npuprobe

2026-10-29T23:02:25+09:00

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
| 256 | 5.5 MiB | host | NPU | 7.08 | 0.16 |  |
| 256 | 5.5 MiB | state | NPU | 6.64 | 0.06 |  |
| 256 | 5.5 MiB | host-grouped | CPU 332 | 26.01 | 1.01 |  |
| 256 | 5.5 MiB | state-grouped | error |  |  | coreml: load /var/folders/vk/36d9fgzs1cs1nt8m36cqsk7m0000gn/T/hwnpu-npuprobe-1143380936/probe_kv_state-grouped_11_256.mlpackage: Failed to build the model execution plan using a model architecture fil… |
| 1024 | 22.0 MiB | host | NPU | 10.96 | 0.18 |  |
| 1024 | 22.0 MiB | state | NPU | 10.04 | 0.04 |  |
| 1024 | 22.0 MiB | host-grouped | CPU 332 | 31.59 | 1.01 |  |
| 1024 | 22.0 MiB | state-grouped | error |  |  | coreml: load /var/folders/vk/36d9fgzs1cs1nt8m36cqsk7m0000gn/T/hwnpu-npuprobe-1143380936/probe_kv_state-grouped_11_1024.mlpackage: Failed to build the model execution plan using a model architecture fi… |
| 4096 | 88.0 MiB | host | NPU | 27.04 | 0.16 |  |
| 4096 | 88.0 MiB | state | NPU | 26.78 | 0.02 |  |
| 4096 | 88.0 MiB | host-grouped | CPU 332 | 63.41 | 0.93 |  |
| 4096 | 88.0 MiB | state-grouped | error |  |  | coreml: load /var/folders/vk/36d9fgzs1cs1nt8m36cqsk7m0000gn/T/hwnpu-npuprobe-1143380936/probe_kv_state-grouped_11_4096.mlpackage: Failed to build the model execution plan using a model architecture fi… |
| 8192 | 176.0 MiB | host | NPU | 49.45 | 0.15 |  |
| 8192 | 176.0 MiB | state | NPU | 49.27 | 0.02 |  |
| 8192 | 176.0 MiB | host-grouped | CPU 332 | 85.74 | 0.99 |  |
| 8192 | 176.0 MiB | state-grouped | error |  |  | coreml: load /var/folders/vk/36d9fgzs1cs1nt8m36cqsk7m0000gn/T/hwnpu-npuprobe-1143380936/probe_kv_state-grouped_11_8192.mlpackage: Failed to build the model execution plan using a model architecture fi… |

One token through lut4 layers of the bench size with attention over a cache of the given length. host binds the cache as inputs and outputs the token's keys and values; state keeps it in Core ML states (MLState) and writes the token in with a one-hot row.

Placement: Apple, Core ML's compute plan (MLComputePlan) for every operation; Intel, NPU when the
driver compiler built the graph (it has no fallback). CPU use is the process CPU time divided by
the wall time while the model runs: near 0 when the NPU does the work, 1 or more when the CPU
does. weights GB/s is the weight bytes times calls per second. net W is the device's power meter
while running minus idle, when it has one.
