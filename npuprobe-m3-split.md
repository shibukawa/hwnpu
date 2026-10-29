# hwnpu npuprobe

2026-10-29T19:52:44+09:00

| | |
|---|---|
| CPU | Apple M3 |
| machine | Mac15,13 |
| OS | macOS 27.0.1 |
| RAM | 16 GiB |
| NPU | Apple Neural Engine (16 cores) (coreml) |
| driver | Core ML / Version 27.0.1 (Build 26A434)  |
| hwnpu | (devel), go1.27.0 |

Models: one token through Llama 3.2 1B sized layers (61 M weights per layer, random values), so a call is the work of generating one token without the attention over the context.

## A model split into parts run one after another

| format | weights | parts | placement of the parts | ms/token | parts alone added up | tokens/s | weights GB/s | CPU use | mJ/token | error after each part | error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lut4 | 64 layers, 1.81 GiB | 2 | 32 layers 0.91 GiB NPU<br>32 layers 0.91 GiB NPU | 30.0 | 30.5 | 33.3 | 65 | 0.02 |  | vs GPU: 3.32%, 4.91% |  |
| lut4 | 64 layers, 1.81 GiB | 4 | 16 layers 0.45 GiB NPU<br>16 layers 0.45 GiB NPU<br>16 layers 0.45 GiB NPU<br>16 layers 0.45 GiB NPU | 30.2 | 30.4 | 33.1 | 64 | 0.02 |  | vs GPU: 2.31%, 3.32%, 4.21%, 4.91% |  |

Every part is loaded before any is timed. ms/token runs each part once, the output buffer of one bound as the input of the next; parts alone added up is the sum of each part's own time. The error after each part compares the hidden state with the same parts run one at a time on the reference units (rms of the difference over rms of the reference).

Placement: Apple, Core ML's compute plan (MLComputePlan) for every operation; Intel, NPU when the
driver compiler built the graph (it has no fallback). CPU use is the process CPU time divided by
the wall time while the model runs: near 0 when the NPU does the work, 1 or more when the CPU
does. weights GB/s is the weight bytes times calls per second. net W is the device's power meter
while running minus idle, when it has one.
