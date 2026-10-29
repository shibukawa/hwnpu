# hwnpu npuprobe

2026-10-29T21:37:52+09:00

| | |
|---|---|
| CPU | Apple M3 |
| machine | Mac15,13 |
| OS | macOS 27.0.1 |
| RAM | 16 GiB |
| NPU | Apple Neural Engine (16 cores) (coreml) |
| driver | Core ML / Version 27.0.1 (Build 26A434)  |
| hwnpu | v0.0.0-20261029110559-d8e3662c55f3+dirty d8e3662c55f3fe4379068b0553c77fd0ddd47046, go1.27.0 |

Models: one token through Llama 3.2 1B sized layers (61 M weights per layer, random values), so a call is the work of generating one token without the attention over the context.

## A model split into parts run one after another

| format | weights | parts | layout | placement of the parts | ms/token | parts alone added up | tokens/s | weights GB/s | CPU use | peak RSS | mJ/token | error after each part | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lut4 | 64 layers, 1.81 GiB | 2 | functions | 32 layers 0.91 GiB CPU 513<br>32 layers 0.91 GiB CPU 514 | 12549.3 | 126.7 | 0.1 | 0 | 0.26 | 5.26 GiB |  | vs GPU: 8.24%, 10.87% |  |
| lut4 | 64 layers, 1.81 GiB | 2 | files | 32 layers 0.91 GiB NPU<br>32 layers 0.91 GiB NPU | 30.6 | 31.0 | 32.6 | 64 | 0.04 | 1.03 GiB |  | vs GPU: 3.32%, 4.91% |  |
| lut4 | 64 layers, 1.81 GiB | 4 | functions | 16 layers 0.45 GiB CPU 257<br>16 layers 0.45 GiB CPU 257<br>16 layers 0.45 GiB CPU 257<br>16 layers 0.45 GiB CPU 258 | 12362.0 | 139.9 | 0.1 | 0 | 0.30 | 6.42 GiB |  | vs GPU: 5.92%, 8.24%, 9.67%, 10.87% |  |
| lut4 | 64 layers, 1.81 GiB | 4 | files | 16 layers 0.45 GiB NPU<br>16 layers 0.45 GiB NPU<br>16 layers 0.45 GiB NPU<br>16 layers 0.45 GiB NPU | 31.3 | 31.4 | 32.0 | 62 | 0.06 | 0.55 GiB |  | vs GPU: 2.31%, 3.32%, 4.21%, 4.91% |  |

Every part is loaded before any is timed. ms/token runs each part once, the output buffer of one bound as the input of the next; parts alone added up is the sum of each part's own time. Layout functions writes the parts as the functions of one package, loaded one function at a time. Peak RSS is the largest resident memory of the process that loaded and ran the parts (it leaves out the system's own Core ML and Neural Engine processes). The error after each part compares the hidden state with the same parts run one at a time on the reference units (rms of the difference over rms of the reference).

Placement: Apple, Core ML's compute plan (MLComputePlan) for every operation; Intel, NPU when the
driver compiler built the graph (it has no fallback). CPU use is the process CPU time divided by
the wall time while the model runs: near 0 when the NPU does the work, 1 or more when the CPU
does. weights GB/s is the weight bytes times calls per second. net W is the device's power meter
while running minus idle, when it has one.
