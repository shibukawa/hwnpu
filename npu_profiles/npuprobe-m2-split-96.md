# hwnpu npuprobe

2026-11-09T00:06:45+09:00

| | |
|---|---|
| CPU | Apple M2 |
| machine | Mac14,2 |
| OS | macOS 27.2 |
| RAM | 16 GiB |
| NPU | Apple Neural Engine (16 cores) (coreml) |
| driver | Core ML / Version 27.2 (Build 26B123)  |
| hwnpu | (devel), go1.27.3 |

Models: one token through Llama 3.2 1B sized layers (61 M weights per layer, random values), so a call is the work of generating one token without the attention over the context.

## A model split into parts run one after another

| format | weights | parts | layout | placement of the parts | ms/token | parts alone added up | tokens/s | weights GB/s | CPU use | peak RSS | Neural Engine | mJ/token | error after each part | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lut4 | 96 layers, 2.72 GiB | 8 | files | 12 layers 0.34 GiB NPU<br>12 layers 0.34 GiB NPU<br>12 layers 0.34 GiB NPU<br>12 layers 0.34 GiB NPU<br>12 layers 0.34 GiB NPU<br>12 layers 0.34 GiB NPU<br>12 layers 0.34 GiB NPU<br>12 layers 0.34 GiB NPU | 46.8 | 47.4 | 21.4 | 62 | 0.03 | 0.44 GiB | 2.74 GiB |  |  |  |

Every part is loaded before any is timed. ms/token runs each part once, the output buffer of one bound as the input of the next; parts alone added up is the sum of each part's own time. Layout functions writes the parts as the functions of one package, loaded one function at a time. Peak RSS is the largest resident memory of the process that loaded and ran the parts (it leaves out the system's own Core ML and Neural Engine processes). Neural Engine is what loading the parts and running them once added to the wired kernel memory of the Neural Engine's driver (com.apple.driver.AppleH11ANEInterface, read with mach_memory_info): where the weights on the Neural Engine are, in no process's RSS. The error after each part compares the hidden state with the same parts run one at a time on the reference units (rms of the difference over rms of the reference).

Placement: Apple, Core ML's compute plan (MLComputePlan) for every operation; Intel, NPU when the
driver compiler built the graph (it has no fallback). CPU use is the process CPU time divided by
the wall time while the model runs: near 0 when the NPU does the work, 1 or more when the CPU
does. weights GB/s is the weight bytes times calls per second. net W is the device's power meter
while running minus idle, when it has one.
