# hwnpu npuprobe

2026-11-04T10:49:20+09:00

| | |
|---|---|
| CPU | Apple M3 |
| machine | Mac15,13 |
| OS | macOS 27.0.1 |
| RAM | 16 GiB |
| NPU | Apple Neural Engine (16 cores) (coreml) |
| driver | Core ML / Version 27.0.1 (Build 26A434)  |
| hwnpu | (devel), go1.27.3 |

Models: one token through Llama 3.2 1B sized layers (61 M weights per layer, random values), so a call is the work of generating one token without the attention over the context.

## A model split at its attention: weighted parts and the chunk attention interleaved

| weights | parts | Q | attention | placement of the parts | of the attention | used | chunks | ms/step weights alone | attention alone | interleaved | ms per seam | seams | CPU use | peak RSS | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lut4 40 layers, 1.13 GiB | 7 | 8 | grouped, 6 layers, chunks of 8192 | NPU | NPU | 8192 | 1 | 20.21 | 8.83 | 29.01 | -0.003 | 12 | 0.18 | 0.25 GiB |  |
| lut4 40 layers, 1.13 GiB | 7 | 8 | grouped, 6 layers, chunks of 8192 | NPU | NPU | 65536 | 8 | 20.11 | 70.57 | 95.33 | +0.387 | 12 | 0.38 | 0.84 GiB |  |
| lut4 40 layers, 1.13 GiB | 7 | 64 | grouped, 6 layers, chunks of 8192 | NPU | NPU | 8192 | 1 | 26.84 | 28.72 | 49.91 | -0.470 | 12 | 0.16 | 0.25 GiB |  |
| lut4 40 layers, 1.13 GiB | 7 | 64 | grouped, 6 layers, chunks of 8192 | NPU | NPU | 65536 | 8 | 27.08 | 210.04 | 239.78 | +0.221 | 12 | 0.22 | 0.85 GiB |  |

The weighted parts are the bench layers split into one more part than there are attention layers, every part loaded with the attention program at once. A step runs the parts one after another (weights alone), the attention of every layer over the used chunks with the host's merge (attention alone), or the two interleaved as a model split at its attention would run them: part, attention, part, … (interleaved). Each is the fastest of two rounds. ms per seam is what interleaving adds over the two apart, divided by the switches between a part and the attention; near 0, the Neural Engine keeps both kinds of program, and a reload would show as milliseconds.

Placement: Apple, Core ML's compute plan (MLComputePlan) for every operation; Intel, NPU when the
driver compiler built the graph (it has no fallback). CPU use is the process CPU time divided by
the wall time while the model runs: near 0 when the NPU does the work, 1 or more when the CPU
does. weights GB/s is the weight bytes times calls per second. net W is the device's power meter
while running minus idle, when it has one.
