# Proving hwnpu on your machine

How to check, on a real machine, that hwnpu runs its NPU, how fast an LLM sized model runs there, how large a model the NPU accepts and how much energy it uses, all with `go run` from the module path. Nothing has to be cloned or downloaded besides Go modules: the tools generate their models in Go, so neither Python nor model files are needed.

日本語: [PROBE.ja.md](PROBE.ja.md)

## What to run

| Machine | 1. Does it work | 2. Speed and size limit | 3. Energy |
|---|---|---|---|
| Apple Silicon Mac | `cmd/report` | `cmd/npuprobe` | `cmd/powerbench -decode 8,16 -gpu` (sudo) |
| Intel Core Ultra, Linux | `cmd/report` | `cmd/npuprobe` | `cmd/npuprobe` run as root (RAPL) |
| Intel Core Ultra, Windows | `cmd/report` | `cmd/npuprobe` | not available yet |
| AMD Ryzen AI, Linux | `cmd/report` | not available: the AMD backend runs kernels, not models | |

Set up the machine first: [INSTALL_APPLE.md](INSTALL_APPLE.md), [INSTALL_INTEL.md](INSTALL_INTEL.md), [INSTALL_AMD.md](INSTALL_AMD.md). Go 1.27 or later is needed; an older `go` command downloads it by itself (`GOTOOLCHAIN=auto`, the default).

### 1. Does it work

```bash
go run github.com/shibukawa/hwnpu/cmd/report@latest -o report.md
```

It compiles small embedded models for every NPU found, compares the results with fp32 references and measures latency. If this fails, the steps below fail the same way; the report says why (driver, library, placement).

### 2. Speed and size limit

```bash
go run github.com/shibukawa/hwnpu/cmd/npuprobe@latest
```

It takes about 40 minutes and writes `npuprobe-<machine>-<time>.md` and `.json` to the current directory; what was measured is kept if it is interrupted with Ctrl-C. `bench` or `limit` after the command name runs one part only. `split` checks a model over the limit run in parts (see below). The speculative decoding measurement (verify, below) runs only when `verify` is given.

### 3. Energy

On Apple, energy needs `powermetrics`, which needs root; the tool asks for the sudo password once:

```bash
go run github.com/shibukawa/hwnpu/cmd/powerbench@latest -decode 8,16 -gpu -o power.md
```

On Intel with Linux, npuprobe reads the RAPL energy counters when it can, which needs root. Build the command, then run it with sudo:

```bash
GOBIN=$PWD go install github.com/shibukawa/hwnpu/cmd/npuprobe@latest
```

```bash
sudo ./npuprobe bench
```

## The models

Every model has the shape of one decoding step of an LLM: one token goes through layers the size of Llama 3.2 1B (hidden 2048, MLP 8192, grouped query attention with 512 wide keys and values), so each call reads every weight once, as generating a token does. A layer has 61 M weights; 8 layers are 0.49 B weights and 16 layers 0.97 B. The weights are random (the numbers are about speed and placement, not accuracy) and attention over a KV cache is left out, since with one query it is a small read next to the weights.

| Format | Bytes per weight | Apple (Core ML) | Intel (OpenVINO IR) |
|---|---|---|---|
| fp16 | 2 | fp16 constants | fp16 constants |
| int8 | 1 | int8, one scale per output row | i8 → Convert → Multiply, one scale per row |
| lut4 | 0.5 | 4 bit lookup tables, a 16 entry table per 16 rows | — |
| int4 | 0.5 | int4, one scale per 32 inputs (CPU and GPU only) | i4 → Convert → Multiply, one scale per row |

The generators are in [internal/probe](internal/probe/model.go). The Intel models were checked with OpenVINO 2026.4, which read them and ran them on the CPU.

## What npuprobe measures

**bench** runs 8 and 16 layers in every format on the NPU and, on Apple, on the CPU and the GPU through Core ML for comparison. Each model runs for at least 3 seconds after two warm-up calls.

**limit** grows the model in each format (2, 4, 8, 12, 16, 24, 32, … layers) until it leaves the NPU, then bisects to one layer between the largest model on the NPU and the smallest not on it. It then loads several models just under the boundary at once, enough to hold twice the failing size, and runs each again with all of them loaded. If they all stay on the NPU at the same speed, the limit is per model and a larger network can be split into several models. Models that leave the NPU run only once, since they can be very slow on the CPU.

**split** runs only when asked for with `npuprobe split`. It builds a model larger than the limit (64 layers by default, 1.8 GiB of lut4) and splits it into parts small enough for the NPU (2 and 4 by default). The output buffer of each part is bound as the input of the next, and the parts run in order, as a split LLM generates one token. Each layer has the same weights however the model is split, so running the parts in order computes the whole model. It checks three things:

- whether every part stays on the NPU with all of them loaded
- the time per token (every part once) against the parts' own times added up. A large gap means the weights are loaded again at every switch between parts
- the error of the hidden state after each part against the same parts run one at a time on the GPU (rms of the difference over rms of the reference), to show that nothing breaks at the boundaries

It also records the peak resident memory (RSS) of the process that ran the parts, without the system's own Core ML and Neural Engine processes. `-layout functions` writes the parts as the functions of one package instead of a package each, and loads each function as a model of its own.

```bash
go run github.com/shibukawa/hwnpu/cmd/npuprobe@latest split
go run github.com/shibukawa/hwnpu/cmd/npuprobe@latest split -split-layers 128 -parts 2,3,4   # machines with a 2 GiB limit such as the M4 (3.7 GiB)
```

**verify** runs the bench models on the NPU with K tokens per call (1, 2, 4 and 8 by default): the call that checks K drafted tokens in speculative decoding, which reads every weight once whatever K is. While the time per call barely grows with K, the call is bound by reading the weights and checking K drafts costs about as much as generating one token. The K at which the time starts to grow is about the most speculative decoding can gain. Attention is only the per-token stand-in; there is none among the drafted tokens. The formats are those the NPU runs (fp16, int8 and lut4 on Apple).

Each measurement runs in a child process of its own, so a framework or driver that aborts (Core ML's GPU path has asserted on some graphs) costs one row of the report, with the error, not the run.

How to read the columns:

| Column | Meaning |
|---|---|
| placement | Apple: Core ML's compute plan (`MLComputePlan`) for every operation, e.g. `NPU` or `CPU 258`. Intel: `NPU` when the driver compiler built the graph; it has no fallback, so anything else is a compile error. |
| ms/token, tokens/s | Wall time per call. Without the attention over the context, so an upper bound for real decoding. In verify, the time per call divided by K: the bound if every draft were accepted. |
| × K=1 | verify: the time per call over that at K=1. Near 1, the K tokens are checked almost for free. |
| weights GB/s | Weight bytes times calls per second: how close a run is to the memory bandwidth. |
| CPU use | Process CPU time divided by wall time while the model runs. Near 0 when the NPU (or the GPU) does the work, 1 or more when the CPU does. It confirms the placement without root. |
| net W, mJ/token | Device power meter while running minus idle, when there is a readable meter. |

Options:

| Flag | Default | |
|---|---|---|
| `-formats` | Apple `fp16,int8,lut4,int4`, Intel `fp16,int8,int4` | int4 is not tried on the Neural Engine |
| `-layers` | `8,16` | bench and verify layer counts |
| `-tokens` | `1,2,4,8` | verify: tokens per call, K |
| `-baselines` | `cpu,gpu` | Apple: Core ML units to compare with; empty for none |
| `-max-gb` | a quarter of the RAM on Apple, a sixth on Intel, at most 8 | limit: largest weight size tried. Apple needs free disk space of about twice this |
| `-several` | `true` | limit: load several models at once after the sweep |
| `-split-layers` | `64` | split: layers of the whole model |
| `-parts` | `2,4` | split: numbers of parts to try; `1` runs the model unsplit |
| `-layout` | `files` | split: how the parts are written. `files`, a package per part; `functions`, the functions of one package (Core ML, macOS 15 and later). `files,functions` compares both |
| `-ref` | `gpu` on Apple, `none` elsewhere | split: Core ML units for the reference output (`cpu`, `gpu` or `none`). The CPU expands lut4 to fp16, so a large model can run out of memory there |
| `-run` | `3s` | time per model |
| `-dir`, `-o` | temporary directory, `npuprobe-<machine>-<time>.md` | |

## Results so far

### Apple M3 (MacBook Air, 16 GB, macOS 27.0.1)

Size limit, one model at a time:

| Format | Largest on the Neural Engine | Smallest not on it |
|---|---|---|
| fp16 | 8 layers, 0.91 GiB | 9 layers, 1.02 GiB |
| int8 | 17 layers, 0.96 GiB | 18 layers, 1.02 GiB |
| lut4 | 35 layers, 0.99 GiB | 36 layers, 1.02 GiB |

- The Neural Engine takes a model of less than 1 GiB of stored weights, in any format. A larger model is placed entirely on the CPU.
- The limit is per model: three 0.91 GiB fp16 models loaded at once (2.7 GiB) all stayed on the Neural Engine, each at 45 ms per call, the same as alone.
- A lut4 model that falls to the CPU is extremely slow (5 to 13 seconds per call at 1.0 to 1.4 GiB): probably because the CPU path expands the tables to fp16, four times the size, on a 16 GB machine.

Speed and energy at 16 layers (0.97 B weights). Speed from `npuprobe bench`, energy from `powerbench` on the same model shapes:

| format | weights | asked for | placed on | ms/token | tokens/s | weights GB/s | CPU use | mJ/token |
|---|---|---|---|---|---|---|---|---|
| fp16 | 1.81 GiB | NPU | CPU (over the limit) | 26.9 | 37 | 72 | 3.72 | |
| fp16 | 1.81 GiB | CPU | CPU | 29.6 | 34 | 66 | 3.29 | 276 |
| fp16 | 1.81 GiB | GPU | GPU | 24.5 | 41 | 79 | 0.13 | 69 |
| int8 | 0.91 GiB | NPU | NPU | 41.6 | 24 | 23 | 0.01 | 95 |
| int8 | 0.91 GiB | CPU | CPU | 29.1 | 34 | 33 | 3.43 | 262 |
| int8 | 0.91 GiB | GPU | GPU | 42.0 | 24 | 23 | 0.08 | 388 |
| **lut4** | 0.45 GiB | **NPU** | **NPU** | **7.5** | **133** | 65 | 0.02 | **37** |
| lut4 | 0.45 GiB | CPU | CPU | 27.3 | 37 | 18 | 3.67 | 256 |
| lut4 | 0.45 GiB | GPU | GPU (plan not readable) | 65.1 | 15 | 7 | 0.14 | 256 |
| int4 | 0.45 GiB | CPU | CPU | 28.1 | 36 | 17 | 3.53 | 300 |
| **int4** | 0.45 GiB | **GPU** | **GPU** | **8.5** | **117** | 57 | 0.47 | **35** |

- Only lut4 runs at full speed on the Neural Engine: 4 bit weights halve the bytes read per token against int8, and decoding is bound by the weight reads.
- The Neural Engine and the GPU with int4 cost about the same energy per token (37 and 35 mJ); the CPU costs about 7 times more.
- fp16 at 1 B is over the size limit and runs on the CPU.
- The CPU is limited by the arithmetic, not the bytes: about 30 ms per token in every format, at 3.3 to 3.7 cores busy. Quantization only pays on the NPU and the GPU.

A model over the limit, split by layers (`npuprobe split`, lut4, 64 layers, 1.81 GiB; report: [npuprobe-m3-split.md](npuprobe-m3-split.md)):

| parts | each part | placement | ms/token | parts alone added up | weights GB/s | CPU use | error vs GPU after each part |
|---|---|---|---|---|---|---|---|
| 2 | 32 layers, 0.91 GiB | all NPU | 30.0 | 30.5 | 65 | 0.02 | 3.32%, 4.91% |
| 4 | 16 layers, 0.45 GiB | all NPU | 30.2 | 30.4 | 64 | 0.02 | 2.31%, 3.32%, 4.21%, 4.91% |

- Splitting by layers takes a model past the 1 GiB limit at full speed. 1.81 GiB in one model runs on the CPU; in parts, every part stays on the Neural Engine and the model reads its weights at 65 GB/s, the same as a single 16 layer model (7.5 ms, 0.45 GiB).
- The parts cost nothing to switch between: one token through all of them takes no longer than the parts timed alone added up, so no part reloads its weights. Four parts take 0.2 ms more than two.
- Splitting loses nothing: the error after 32 and 64 layers is the same to every digit with 2 and 4 parts. The error is the Neural Engine's arithmetic against the GPU growing with depth, not the boundaries.
- Smaller parts compile faster in total (8 s each for 16 layers, 16 to 30 s each for 32 layers).

Against the parts written as the functions of one package (`-layout functions,files`; report: [npuprobe-m3-functions.md](npuprobe-m3-functions.md)):

| parts | written as | placement | ms/token | peak RSS of the process |
|---|---|---|---|---|
| 2 | functions of one package | all CPU | 12549 | 5.26 GiB |
| 2 | a package each | all NPU | 30.6 | 1.03 GiB |
| 4 | functions of one package | all CPU | 12362 | 6.42 GiB |
| 4 | a package each | all NPU | 31.3 | 0.55 GiB |

- The 1 GiB limit counts the whole package, not the function loaded. Split a model into a package per part.
- With a package per part, the peak memory of the process is about the weights of one part: it falls with more parts, at almost the same speed.

### Apple M4 (measured by a user; report not included yet)

A 3.6 GiB model (`npuprobe split -split-layers 128`) split several ways:

| parts | each part | ms/token |
|---|---|---|
| 2 | 1.8 GiB | 58.3 |
| 4 | 0.9 GiB | 58.8 |
| 8 | 0.45 GiB | 59.4 |
| 16 | 0.23 GiB | 60.6 |

- No error with any split, and the memory used fell sharply as the parts got smaller (the measure and the numbers are not recorded yet).
- Each boundary costs about 0.16 ms: 16 parts are only 4% slower than 2.
- A 1.8 GiB part ran, so the M4 takes at least 1.8 GiB per model.

### Wanted

M4 and later Apple chips (does the 1 GiB limit change with the chip or the RAM?), Intel Core Ultra on Linux and Windows (the Intel backend has not run on real hardware yet), and AMD Ryzen AI for `cmd/report`. Open an issue with the `.md` files.

## Limits of these measurements

- Random weights: the numbers say nothing about accuracy. For accuracy, see [MODEL.md](MODEL.md).
- No KV cache: real decoding adds the attention over the context, small next to the weights at short contexts.
- Core ML's compute plan is an estimate of placement; CPU use is the cross-check.
- `powerbench -decode` and the Intel paths of npuprobe have not been run on hardware other than the M3 yet.
