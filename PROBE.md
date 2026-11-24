# Proving BonsaiAI on your machine

How to check, on a real machine, that BonsaiAI runs its NPU, how fast an LLM sized model runs there, how large a model the NPU accepts and how much energy it uses, all with `go run` from the module path. Nothing has to be cloned or downloaded besides Go modules: the tools generate their models in Go, so neither Python nor model files are needed (on AMD's NPU, IRON's Python environment, which BonsaiAI needs there anyway).

日本語: [PROBE.ja.md](PROBE.ja.md)

## What to run

| Machine | 1. Does it work | 2. Speed and size limit | 3. Energy |
|---|---|---|---|
| Apple Silicon Mac | `bonsaiai-dev report` | `bonsaiai-dev probe` | `bonsaiai-dev powerbench -decode 8,16 -gpu` (sudo) |
| Intel Core Ultra, Linux | `bonsaiai-dev report` | `bonsaiai-dev probe` | `bonsaiai-dev probe` run as root (RAPL) |
| Intel Core Ultra, Windows | `bonsaiai-dev report` | `bonsaiai-dev probe` | not available yet |
| AMD Ryzen AI, Linux | `bonsaiai-dev report` | `bonsaiai-dev probe` with IRON's environment (below) | the same run (amdgpu PPT, no root) |

Set up the machine first: [INSTALL_APPLE.md](INSTALL_APPLE.md), [INSTALL_INTEL.md](INSTALL_INTEL.md), [INSTALL_AMD.md](INSTALL_AMD.md). Go 1.27 or later is needed; an older `go` command downloads it by itself (`GOTOOLCHAIN=auto`, the default).

### 1. Does it work

```bash
go run github.com/shibukawa/bonsaiai/cmd/bonsaiai-dev@latest report -o report.md
```

It compiles small embedded models for every NPU found, compares the results with fp32 references and measures latency. If this fails, the steps below fail the same way; the report says why (driver, library, placement).

### 2. Speed and size limit

```bash
go run github.com/shibukawa/bonsaiai/cmd/bonsaiai-dev@latest probe
```

It takes about 40 minutes and writes `npuprobe-<machine>-<time>.md` and `.json` to the current directory; what was measured is kept if it is interrupted with Ctrl-C. `bench` or `limit` after the command name runs one part only. `split` checks a model over the limit run in parts (see below). The speculative decoding measurement (verify, below) runs only when `verify` is given.

On AMD's NPU the probe exports its shapes with IRON, the same exporters bonsaiai-chat runs on its first start (built into bonsaiai-dev), so IRON's virtual environment is needed ([INSTALL_AMD.md](INSTALL_AMD.md)); no model files are downloaded:

```bash
BONSAIAI_IRON_ENV=~/IRON/ironenv go run github.com/shibukawa/bonsaiai/cmd/bonsaiai-dev@latest probe
```

It takes about 20 minutes (each of its 39 graphs is exported in 5 to 55 seconds and timed for 3) and runs bench, verify and attn (below). Close other programs that use the NPU first (`pgrep -fa 'bonsaiai|iron_'`): another model running on it moved a step of FLUX.2 klein from 1.3 to 2.3 s.

### 3. Energy

On Apple, energy needs `powermetrics`, which needs root; the tool asks for the sudo password once:

```bash
go run github.com/shibukawa/bonsaiai/cmd/bonsaiai-dev@latest powerbench -decode 8,16 -gpu -o power.md
```

On Intel with Linux, bonsaiai-dev probe reads the RAPL energy counters when it can, which needs root. Build the command, then run it with sudo:

```bash
GOBIN=$PWD go install github.com/shibukawa/bonsaiai/cmd/bonsaiai-dev@latest probe
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

## What bonsaiai-dev probe measures

**bench** runs 8 and 16 layers in every format (on Apple fp16, int8, lut4, lut6, lut8 and int4: lut6 and lut8 are lut4's tables with 64 and 256 entries, the weights of `bonsaiai-dev convert -bits 6` and `8`) on the NPU and, on Apple, on the CPU and the GPU through Core ML for comparison. Each model runs for at least 3 seconds after two warm-up calls.

**limit** grows the model in each format (2, 4, 8, 12, 16, 24, 32, … layers) until it leaves the NPU, then bisects to one layer between the largest model on the NPU and the smallest not on it. It then loads several models just under the boundary at once, enough to hold twice the failing size, and runs each again with all of them loaded. If they all stay on the NPU at the same speed, the limit is per model and a larger network can be split into several models. Models that leave the NPU run only once, since they can be very slow on the CPU.

**split** runs only when asked for with `bonsaiai-dev probe split`. It builds a model larger than the limit (64 layers by default, 1.8 GiB of lut4) and splits it into parts small enough for the NPU (2 and 4 by default). The output buffer of each part is bound as the input of the next, and the parts run in order, as a split LLM generates one token. Each layer has the same weights however the model is split, so running the parts in order computes the whole model. It checks three things:

- whether every part stays on the NPU with all of them loaded
- the time per token (every part once) against the parts' own times added up. A large gap means the weights are loaded again at every switch between parts
- the error of the hidden state after each part against the same parts run one at a time on the GPU (rms of the difference over rms of the reference), to show that nothing breaks at the boundaries

It also records the peak resident memory (RSS) of the process that ran the parts, without the system's own Core ML and Neural Engine processes. `-layout functions` writes the parts as the functions of one package instead of a package each, and loads each function as a model of its own.

```bash
go run github.com/shibukawa/bonsaiai/cmd/bonsaiai-dev@latest probe split
go run github.com/shibukawa/bonsaiai/cmd/bonsaiai-dev@latest probe split -split-layers 128 -parts 2,3,4   # machines with a 2 GiB limit such as the M4 (3.7 GiB)
```

**kv** runs only when asked for, as `bonsaiai-dev probe kv`. It passes one token through 11 lut4 layers of the bench size with real attention, over a KV cache of each length (256, 1024 and 4096 tokens by default), the cache held either in host buffers bound as inputs and outputs (`host`) or in Core ML states, MLState (`state`). `-kv-cache`, `-kv-layers` and `-kv-modes` change them. `-kv-modes host-mha` gives each of the 16 query heads its own keys and values instead of repeating the 4 cached heads (a cache 4 times as large), to tell the cost of the repeat and of reading the cache apart. `host-int8` passes the cache as int8 with an fp16 scale per token and casts it back in the program (Core ML specification version 10, macOS 26 and later). `host-chunk` splits the cache into chunks of at most 16384 tokens, each passed as an input of its own (`host-chunk8192` sets the chunk length): the form that puts contexts over the 16384 dimension limit on the Neural Engine.

**attn** runs only when asked for, as `bonsaiai-dev probe attn` (Core ML only). It builds attention without weights over one chunk of a KV cache: the program a model split at its attention would call once per used chunk and layer (docs/model-fit.md T2). It checks where the program is placed and times one call with one chunk bound. Core ML puts the forms with little work (one row over a chunk of 1024) on the CPU; those are measured on to give the CPU's time. Then, for each used length (4096, 16384, 65536 and 262144 tokens by default), it times one step's attention: every layer's chunks in buffers of their own, the outputs merged on the host in float32. The report gives the time per step, the part of it spent merging, and the process's peak resident memory, which show whether the cost follows the tokens used instead of the capacity, what binding many buffers costs, and whether the inputs are copied. `-attn-chunks` (chunk lengths, 1024, 2048, 4096, 8192 and 16384 by default), `-attn-tokens` (query rows per call, 1, 8 and 64), `-attn-forms` (`grouped` and `tile`, `grouped` by default), `-attn-used` and `-attn-layers` (6 by default, LFM3-1.7B's full attention layers) change them. At 262144 the 6 layers hold 3 GiB of buffers at once.

**seam** runs only when asked for, as `bonsaiai-dev probe seam` (Core ML only). It builds a model split at its attention: weighted parts (40 lut4 layers by default, 1.14 GiB, about LFM3-1.7B in lut6) in one more part than there are attention layers, with the chunk attention of attn between each two, all loaded at once. It times a step three ways: the parts alone, the attention alone, and the two interleaved (part, attention, part, …). What the interleaved step takes over the two apart, divided by the switches, is the cost of one seam between a program with weights and one without. Alternating two programs with weights has cost a reload of the Neural Engine, 35 ms a call; a seam within one chain cost 0.1 ms. `-seam-layers`, `-seam-format`, `-seam-tokens` (8 and 64 by default), `-seam-chunk` (8192), `-seam-used` (8192 and 65536), `-seam-form` and `-attn-layers` change it.

**cache** runs only when asked for, as `bonsaiai-dev probe cache` (Core ML only). It measures the Neural Engine's own cache of compiled programs, the reason a program that BonsaiAI's compile cache holds loads in 0.1 s on one run and compiles again for one or two minutes on the next ([docs/handoff-ane-cache.md](docs/handoff-ane-cache.md)). A configuration writes N programs of S GiB once (`-cache-sizes`, 0.5 and 1 GiB of fp16 by default; N from `-cache-totals`, 1 to 8 GiB), then loads them in order and runs each once in a new process per round (`-cache-rounds`, 3), keeping BonsaiAI's compile cache; a load and first call of `-cache-hit` (1 s) or less is a hit. The report gives the hits of every round, the largest total that stays warm per size, and the size of `~/Library/Caches/NAME/com.apple.e5rt.e5bundlecache` after each round. `-cache-hold open` keeps every program open instead of closing each before the next, `-cache-order reverse` reverses every other round (an LRU then hits more), `-cache-passes 2` loads again within the same process, `-cache-names other` adds a round under a copy of the executable named other (then one under its own name again), and `-cache-keep` with `-dir` keeps the programs for a later run, after a restart. The probe holds BonsaiAI's Neural Engine lock while it runs; quit other programs that use the Neural Engine. The defaults write up to 8 GiB of programs into a temporary directory and take an hour or more.

**resident** runs only when asked for, as `bonsaiai-dev probe resident` (Intel and Apple). It measures how much the NPU holds at once: programs of `-resident-size` GiB (0.25 GiB of fp16 by default) are loaded one after another and kept open, each run once, until a load fails or `-resident-max-gb` (60 % of the RAM, at most 16) is reached; then every held program runs again. After running out of memory, the oldest program is closed and the failed one loaded again: whether closing a program gives its memory back, which bonsaiai-image's reloading of Anima's programs counts on. `-resident-hold close` loads, runs and closes each in turn up to the same total, to tell a limit on what is held from one on what has been loaded. `-resident-formats int8` tells whether the limit counts the weights' bytes, and `-resident-run=false` loads without running. The child reports each load as it goes, so when the driver kills it (0xc0000005 on Intel) the report keeps what was held before. Intel's NPU (Core Ultra 5 135U) failed with ERROR_OUT_OF_HOST_MEMORY at about 4.6 GB of Anima's fp16 programs held at once. The report gives each load's time and, on Windows, the process's private memory after it. A load that does not fail is not a program that stays on the NPU: Core ML evicts without an error and loads the program onto the Neural Engine again at its next call, so an Apple Mac reaches `-resident-max-gb` however little the Neural Engine keeps. With `-resident-cycle` (the default; open only), after each load every held program runs in turn twice and the second pass is timed: the report gives the most held at full speed in turn and the least held where a call in turn was over twice as slow (the M1 ran parts in turn 12 times slower past about 2.9 GiB). Running in turn stops after 3 slow steps in a row, the loads go on. On Apple each load also records what the wired memory of the Neural Engine's driver grew by: what it really holds, not the weights added up. `-resident-size` takes several sizes, comma separated (`0.57,1.2`: the sizes of Anima's DiT groups and text encoder), each a run of its own.

**verify** runs the bench models on the NPU with K tokens per call (1 to 128 by default): the call that checks K drafted tokens in speculative decoding, or reads a chunk of the prompt, which reads every weight once whatever K is. While the time per call barely grows with K, the call is bound by reading the weights and checking K drafts costs about as much as generating one token. Past some K the compute runs out and the time grows with K. The report gives that rate in TFLOPS and the K at which reading the weights and computing take as long, from the ratio of the NPU's compute to its weight reading: about the most speculative decoding can gain, and the prompt chunk that uses the NPU best. Attention is only the per-token stand-in; there is none among the drafted tokens. The formats are those the NPU runs (fp16, int8, lut4, lut6 and lut8 on Apple).

**On AMD's NPU** (Ryzen AI, XDNA2) there is no graph backend: BonsaiAI runs IRON exports there, so the probe builds its graphs with `tools/amd/iron_probe.py` from random weights, with the operators BonsaiAI's exports use, and times them through `internal/llm/iron` as BonsaiAI runs a model. The numbers are therefore the ceilings of the paths the models take, not of IRON's samples ([docs/amd-xdna2-bench.md](docs/amd-xdna2-bench.md), measured before BonsaiAI had its own kernels).

- **bench**: the same 8 and 16 layers in q4 (q4nx, 5 bits a weight), q6 (6.5 bits), q8 (8.5 bits) and bf16, one row a call, two ways. `layers` runs each layer's seven linears and the SwiGLU's product one after another, as BonsaiAI's exports of the chat models run a decode step: IRON loads every operator that follows another onto the cores again. `stream` puts the same bytes in GEMVs of one shape back to back, which are not loaded again: the time is the weights' transfer, the ceiling of a decode step run as one resident array (gpt-oss-20b's `GptOssStep`, [docs/design/moe.md](docs/design/moe.md)).
- **verify**: the layers with K rows a call, on the path each K takes in bonsaiai: 1 row the decode step's QGEMV, 2 to 8 rows of q8 or q6 QGEMVRows (the call that checks drafts, `--verify`), multiples of 256 the prompt chunk's GEMM (q4 and q8 dequantize their weights on the device at every call; bfp16, the image DiTs' weights packed on the host, are read by the GEMM as they are). The default K are 1, 2, 4, 8, 256, 1024 and 4096, at the first `-layers` (8) unless `-layers` is given.
- **attn**: the image DiTs' attention as FLUX.2 klein's export has it (`iron_mha.ImageMHA`, QKᵀ and P·V through bfp16, the softmax's exponential on one bf16 limb), 24 heads of 128, 1536 and 4608 rows (512² and 1024² with 512 rows of text; `-image-rows`, `-image-heads`, `-image-dim`).

`bonsaiai-dev probe` on AMD runs all three; `bench`, `verify` or `attn` runs one. `-formats` takes q4, q6, q8, bf16 and bfp16 there.

Each measurement runs in a child process of its own, so a framework or driver that aborts (Core ML's GPU path has asserted on some graphs) costs one row of the report, with the error, not the run.

How to read the columns:

| Column | Meaning |
|---|---|
| placement | Apple: Core ML's compute plan (`MLComputePlan`) for every operation, e.g. `NPU` or `CPU 258`. Intel: `NPU` when the driver compiler built the graph; it has no fallback, so anything else is a compile error. |
| ms/token, tokens/s | Wall time per call. Without the attention over the context, so an upper bound for real decoding. In verify, the time per call divided by K: the bound if every draft were accepted. |
| × K=1 | verify: the time per call over that at K=1. Near 1, the K tokens are checked almost for free. |
| TFLOPS | verify: a multiply and an add per weight and token, over the time per call. It stops growing with K once the compute runs out. |
| weights GB/s | Weight bytes times calls per second: how close a run is to the memory bandwidth. |
| CPU use | Process CPU time divided by wall time while the model runs. Near 0 when the NPU (or the GPU) does the work, 1 or more when the CPU does. It confirms the placement without root. |
| net W, mJ/token | Device power meter while running minus idle, when there is a readable meter. |

Options:

| Flag | Default | |
|---|---|---|
| `-formats` | Apple `fp16,int8,lut4,int4`, Intel `fp16,int8,int4` | int4 is not tried on the Neural Engine |
| `-layers` | `8,16` | bench and verify layer counts |
| `-tokens` | `1,2,4,8,16,32,64,128` | verify: tokens per call, K |
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

The Apple results below were measured through Core ML, before BonsaiAI's chat and image models moved to Core AI (macOS 27, [docs/design/apple-coreai.md](docs/design/apple-coreai.md)). What depends on the hardware still holds: the Neural Engine reads weights at about 62 to 64 GB/s (44 to 48 on the M1; Core AI's chat models read theirs at 48 GB/s on an M1, [docs/design/performance.md](docs/design/performance.md)). What depends on Core ML does not carry over: the 1 GiB limit is Core ML's (one Core AI program holds 2.5 to 3 GiB on an M3), and Core AI's large programs run in turn pay 50 to 70 ms each instead of the M1's limit on the total. The probe has no Core AI path yet.

### AMD Ryzen AI 9 365 (Strix Point, XDNA2, 32 cores; Linux, amdxdna 0.7, firmware 1.1.2.64)

`bonsaiai-dev probe` with the defaults, 2026-11-19, no other program on the NPU (report: [npuprobe-strix.md](npu_profiles/npuprobe-strix.md)). A decode step, one row a call, 16 layers (0.97 B weights) in every format the exports use:

| format | bits a weight | weights | form | ms/token | tokens/s | weights GB/s | G weights/s | net W | mJ/token |
|---|---|---|---|---|---|---|---|---|---|
| q4 | 5 | 0.57 GiB | layers | 31.3 | 32 | 19 | 31 | 9.4 | 293 |
| q4 | 5 | 0.57 GiB | **stream** | **11.6** | **87** | **53** | **84** | 11.8 | **136** |
| q6 | 6.5 | 0.74 GiB | layers | 40.2 | 25 | 20 | 24 | 9.0 | 363 |
| q6 | 6.5 | 0.74 GiB | stream | 20.7 | 48 | **38** | 47 | 9.5 | 197 |
| q8 | 8.5 | 0.96 GiB | layers | 38.2 | 26 | 27 | 25 | 8.7 | 332 |
| q8 | 8.5 | 0.96 GiB | stream | 19.0 | 53 | 54 | 51 | 10.9 | 207 |
| bf16 | 16 | 1.81 GiB | layers | 41.0 | 24 | 47 | 24 | 9.1 | 374 |
| bf16 | 16 | 1.81 GiB | stream | 35.3 | 28 | 55 | 28 | 9.3 | 328 |

G weights/s is the weights read a second: a model of N billion weights decodes at about that over N tokens/s, whatever its format. q4 is q4nx (4 bit codes with a bf16 scale and minimum per 32), q6 and q8 6 and 8 bit codes with a bf16 scale per 32.

- **The NPU reads weights at 53 to 55 GB/s in q4, q8 and bf16**, when the operators are not loaded again (stream), as the DMA's ceiling of 55 to 57 GB/s. The Neural Engine reads at 62 to 65 GB/s (M2, M3, M4), so with 4-bit weights the M3 is 1.5 times as fast per token at the ceiling (7.5 ms against 11.6, 129 G weights/s against 84; q4nx is 5 bits a weight with its scales and minimums, lut4 a little over 4).
- **q6 is slower than q8 although it reads fewer bytes**: it alone stays under the ceiling, at 38 GB/s, because its kernel unpacks four 6 bit codes from three bytes on the cores, and that work, not the transfer, bounds it. It reads 47 G weights/s against q8's 51 (20.7 against 19.0 ms at the ceiling, 40.2 against 38.2 in the layers form), and checking 8 drafts costs 3.9 steps against q8's 2.9 (below). On AMD q6 only saves memory (a quarter less than q8); q8 is both faster and closer to the original. q4 is the fastest by far.
- **As the chat models' exports run a step, the ceiling is far off**: q4 reaches 19 GB/s, a third of it. A layer is 8 operators, and an operator that follows another is loaded onto the cores again (about 67 µs plus 1.2 µs a KB of its programs, [docs/design/moe.md](docs/design/moe.md)): 20 of q4's 31 ms. In that form every format but q4 takes about the same time (38 to 41 ms), the reloads hiding what the formats save. bf16 hides it best (47 GB/s), its transfers being 3.6 times as long. gpt-oss-20b, whose step is one resident array, runs at 93 % of the ceiling; for the other models that form is the largest gain left on AMD (2.7 times for q4, by these numbers).
- Energy: a token of q4 at the ceiling costs 136 mJ above idle, against 37 mJ on the M3's Neural Engine (lut4, `bonsaiai-dev powerbench`). The meters differ (amdgpu's PPT is the whole package, powermetrics' figure the Neural Engine's own), so take the ratio as rough. The layers form costs more energy, not only time (293 against 136 mJ).

K rows a call (8 layers, 0.49 B weights):

| format | K | path | ms/call | × K=1 | ms/token | TFLOPS |
|---|---|---|---|---|---|---|
| q8 | 1 | QGEMV | 19.2 | 1.00 | 19.2 | 0.05 |
| q8 | 2 | QGEMVRows (checking drafts) | 19.9 | 1.04 | 9.9 | 0.10 |
| q8 | 4 | QGEMVRows | 24.4 | 1.27 | 6.1 | 0.16 |
| q8 | 8 | QGEMVRows | 54.6 | 2.85 | 6.8 | 0.14 |
| q8 | 256 | dequantize and GEMM (prompt chunk) | 88.0 | 4.59 | 0.34 | 2.8 |
| q8 | 4096 | 〃 | 361.9 | 18.9 | 0.088 | 11.0 |
| q6 | 1 | QGEMV | 20.2 | 1.00 | 20.2 | 0.05 |
| q6 | 2 | QGEMVRows | 20.8 | 1.03 | 10.4 | 0.09 |
| q6 | 4 | QGEMVRows | 25.0 | 1.24 | 6.2 | 0.16 |
| q6 | 8 | QGEMVRows | 79.2 | 3.92 | 9.9 | 0.10 |
| q6 | 256 | dequantize and GEMM | 92.3 | 4.57 | 0.36 | 2.7 |
| q6 | 4096 | 〃 | 366.8 | 18.2 | 0.090 | 10.9 |
| q4 | 1 | QGEMV | 15.9 | 1.00 | 15.9 | 0.06 |
| q4 | 256 | DequantBFP and GEMM (prompt chunk) | 85.9 | 5.39 | 0.34 | 2.9 |
| q4 | 4096 | 〃 | 320.5 | 20.1 | 0.078 | 12.4 |
| bfp16 | 256 | GEMM, weights packed on the host | 27.5 | | 0.11 | 9.0 |
| bfp16 | 1024 | 〃 | 75.3 | | 0.073 | 13.2 |
| **bfp16** | **4096** | 〃 | **261.7** | | **0.064** | **15.2** |

q4 has no call for 2 to 8 rows (QGEMVRows is q8 and q6 only).

- **Speculative decoding has little room on AMD**: 2 drafts cost 1.04 steps and 4 cost 1.27, but 8 cost 2.85 (q6: 3.92), and a prompt chunk's GEMM is 4.6 steps whatever its rows up to 256. The best a call can advance is about 4 tokens (6.1 ms a token if every draft were kept). The Neural Engine checks 64 tokens for 1.10 calls (M3, below), so a draft model or MTP drafts pay much more there.
- **The compute ceiling is AMD's**: 15.2 TFLOPS at 4096 rows with the weights packed as bfp16, against 13.7 on the M3's Neural Engine (K=64, below) and about 5 on the M1's (Anima's MLP, [docs/design/performance.md](docs/design/performance.md)). This is why the image models, bound by compute at 4096 rows and more, became faster on AMD than on the Neural Engine once the GEMMs read bfp16 ([docs/design/image-amd.md](docs/design/image-amd.md)): FLUX.2 klein 1024² in about 29 s against the M1's 41 s, a Qwen-Image step 8.4 s against 15.9.
- **A prompt chunk of q4, q6 or q8 pays its dequantize at every call**: 256 rows take 86 to 92 ms against bfp16's 28 ms, three times as long, and the chunk only nears bfp16's rate at 4096 rows. Holding a chat model's weights for the prompt in bfp16 too (9 bits a weight more) would read prompts up to 3 times as fast; not done.

The image DiTs' attention (24 heads of 128): 12.8 ms at 1536 rows (2.3 TFLOPS), 81.8 ms at 4608 rows (3.2 TFLOPS). It is a fifth of the GEMMs' rate: what is left of a DiT block's time on AMD is the attention (FLUX.2 klein's single block: attention 118 of 198 ms before the exponential's change, [docs/design/image-amd.md](docs/design/image-amd.md)).

Where the power goes (amdgpu's gpu_metrics, the SMU's own split, sampled every 50 ms; `bonsaiai-dev probe` reports it on AMD). At rest the package draws 4.4 W:

| run | package W | above rest | NPU rail | CPU cores | the rest of the package | DRAM GB/s |
|---|---|---|---|---|---|---|
| q4 decode, layers (as the chat models run) | 13.3 | +9.0 | 1.5 (17 %) | +1.7 | +5.7 | 20 |
| q4 decode, stream (the ceiling) | 17.2 | +12.8 | 0.75 (6 %) | +1.8 | +10.3 | 48 |
| bf16 decode, stream | 14.1 | +9.7 | 0.62 (6 %) | +1.6 | +7.5 | 50 |
| bfp16 GEMM, 4096 rows (15.2 TFLOPS) | 24.7 | +20.3 | 2.0 (10 %) | +0.7 | +17.6 | 50 |
| image attention, 4608 rows (3.2 TFLOPS) | 15.5 | +11.1 | 0.60 (5 %) | +0.4 | +10.1 | 17 |

- **The NPU's rail is not the NPU's power**: it is 1.5 W in the layers form and falls to 0.6 to 0.75 W as the work grows (stream, attention), and is at most 2.5 W at 15 TFLOPS. It follows the operators' loads more than the work; the AIE array draws mostly from what gpu_metrics leaves in the rest. The 15 % read before was most likely this rail in the chat models' layers form ([docs/design/architecture.md](docs/design/architecture.md): about 1 W while decoding).
- So the rest is not the memory alone. Fitted over the 39 runs, the NPU rail and the rest above rest are about 6.4 W for being active (the fabric and the NPU's clocks up), plus 0.056 W per GB/s of DRAM traffic (56 pJ a byte) and 0.66 W per TFLOPS (0.66 pJ a FLOP), within 1.2 W for most runs (the q4 stream and the attention 1.4 to 1.8 W over: dequantizing and the softmax are work the FLOPs do not count). The DRAM chips themselves are outside the package and not in these numbers.
- By that fit, **moving the weights is a small part even at the ceiling**: about 2.7 W of the q4 stream's 11 W, against 6.4 W for being on. In the GEMM at 15 TFLOPS the compute is about 10 of 19.6 W, half: the jobs bound by compute (image models, prompts) now spend their energy on the NPU's arithmetic, decoding on staying awake. Running the same work in less time is what saves energy: the q4 stream costs 136 mJ a token against the layers form's 293.

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

Speed and energy at 16 layers (0.97 B weights). Speed from `bonsaiai-dev probe bench`, energy from `bonsaiai-dev powerbench` on the same model shapes:

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

- Only the tables run at full speed on the Neural Engine (lut4 here; lut6 and lut8 too, below. On the M2 and the M1, int8 and fp16 did too; see their sections. On the M3 with macOS 27.2 they stay slow, so the chip makes the difference, not the OS): 4 bit weights halve the bytes read per token against int8, and decoding is bound by the weight reads.
- The Neural Engine and the GPU with int4 cost about the same energy per token (37 and 35 mJ); the CPU costs about 7 times more.
- fp16 at 1 B is over the size limit and runs on the CPU.
- The CPU is limited by the arithmetic, not the bytes: about 30 ms per token in every format, at 3.3 to 3.7 cores busy. Quantization only pays on the NPU and the GPU.

A model over the limit run in parts (`bonsaiai-dev probe split`, 64 layers of lut4, 1.81 GiB, 3.9 B weights; report: [npuprobe-m3-split.md](npu_profiles/npuprobe-m3-split.md)):

| split | each part | placement | ms/call of each part | ms/token | parts added up | weights GB/s | CPU use | error against the GPU after each part |
|---|---|---|---|---|---|---|---|---|
| 2 ways | 32 layers, 0.91 GiB | all NPU | 15.2, 15.3 | 30.0 | 30.5 | 65 | 0.02 | 3.32%, 4.91% |
| 4 ways | 16 layers, 0.45 GiB | all NPU | 7.6, 7.6, 7.7, 7.5 | 30.2 | 30.4 | 64 | 0.02 | 2.31%, 3.32%, 4.21%, 4.91% |

- The 1 GiB limit is per model, and splitting gets past it: 1.81 GiB of weights, which fall to the CPU as one model, all ran on the Neural Engine.
- Switching between parts costs nothing visible. The time per token equals the parts' own times added up (30.0 and 30.5 ms), and the weight bandwidth is that of a single 16 layer model (65 GB/s). The weights are not loaded again at each switch.
- The split does not change the result. The error after 32 layers is 3.32% whether the model is split in 2 or in 4, and 4.91% at the end in both, so nothing breaks at the boundaries. The error grows slowly with the layers; the GPU result it is compared with has fp16 error of its own.
- 64 layers is the size of a 4 B model (random weights, no attention over the context). Split into lut4 parts under 1 GiB, a 4 B class model should run on the M3's Neural Engine at about 30 ms per token (about 33 tokens/s).

Against the parts written as the functions of one package (`-layout functions,files`; report: [npuprobe-m3-functions.md](npu_profiles/npuprobe-m3-functions.md)):

| parts | written as | placement | ms/token | peak RSS of the process |
|---|---|---|---|---|
| 2 | functions of one package | all CPU | 12549 | 5.26 GiB |
| 2 | a package each | all NPU | 30.6 | 1.03 GiB |
| 4 | functions of one package | all CPU | 12362 | 6.42 GiB |
| 4 | a package each | all NPU | 31.3 | 0.55 GiB |

- The 1 GiB limit counts the whole package, not the function loaded. Split a model into a package per part.
- With a package per part, the peak memory of the process is about the weights of one part: it falls with more parts, at almost the same speed.

K tokens per call (`bonsaiai-dev probe verify -formats lut4 -layers 16`, 0.97 B weights in lut4):

| K | ms/call | × K=1 | ms/token | TFLOPS |
|---|---|---|---|---|
| 1 | 8.2 | 1.00 | 8.2 | 0.24 |
| 8 | 8.3 | 1.01 | 1.04 | 1.9 |
| 16 | 8.3 | 1.01 | 0.52 | 3.8 |
| 32 | 7.9 | 0.96 | 0.25 | 7.9 |
| 64 | 9.1 | 1.10 | 0.14 | 13.7 |
| 128 | 22.1 | 2.69 | 0.17 | 11.3 |

- Up to 64 tokens per call take about as long as one: the call is bound by reading the weights (59 GB/s). Generating one token uses 0.24 TFLOPS, under 2% of the Neural Engine's compute.
- At 64 tokens the compute reaches 13.7 TFLOPS, three quarters of the M3's 18 TOPS (fp16). Reading the weights once takes as long as computing about 58 tokens at that rate.
- 128 tokens take 2.7 times as long as 64, slower per token than 64: past the limit, splitting a prompt into calls of 64 is faster than calls of 128.
- So speculative decoding can check up to about 60 drafted tokens for the time of one, and a prompt is read fastest in chunks of about 64 tokens (about 7,000 tokens/s for 1 B weights). LFM3-1.7B converted with `bonsaiai-dev convert -seq` behaved alike: 13.7 ms for 1 token, 15.2 ms for 48, 30.0 ms for 96 (11 TFLOPS).

Every format again on macOS 27.2.1 (`bonsaiai-dev probe bench` and `bonsaiai-dev probe verify -layers 8`, 2026-11-20). 16 layers (0.97 B) on the Neural Engine; fp16 at 16 layers falls to the CPU, so its row is 8 layers:

| format | bits a weight | weights | ms/token | weights GB/s | G weights/s | M2 G weights/s |
|---|---|---|---|---|---|---|
| lut4 | 4 | 0.45 GiB | **7.6** | 64 | **128** | 122 |
| lut6 | 6 | 0.68 GiB | 11.1 | 66 | 88 | 85 |
| lut8 | 8 | 0.91 GiB | 14.9 | 65 | 65 | 64 |
| int8 | 8 | 0.91 GiB | 42.0 | 23 | 23 | 63 |
| fp16 (8 layers) | 16 | 0.91 GiB | 44.1 | 22 | 11 | 31 |
| int4 on the GPU | 4 | 0.45 GiB | 8.8 | 55 | 111 | 76 |

- **On the M3 every table runs at full speed, not only lut4**: lut6 and lut8 read at 65 to 66 GB/s, as on the M2. int8 and fp16 stay slow (23 and 22 GB/s). So lut8, the same bytes as int8, is 2.8 times as fast as int8 on the M3 (14.9 against 42.0 ms): 8 bit weights should be lut8, not int8, on the Neural Engine.
- The CPU takes about 26 ms at 16 layers in every format, and the GPU is fastest with int4 (8.8 ms), between lut4 and lut6 on the Neural Engine.

K tokens per call (8 layers, 0.49 B weights):

| format | K=1 ms | K=8 × K=1 | K=64 × K=1 | K=64 TFLOPS | K=128 × K=1 | K=128 TFLOPS |
|---|---|---|---|---|---|---|
| fp16 | 43.9 | 1.00 | 1.00 | 1.4 | 1.07 | 2.7 |
| int8 | 21.1 | 1.00 | 1.01 | 2.9 | 1.12 | 5.3 |
| lut4 | 3.8 | 1.03 | 1.34 | **12.2** | 3.18 | 10.3 |
| lut6 | 5.7 | 1.00 | 1.00 | 10.9 | 1.95 | 11.2 |
| lut8 | 7.6 | 1.00 | 1.00 | 8.2 | 2.13 | 7.7 |

- lut6's K=1 measured 7.5 ms, the others 5.7 to 5.8 from K=2 on; the table takes 5.7 as one call.
- lut6 and lut8 check 64 tokens for one call, as on the M2. lut4 already gains at 64 (1.34 calls), and every table falls past 64 (2 to 3.2 times at 128).
- As on the M2, lut8 reaches less compute than the smaller tables at 128 tokens (7.7 TFLOPS against 10 to 11).

### Apple M2 (MacBook Air, 16 GB, macOS 27.2)

`bonsaiai-dev probe` with the defaults (bench and limit; report: [npuprobe-m2.md](npu_profiles/npuprobe-m2.md)). 16 layers (0.97 B), against the M3 above:

| format | weights | unit | M2 ms/token | M3 ms/token | M2 weights GB/s | M3 weights GB/s |
|---|---|---|---|---|---|---|
| lut4 | 0.45 GiB | NPU | 7.8 | 7.5 | 63 | 65 |
| int8 | 0.91 GiB | NPU | 15.6 | 41.6 | 62 | 23 |
| int4 | 0.45 GiB | GPU | 8.5 | 8.5 | 57 | 57 |
| fp16 | 1.81 GiB | GPU | 25.5 | 24.5 | 76 | 79 |
| lut4 | 0.45 GiB | GPU | 55.4 | 65.1 | 9 | 7 |
| every format | | CPU | 35 or so | 27 to 30 | | |

- The size limit is the M3's to the layer: fp16 8 layers (0.91 GiB) on the Neural Engine and 9 (1.02 GiB) not, int8 17 and 18, lut4 35 (0.99 GiB) and 36. Three fp16 models of 0.91 GiB loaded at once all stay on the Neural Engine at 15 to 16 ms a call, as alone.
- lut4 runs at the M3's speed, so decoding bound by the weights should be about as fast. The CPU is about a fifth slower.
- int8 and fp16 run at full speed too: every format reads its weights at about 60 GB/s (0.49 B fp16: 15.9 ms). On the M3 only the tables (lut4, lut6, lut8) did, and the chip makes the difference, not the OS: rerun on the M3 with the M2's macOS 27.2 (the same Core ML, Build 26B123), `bonsaiai-dev probe bench` still gives int8 on the Neural Engine 42.2 ms at 16 layers (23 GB/s) and fp16 45.6 ms at 8 layers (21 GB/s), as on 27.0.1, and only lut4 7.6 ms (64 GB/s) (report: [npuprobe-m3-macos27.2.md](npu_profiles/npuprobe-m3-macos27.2.md)).
- lut4 put on the CPU over the limit is as slow as on the M3 (40 layers 6.3 s a call, 48 layers 17.7 s).
- Compiling takes longer than on the M3 (lut4 32 layers: 25 s).
- **No limit on the total of parts run in turn, unlike the M1** (`bonsaiai-dev probe split -formats lut4 -parts 8 -ref none -split-layers N`, 2026-11-09; the same sizes as the M1's table below; reports: [96](npu_profiles/npuprobe-m2-split-96.md), [100](npu_profiles/npuprobe-m2-split-100.md), [104](npu_profiles/npuprobe-m2-split-104.md), [128](npu_profiles/npuprobe-m2-split-128.md) layers). Up to 3.62 GiB, the parts loaded at once and run one after another take their own times added up, at 63 GB/s:

| total | parts | Neural Engine holds | ms/token | parts alone added up | weights GB/s |
|---|---|---|---|---|---|
| 2.72 GiB | 8 × 0.34 | 2.74 GiB | 46.8 | 47.4 | 62 |
| 2.83 GiB | 8 × 0.35 | 2.85 GiB | 48.6 | 49.2 | 63 |
| 2.95 GiB | 8 × 0.37 | 2.97 GiB | 50.5 | 51.2 | 63 |
| 3.62 GiB | 8 × 0.45 | 3.65 GiB | 62.2 | 62.4 | 63 |

  So the M1 alone slows down past about 2.9 GiB; the M2, the M3 (Qwen3.5-4B in lut8, 3.96 GiB in one program, at its expected 8 tokens/s, [docs/design/llm-qwen.md](docs/design/llm-qwen.md)) and the M4 (3.62 GiB) do not.

Every format, as on AMD above (`bonsaiai-dev probe bench`, macOS 27.2.1, 2026-11-20; report: [npuprobe-m2-formats.md](npu_profiles/npuprobe-m2-formats.md)). 16 layers (0.97 B) on the Neural Engine; fp16 at 16 layers (1.81 GiB) is over the limit and falls to the CPU, so its row is 8 layers:

| format | bits a weight | weights | ms/token | weights GB/s | G weights/s | AMD at its ceiling (stream), G weights/s |
|---|---|---|---|---|---|---|
| lut4 | 4 | 0.45 GiB | **8.0** | 61 | **122** | q4: 84 |
| lut6 | 6 | 0.68 GiB | 11.4 | 64 | 85 | q6: 47 |
| lut8 | 8 | 0.91 GiB | 15.3 | 64 | 64 | q8: 51 |
| int8 | 8 | 0.91 GiB | 15.4 | 63 | 63 | q8: 51 |
| fp16 (8 layers) | 16 | 0.91 GiB | 15.9 | 61 | 31 | bf16: 28 |

- **Every format reads its weights at 61 to 64 GB/s, so the time follows the bytes**: lut6 lies between lut4 and lut8 as its 6 bits say (11.4 ms, 1.43 times lut4). Unlike AMD's q6, which is slower than q8, the 6 bit tables cost the Neural Engine nothing to unpack. lut8 runs as fast as int8.
- Per weight the M2 is 1.45 times AMD's ceiling in 4 bits, 1.8 times in 6, 1.25 times in 8 and 1.1 times in 16; against AMD's layers form, as its chat models run now (q4: 31 G weights/s), 3.9 times in 4 bits.
- The GPU is fastest with int4 (12.8 ms), slower than the Neural Engine with lut4 or lut6. The GPU run of int8 at 16 layers stopped on an exception in Core ML (NSException).

K tokens per call in every format the Neural Engine runs (`bonsaiai-dev probe verify -layers 8`, 0.49 B weights, macOS 27.2.1, 2026-11-20; report: [npuprobe-m2-verify.md](npu_profiles/npuprobe-m2-verify.md)):

| format | K=1 ms | G weights/s | K=8 × K=1 | K=64 × K=1 | K=64 TFLOPS | K=128 × K=1 | K=128 TFLOPS |
|---|---|---|---|---|---|---|---|
| fp16 | 15.6 | 31 | 0.99 | 1.02 | 3.9 | 1.35 | 5.9 |
| int8 | 7.9 | 62 | 1.07 | 1.04 | 7.6 | 1.51 | 10.5 |
| lut4 | 4.1 | 119 | 1.03 | 1.13 | **13.4** | 2.98 | 10.2 |
| lut6 | 5.8 | 84 | 1.01 | 1.02 | 10.6 | 2.21 | 9.8 |
| lut8 | 7.7 | 64 | 1.01 | 1.02 | 8.0 | 2.28 | 7.1 |

- **Every format checks 64 tokens for about the time of one** (1.02 to 1.13 calls), against AMD's 1.27 for 4 and 2.85 for 8: on the Neural Engine speculative decoding and prompt chunks of 64 cost almost nothing in any format.
- lut4 reaches 13.4 TFLOPS at 64 tokens, the M3's 13.7; fp16 and int8, which read 4 and 2 times the bytes, stay bound by the weights to 64 and still gain at 128 (1.35 and 1.51 times one call). Past 64 the tables fall (2.2 to 3.0 times at 128): a prompt in lut4, lut6 or lut8 is read fastest in chunks of 64.
- Where the compute bounds it, the larger tables cost: at 128 tokens lut8 reaches 7.1 TFLOPS against int8's 10.5, for the same bytes (17.4 against 11.9 ms). For long prompts int8 beats lut8; for decoding the two are even.

### Apple M1 (MacBook Pro 13", 16 GB, macOS 27.2)

`bonsaiai-dev probe` with the defaults (bench and limit; report: [npuprobe-m1.md](npu_profiles/npuprobe-m1.md)). `bonsaiai-dev report` passes all four checks. 16 layers (0.97 B), against the M2 and the M3 above:

| format | weights | unit | M1 ms/token | M2 ms/token | M3 ms/token | M1 weights GB/s | M2 weights GB/s | M3 weights GB/s |
|---|---|---|---|---|---|---|---|---|
| lut4 | 0.45 GiB | NPU | 11.0 | 7.8 | 7.5 | 44 | 63 | 65 |
| int8 | 0.91 GiB | NPU | 15.7 | 15.6 | 41.6 | 62 | 62 | 23 |
| int4 | 0.45 GiB | GPU | 14.8 | 8.5 | 8.5 | 33 | 57 | 57 |
| fp16 | 1.81 GiB | GPU | 35.3 | 25.5 | 24.5 | 55 | 76 | 79 |
| lut4 | 0.45 GiB | GPU | 83.2 | 55.4 | 65.1 | 6 | 9 | 7 |
| every format | | CPU | 38 or so | 35 or so | 27 to 30 | | | |

- The size limit is the M2's and the M3's to the layer: fp16 8 layers (0.91 GiB) on the Neural Engine and 9 (1.02 GiB) not, int8 17 and 18, lut4 35 (0.99 GiB) and 36. Three fp16 models of 0.91 GiB loaded at once all stay on the Neural Engine at 15.4 ms a call, as alone. Parts under 1 GiB (`bonsaiai-dev convert`'s default `-target any`) fit every chip measured so far.
- int8 and fp16 run at full speed, as on the M2: about 62 GB/s (0.49 B fp16: 15.6 ms), about 90% of the M1's memory bandwidth (68 GB/s). Of the three chips only the M3 is slow with them.
- lut4 is the slow format on the M1: 44 GB/s at 16 layers and 39 to 45 GB/s at every size from 2 to 35 layers, against 62 for int8 on the same chip. The memory is not what holds it back; unpacking the 4 bit tables seems to cost on the M1's Neural Engine. lut4 is still the fastest format per token, since it reads half the bytes, but 1.4 times faster than int8 (11.0 against 15.7 ms), not 2.
- The GPU is slower than on the M2 and the M3 (int4 14.8 ms against 8.5; fp16 55 GB/s against 76 to 79). So on the M1 the Neural Engine with lut4 is the fastest unit, ahead of the GPU with int4 (11.0 against 14.8 ms); on the M2 and the M3 the two are about even.
- The CPU takes about 38 ms per token in every format, at about 5 cores busy.
- lut4 put on the CPU over the limit is slow, though less than on the M2 (36 layers 2.7 s a call, 40 layers 3.3 s, 48 layers 6.3 s).
- Compiling is quicker than on the M2 (lut4 32 layers: 16.2 s against 25 s).
- **The M1 has a limit on the total too, when the parts run in turn** (`bonsaiai-dev probe split -formats lut4 -parts 8 -ref none`, 2026-11-08; report: [npuprobe-m1-split-104.md](npu_profiles/npuprobe-m1-split-104.md)). Up to 2.83 GiB of weights, all the parts loaded and run one after another take their own times added up; from 2.95 GiB every token takes about 12 times as long, as if parts were loaded again on each call. The number of parts does not matter (2.95 GiB in 4 parts: 986 ms). The M4 runs 3.62 GiB in 8 parts at its full speed (below). `bonsaiai-dev probe limit`'s several models (3 × 0.99 GiB) did not show it, because each model is timed alone, many calls in a row:

| total | parts | Neural Engine holds | ms/token | parts alone added up | weights GB/s |
|---|---|---|---|---|---|
| 1.81 GiB | 8 × 0.23 | 1.83 GiB | 44.9 | 44.7 | 43 |
| 2.27 GiB | 8 × 0.28 | 2.29 GiB | 55.2 | 55.2 | 44 |
| 2.72 GiB | 8 × 0.34 | 2.74 GiB | 65.8 | 65.9 | 44 |
| 2.83 GiB | 8 × 0.35 | 2.86 GiB | 68.6 | 68.9 | 44 |
| **2.95 GiB** | 8 × 0.37 | 2.97 GiB | **828** | 71.9 | 4 |
| 2.95 GiB | 4 × 0.74 | 2.97 GiB | 986 | 70.9 | 3 |
| 3.17 GiB | 8 × 0.40 | 3.20 GiB | 814 | 77.0 | 4 |
| 3.62 GiB | 8 × 0.45 | 3.66 GiB | 1010 | 86.3 | 4 |

  So on the M1 a model's programs that run in turn should hold under about 2.8 GiB on the Neural Engine. The M2 has no such limit (above). Qwen3.5-4B in lut8 (3.96 GiB) and lut6 (2.95 GiB) are over it and generate at about 1 token/s; in lut4 (1.96 GiB) at 10.9 ([docs/design/llm-decoding.md](docs/design/llm-decoding.md)).

### Apple M4

A model over the limit run in parts (`bonsaiai-dev probe split -split-layers 128 -parts 2,4`, 128 layers of lut4, 3.62 GiB, 7.8 B weights):

| split | each part | ms/token | weights GB/s |
|---|---|---|---|
| 2 ways | 64 layers, 1.81 GiB | 58.3 | 67 |
| 4 ways | 32 layers, 0.91 GiB | 58.8 | 66 |
| 8 ways | 16 layers, 0.45 GiB | 59.4 | 66 |
| 16 ways | 8 layers, 0.23 GiB | 60.6 | 64 |

- Each further part adds about 0.15 ms per token (2.3 ms from 2 to 16 parts): the fixed cost of one Core ML call. In 4 parts under 1 GiB that is 0.6 ms, about 1% of a token. The smaller the parts, the larger the share, so split into the fewest parts under the limit (as `bonsaiai-dev convert` does).
- The split does not change the error; it was the same after each part in 2 and in 4 parts.
- So the M4 loses nothing running the same parts under 1 GiB as the M3 (`bonsaiai-dev convert`'s default `-target any`). Every part is loaded at once, but the peak memory of the process while loading is about the weights of one part (on the M3, 1.03 GiB in 2 parts and 0.55 GiB in 4; table above), so finer parts help a machine short of memory.
- The Neural Engine reads the weights at about the M3's speed (65 GB/s).

### KV cache (Apple M3, `bonsaiai-dev probe kv`)

11 lut4 layers, one token, all on the Neural Engine (report: [npuprobe-m3-kv.md](npu_profiles/npuprobe-m3-kv.md)):

| cache length | KV size | host buffers | MLState |
|---|---|---|---|
| 256 | 5.5 MiB | 7.1 ms (CPU use 0.16) | 6.6 ms (0.06) |
| 1024 | 22 MiB | 11.0 ms (0.18) | 10.0 ms (0.04) |
| 4096 | 88 MiB | 27.0 ms (0.16) | 26.8 ms (0.02) |
| 8192 | 176 MiB | 49.5 ms (0.15) | 49.3 ms (0.02) |

- MLState does not change the speed. Only the CPU use falls: with host buffers Core ML handles the inputs on the CPU.
- Both slow down with the cache's length. The cause is not copying the inputs but the Neural Engine's work over the whole fixed length cache.
- A value written to a state must be read back before it is used, or the Neural Engine compiler fails (`error code: -14`).
- The cost is the attention over the whole cache itself (report: [npuprobe-m3-kv-mha.md](npu_profiles/npuprobe-m3-kv-mha.md)). `host-mha`, which repeats nothing and holds a cache 4 times as large, takes 14.2 / 36.6 / 63.5 ms at 1024 / 4096 / 8192: only 1.3 times `host`. Neither the bandwidth nor the repeat is the limit; the Neural Engine cannot fill itself with matrix-vector products of one query row (16 heads × cache length per layer, about 8 GFLOP/s at 1K tokens).
- An int8 cache does not make it faster either (report: [npuprobe-m3-kv-int8.md](npu_profiles/npuprobe-m3-kv-int8.md)). `host-int8` stays on the Neural Engine with half the memory, but takes 11.9 / 29.5 / 50.8 ms at 1024 / 4096 / 8192 against 10.8 / 27.6 / 46.5 ms for `host` in the same run: a tenth slower, for the cast over the whole cache. Quantizing the cache saves memory, not time.
- The Neural Engine takes tensor dimensions up to 16384 (report: [npuprobe-m3-kv-limit.md](npu_profiles/npuprobe-m3-kv-limit.md)). With `-kv-layers 2`, a key axis of 16384 (cache 16383) stays on the Neural Engine at 16.6 ms; 16385 (cache 16384) puts 39 attention operations on the CPU, at 117 ms.
- A cache in chunks stays on the Neural Engine past that limit (`-kv-layers 6`, LFM3-1.7B's full attention layers; reports: [npuprobe-m3-kv-chunk.md](npu_profiles/npuprobe-m3-kv-chunk.md) and [npuprobe-m3-kv-chunk-long.md](npu_profiles/npuprobe-m3-kv-chunk-long.md)). `host-chunk` takes 13.2 / 43.4 / 84.0 / 201.6 ms at 4096 / 16320 / 32768 / 65536, all on the Neural Engine, and is no slower than the plain `host` (14.5 / 49.1 ms at 4096 / 16320). Each chunk is an input of its own, and the chunks meet in elementwise operations only. Holding the chunks on an axis of one tensor and reducing over it, or weighting each chunk by its share of the sum, put every operation on the CPU from two chunks on.
- Attention without weights called per chunk and merged on the host costs in proportion to the tokens used, not to the capacity (`bonsaiai-dev probe attn`; report: [npuprobe-m3-attn.md](npu_profiles/npuprobe-m3-attn.md)). Grouped, Q=8, chunks of 8192, 6 layers: 8.8 / 17.4 / 69.5 / 295.7 ms at 4096 / 16384 / 65536 / 262144 tokens used. Binding each chunk's own buffers does not change the time per call, and the peak resident memory is the KV bytes (3 GiB at 262144) plus about 0.1 GiB: the inputs are not copied. Core ML puts the forms with little work on the CPU (Q=1 up to chunks of 8192, Q=8 up to 2048); with Q=1 the CPU is twice as fast as the Neural Engine (0.41 against 0.80 ms per 1K).

### Wanted

M4 and later Apple chips (does the 1 GiB limit change with the chip or the RAM?), Intel Core Ultra on Linux and Windows (the Intel backend has not run on real hardware yet), AMD Ryzen AI for `bonsaiai-dev report` and `bonsaiai-dev probe` on other chips than Strix Point (Krackan Point, Strix Halo), and Core AI's path in the probe on Apple (the same shapes as Core AI programs; not written yet). Open an issue with the `.md` files.

The M1 again, for the formats compared as on AMD, the M2 and the M3 above: `bench` now runs lut6 and lut8 too (the weights of `bonsaiai-dev convert -bits 6` and `8`) and gives G weights/s, and `verify` runs every format the Neural Engine takes:

```bash
go run github.com/shibukawa/bonsaiai/cmd/bonsaiai-dev@latest probe bench
go run github.com/shibukawa/bonsaiai/cmd/bonsaiai-dev@latest probe verify -layers 8
```

## Limits of these measurements

- Random weights: the numbers say nothing about accuracy. For accuracy, see [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md).
- No KV cache: real decoding adds the attention over the context, small next to the weights at short contexts.
- Core ML's compute plan is an estimate of placement; CPU use is the cross-check.
- AMD's stream form is a stand-in for a step as one resident array: the same bytes, without what such an array adds (its phases' fixed 25 to 35 µs, a command boundary a layer, [docs/design/moe.md](docs/design/moe.md)).
- `bonsaiai-dev powerbench -decode` and the Intel paths of bonsaiai-dev probe have not been run on hardware other than the M3 yet.
