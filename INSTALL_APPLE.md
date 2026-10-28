# Setting up an Apple Neural Engine for hwnpu

How to use the Neural Engine of an Apple Silicon Mac from hwnpu, from checking the machine to running the report and preparing models.

> **Status.** The Apple backend is the one hwnpu has been developed and measured on (M3, macOS 27). The numbers in [MODEL.md](MODEL.md) come from it.

日本語: [INSTALL_APPLE.ja.md](INSTALL_APPLE.ja.md)

## Supported platforms

| Platform | Supported | Notes |
|---|---|---|
| macOS 14.4 or later on Apple Silicon | Yes | Core ML with `MLComputePlan`, which hwnpu uses to verify that a model runs on the Neural Engine. |
| macOS 13 to 14.3 on Apple Silicon | Partly | Models load and run, but the placement cannot be verified (`PlacementReport.Verified` is false) and strict mode refuses them. |
| Intel Macs | Not applicable | No Neural Engine. Core ML would run everything on the CPU. |
| iOS, iPadOS | **No** | hwnpu is a Go library for macOS; it does not build for iOS. |
| Linux, Windows | Not applicable | The Neural Engine is only reachable through Core ML on macOS. |

## What hwnpu needs

hwnpu calls Core ML through purego and the Objective-C runtime. Nothing has to be installed: Core ML is part of macOS.

| Component | Needed | Notes |
|---|---|---|
| Apple Silicon Mac | yes | Every Apple Silicon chip has a Neural Engine. |
| macOS 14.4 or later | yes | For placement verification; 13 or later to load models. |
| Go | yes | 1.27 or later (see `go.mod`). `CGO_ENABLED=0` works: no Xcode and no C toolchain are needed. |
| Xcode or Command Line Tools | no | |
| Python with coremltools | only for models `hwnpu-convert` does not know | `cmd/hwnpu-convert` converts supported LLM checkpoints (LFM3 so far) without Python (step 4). |
| `sudo` | only for power measurement | `cmd/powerbench` runs `powermetrics`, which needs root. |

Unlike the Intel and AMD backends, Core ML cannot be told "Neural Engine only": hwnpu allows the CPU and the Neural Engine and never the GPU, then checks the compute plan and reports every operation that would run on the CPU.

## 1. Check the machine

```bash
sysctl hw.model && sw_vers
```

Any `Mac`-prefixed model with an Apple chip (M1 or later) qualifies. The macOS version must be 14.4 or later for the full checks.

## 2. Get hwnpu

```bash
go get github.com/shibukawa/hwnpu
```

Or clone the repository to run the tools below. List the devices:

```bash
go run ./cmd/hwnpuinfo
```

The Neural Engine appears with `backend: coreml`.

## 3. Check the installation

The report runs the embedded test models, verifies their placement and times them:

```bash
go run ./cmd/report -o report.md
```

The fp16 models should show all of their operations on the Neural Engine. Two models are there to be rejected: the fp32 one and the small MLP at batch 1, which Core ML places on the CPU; the report checks that strict placement (`WithPlacement(hwnpu.PlacementStrict)`, the default) refuses them, which is the behavior for applications that must not fall back silently. The same MLP padded to batch 64 runs on the Neural Engine.

To measure the energy per call, which needs root for `powermetrics`:

```bash
go run ./cmd/powerbench -model 'path/to/*.mlpackage' -o power.md
```

It asks for the sudo password once and compares the Neural Engine with the CPU (and with the GPU when `-gpu` is given). The results on an M3 and how to read them are in [MODEL.md](MODEL.md).

To measure the Neural Engine with generated models, run the probe. It builds LLM decode shaped models in Go and measures 0.5 B and 1 B weights in fp16, int8, lut4 and int4 on the Neural Engine, the CPU and the GPU (`bench`). Then it grows the model in each format until it leaves the Neural Engine, bisects to the boundary, and loads several models just under it at once to tell a limit per model from a limit on the total (`limit`). It needs no Python, no model files and no root, takes about 40 minutes, needs free disk space of about twice the largest model (a quarter of the RAM by default, `-max-gb` to change), and writes `npuprobe-<machine>-<time>.md`. It runs without cloning:

```bash
go run github.com/shibukawa/hwnpu/cmd/npuprobe@llm-convert
```

For the energy per token, `go run ./cmd/powerbench -decode 8,16 -gpu` measures the same models with powermetrics.

## 4. Preparing models

Core ML takes `.mlpackage` (ML Program), `.mlmodel` and compiled `.mlmodelc`. The first load compiles an `.mlpackage` to `.mlmodelc`; `WithCacheDir` keeps that result so later loads skip it.

What the Neural Engine needs from a model:

- **fp16.** fp32 operations run on the CPU. Convert with `compute_precision=FLOAT16`.
- **Fixed shapes.** Dynamic shapes fall back to the CPU. Export one model per sequence length or batch size.
- **Enough work per call.** Small models are placed on the CPU even in fp16. `tools/coreml/ane_pad.py` pads the batch dimension until the model moves; read the energy results in [MODEL.md](MODEL.md) first, since padding only pays above roughly 200 MFLOP per call.
- **Weights under 1 GiB per model.** On an M3 the boundary lay between 0.99 and 1.02 GiB of stored weights in fp16, int8 and lut4 alike (`cmd/npuprobe limit`), and a model above it is placed entirely on the CPU. The limit is per model: three models of 0.91 GiB loaded at once all stayed on the Neural Engine at full speed, so a larger network can be split into several models. Only the 4 bit lookup-table format runs at full speed.

The intended path is `hwnpu-convert`, in Go, from the bf16 checkpoint; coremltools is only the fallback for architectures it does not know.

- **LLM checkpoints with hwnpu-convert** (Go only, no Python). `go run ./cmd/hwnpu-convert -config config.json -weights model.safetensors -o out.mlpackage -seq 32` builds the graph from `config.json` for a known architecture and palettizes the weights to 4 bit. It knows LFM3 so far; other architectures (Llama, Qwen) need their structure added in `internal/llm`. How to choose the source checkpoint and what the Neural Engine accepts are described in [MODEL.md](MODEL.md).
- **Anything else with coremltools** (Python). A generic converter that accepts arbitrary graphs is out of hwnpu's scope, so models that are not a supported LLM architecture (encoders such as MiniLM, vision models) are converted from PyTorch with `ct.convert(..., convert_to="mlprogram", compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS14)` and fixed input shapes. `tools/coreml/convert_minilm.py` and `tools/coreml/gen_sweep.py` are worked examples, written for the benchmarks.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `PlacementReport` lists operations on the CPU | fp32 operations, dynamic shapes, `gather` (embedding lookups always run on the CPU) or a model too small for the Neural Engine. Convert to fp16 with fixed shapes; keep embedding lookups on the host. |
| `Verified` is false | macOS older than 14.4, or a model in the old `neuralnetwork` format rather than an ML Program. Update macOS and reconvert as an ML Program. |
| Loading crashes instead of returning an error | A hand-written ML Program with dots in value names, or a corrupt package. coremltools and `hwnpu-convert` write valid names. |
| The whole model runs on the CPU although it is fp16 | Weights too large (1 GiB or more per model; split it into several models), or weight compression in a layout the Neural Engine does not accept (lookup tables must cover whole rows, 16 per table; blockwise int4 is not accepted). |
| Results differ from PyTorch by several percent | The Neural Engine's matrix products have an absolute error floor around 2.5e-4; layers whose outputs are tiny lose precision. `hwnpu-convert -post-scale` works around it for LFM3; see [MODEL.md](MODEL.md). |
| Need a CPU or GPU baseline | Set `HWNPU_COREML_UNITS=cpu` or `gpu` when loading (measurement only, not part of the API). |
| `powerbench` stops at `Password:` | `powermetrics` needs root; type the sudo password once, the tool keeps the credentials alive. |
