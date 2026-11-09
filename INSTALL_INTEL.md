# Setting up an Intel NPU for hwnpu

This guide prepares an Intel Core Ultra machine (Meteor Lake, Arrow Lake, Lunar Lake, Panther Lake, Wildcat Lake) so that hwnpu can use its NPU, and shows how to check the result.

> **Status.** The Intel backend has been tested against a simulated Level Zero driver only, and hwnpu-chat and hwnpu-chat agent with the same programs on OpenVINO's CPU. Running `hwnpu-dev report` on real hardware (step 4) and the chat (step 6), and sending the results back, is the most useful help right now.

Japanese: [INSTALL_INTEL.ja.md](INSTALL_INTEL.ja.md)

## Supported platforms

| Platform | Supported | Notes |
|---|---|---|
| Linux (x86-64) | Yes | Ubuntu 24.04 / 26.04 with Intel's NPU driver packages (step 1). |
| Windows 11 (x86-64) | Yes | With Intel's NPU driver (step 2). |
| WSL2 | **No** | WSL2 passes only the GPU to Linux (`/dev/dxg`). The NPU is not visible inside WSL2, so use Windows directly. |
| macOS | Not applicable | Macs have no Intel NPU. On Apple Silicon hwnpu uses Core ML instead. |

## What hwnpu needs

hwnpu talks to the NPU through **Level Zero** and the NPU graph extension, loaded at run time with purego. No cgo, no OpenVINO runtime.

| Component | Linux | Windows |
|---|---|---|
| Kernel driver | `intel_vpu` (in the mainline kernel) | Intel NPU driver |
| Firmware | `intel-fw-npu` package | included in the driver |
| User mode driver | `intel-level-zero-npu` (`libze_intel_npu.so.1`) | included in the driver |
| Driver compiler (to compile OpenVINO IR) | `intel-driver-compiler-npu` | included in the driver |
| Level Zero loader | `libze1` (`libze_loader.so.1`) | `ze_loader.dll` |
| Device access | read/write on `/dev/accel/accel0` (`render` group) | none |
| Go | 1.27 or later (see `go.mod`), `CGO_ENABLED=0` works | same |

The driver, driver compiler and firmware must come from the **same release**. A mismatch shows up as compile errors such as `MAPPED_INFERENCE_VERSION is NOT compatible with the ELF`.

## 1. Linux (Ubuntu 24.04 / 26.04)

These steps follow the instructions of [intel/linux-npu-driver](https://github.com/intel/linux-npu-driver/releases). The commands below are for **v1.38.0** (checked on 2026-10-28); take the file names from the latest release when you install.

Remove old packages:

```bash
sudo dpkg --purge --force-remove-reinstreq intel-driver-compiler-npu intel-fw-npu intel-level-zero-npu intel-level-zero-npu-dbgsym
```

Download and extract the release (Ubuntu 24.04 shown; use the `ubuntu2604` archive on 26.04):

```bash
wget https://github.com/intel/linux-npu-driver/releases/download/v1.38.0/linux-npu-driver-v1.38.0.20260910-34487311128-ubuntu2404.tar.gz
```

```bash
tar -xf linux-npu-driver-v1.38.0.20260910-34487311128-ubuntu2404.tar.gz
```

Install the dependency of the driver compiler, then the packages:

```bash
sudo apt update && sudo apt install libtbb12
```

```bash
sudo dpkg -i *.deb
```

Install the Level Zero loader. The release page links the `libze1` package it was validated with (v1.32.0 for v1.38.0):

```bash
wget https://snapshot.ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu/20260830T100000Z/pool/main/l/level-zero-loader/libze1_1.32.0-1~24.04~ppa1_amd64.deb
```

```bash
sudo dpkg -i libze1_*.deb
```

If `libze1` conflicts with an older Level Zero installation, remove it first:

```bash
sudo dpkg --purge --force-remove-reinstreq level-zero level-zero-devel
```

Give your user access to the device, then log out and back in (or reboot):

```bash
sudo gpasswd -a ${USER} render
```

Check that the device node exists and is accessible:

```bash
ls -l /dev/accel/accel0
```

It should be owned by group `render` with mode `crw-rw----`. If the group is `root`, fix it until the next reboot with:

```bash
sudo chown root:render /dev/accel/accel0 && sudo chmod g+rw /dev/accel/accel0
```

## 2. Windows 11

1. Install the latest **Intel NPU driver** for Windows from Intel's download center, or through Windows Update. Task Manager shows an **NPU** entry under Performance when the driver works.
2. Check that the Level Zero loader is present:

   ```powershell
   where.exe ze_loader.dll
   ```

   hwnpu loads `ze_loader.dll` through the normal DLL search path. Where the NPU driver installs it is **not verified yet**. If `where.exe` finds nothing, look under `C:\Windows\System32` and the driver's folder, or take `ze_loader.dll` from a [Level Zero release](https://github.com/oneapi-src/level-zero/releases), and point hwnpu at it:

   ```powershell
   $env:HWNPU_ZE_LOADER = "C:\path\to\ze_loader.dll"
   ```

WSL2 cannot be used: it exposes the GPU (`/dev/dxg`) but not the NPU.

## 3. Get hwnpu

```bash
git clone https://github.com/shibukawa/hwnpu.git
```

```bash
cd hwnpu
```

On a machine without Go you can cross-compile the report on another machine and copy the single executable; the test models are embedded:

```bash
GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build -o hwnpu-dev.exe ./cmd/hwnpu-dev
```

## 4. Check the installation

```bash
go run ./cmd/hwnpu-dev info
```

An Intel NPU appears with `backend: levelzero`. The `driver` line lists the Level Zero version, the driver version, the graph extension, the compiler version and whether the NPU command queue extension (turbo, workload type) is available.

Then run the report and keep the Markdown file:

```bash
go run ./cmd/hwnpu-dev report -o report.md
```

For each Intel NPU it compiles two OpenVINO IR models in strict mode, compares the results with fp32 references (the NPU computes in fp16, so small differences are expected), and measures latency.

To compare power policies, run it twice. On Linux the package energy (RAPL) is readable by root only, so run these with `sudo` to get the energy columns:

```bash
go run ./cmd/hwnpu-dev report -power low -o low.md
```

```bash
go run ./cmd/hwnpu-dev report -power performance -o performance.md
```

To measure the NPU with generated models, run the probe. It builds LLM decode shaped OpenVINO IR models in Go and measures 0.5 B and 1 B weights in fp16, int8 and int4 (`bench`; with energy per token when the RAPL counters are readable, as root on Linux). Then it grows the model in each format until the driver no longer builds it, bisects to the boundary, and loads several models just under it at once (`limit`). It needs no Python and no model files, takes from tens of minutes to an hour, and writes `npuprobe-<machine>-<time>.md`. It runs without cloning:

```bash
go run github.com/shibukawa/hwnpu/cmd/hwnpu-dev@latest probe
```

Run `hwnpu-dev report` first: if the small models there fail, the probe will fail the same way. How to read the results: [PROBE.md](PROBE.md).

## 5. Preparing models

hwnpu takes models in two forms:

- **OpenVINO IR** (`model.xml` and `model.bin`): compiled by the driver compiler when loaded. Convert from ONNX, PyTorch or TensorFlow with OpenVINO's `ovc` tool or `openvino.convert_model` (this needs OpenVINO only on the machine that converts). Use static shapes: the NPU does not run dynamic shapes.
- **Compiled blob**: the output of a previous compilation for the same driver and compiler. With `hwnpu.WithCacheDir(dir)` hwnpu stores and reuses blobs itself.

```go
dev, _ := hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorIntel), hwnpu.WithCacheDir(cacheDir))
gr, _ := dev.Graph()
g, err := gr.Compile(ctx, hwnpu.GraphSource{Path: "model.xml"})
```

A model with an operation the NPU compiler does not support fails with `hwnpu.ErrNotOnNPU`; the error lists the operations and includes the compiler log.

## 6. hwnpu-chat and hwnpu-chat agent

The chat and the agent run on Intel's NPU with the same commands as on a Mac (not yet tried on the hardware; see "What to check" below).

```bash
go run github.com/shibukawa/hwnpu/cmd/hwnpu-chat@latest              # talk in the terminal
go run github.com/shibukawa/hwnpu/cmd/hwnpu-chat@latest agent "summarize the README"
```

The first run:

1. Downloads the checkpoint from Hugging Face (3.5 GB for LFM3-1.7B).
2. Converts it to OpenVINO IR in `intel/` of the model's directory. Since 2026-11-04 it is split at the full attention layers (`-paged`) as for the Neural Engine, with a context of 262144 (`lfm3-int8-grouped-p8192-c262144-q8.parts.json` and `-q64`, 7 parts each and a graph without weights for the attention): a step costs in proportion to the conversation, and the KV cache's memory grows with it (12 KiB a token). Checked on OpenVINO's CPU (`-backend openvino`), not yet on the NPU. The earlier form, whole (four graphs, one that generates and one that reads prompts for each of the caches 4096 and 16320, `lfm3-int8-c4096-q8.xml` and so on), is `-paged -1`.
3. Has the driver compiler compile them for the NPU, which **can take minutes each** for a large model. The results stay in the compile cache (`-cache`, by default `hwnpu` in the user cache directory), so later runs start faster.

| Flag | Meaning |
|---|---|
| `-backend` | The NPU: `levelzero` (also `intel`), `coreml`, `openvino`. Default: `HWNPU_BACKEND`, else the NPU found |
| `-format` | The chat model's weights. On Intel's NPU `int8` (LFM3's default), `int4-g128`, `int4`, `fp16` |
| `-paged -1` | Not split at the full attention layers: one graph instead of one per part, at a cost that follows the context's capacity (then `-tier` applies) |
| `-tier none` | (with `-paged -1`) No shorter cache to start on: two graphs instead of four, half the first compilation and the memory |
| `-context 4096` | The conversation's length (the KV cache). Shorter compiles faster and runs each token faster |

Starting small makes failures easier to tell apart:

```bash
hwnpu-chat --paged -1 --tier none --context 4096
```

### What to check

1. Whether `hwnpu-chat --paged -1 --tier none --context 4096` gets to `ready:`. If it stops, please send the message (it includes the compiler's log).
2. Whether it answers questions correctly, and the tokens per second shown after each answer.
3. Whether the defaults (`hwnpu-chat` alone) load, how long the compilation takes, and whether memory suffices.
4. Whether `hwnpu-chat agent` calls its tools.
5. The accuracy and speed of int8, int4-g128 and int4 with `hwnpu-dev formats` (next section).

### Comparing weight formats (hwnpu-dev formats)

`hwnpu-dev formats` converts the chat model in each format, loads it on the NPU, reads a fixed conversation (Japanese, and English with code) through it 8 tokens per call as hwnpu-chat generates, and prints:

| Column | Meaning |
|---|---|
| ms per call | Time of one call (median): how fast it generates |
| next token as reference | How often the most likely next token is the reference format's (fp16 by default) |
| logits rel. err. | Relative error of the logits against the reference |
| perplexity | Of the answers' tokens; lower is better, and comparable without a reference |

```bash
go run ./cmd/hwnpu-dev formats                                    # int8, int4-g128 and int4 against fp16
go run ./cmd/hwnpu-dev formats -formats int4-g128 -ref int8       # when fp16 does not fit the NPU
go run ./cmd/hwnpu-dev formats -json formats.json                 # also write the results to a file
```

The fp16 reference has 3.2 GiB of weights, which the NPU compiler may take long over or reject. Use `-ref int8` then (nearly the same as fp16), or, with OpenVINO installed, compute only the reference on the CPU:

```bash
HWNPU_OV_PYTHON=~/ov/bin/python go run ./cmd/hwnpu-dev formats -ref-backend openvino
```

The programs stay in `intel/formats/` of the model's directory, so later runs skip the conversion and the compilation. On OpenVINO's CPU on Linux x86 (`-backend openvino`) the results were as below; the numbers from the NPU will decide Intel's default format.

| Format | Weights (GiB) | Next token as fp16's | Perplexity |
|---|---|---|---|
| fp16 | 3.22 | (reference) | 3.81 |
| int8 | 1.61 | 98% | 3.84 |
| int4-g128 | 0.83 | 80% | 4.28 |
| int4 | 0.81 | 76% | 4.89 |

To use a format in hwnpu-chat and hwnpu-chat agent, pass it as `--format int4-g128`.

### Telling problems apart with OpenVINO

When hwnpu's Level Zero path fails, running the same IR with OpenVINO tells whether the problem is hwnpu's or the model's (the IR's). With a Python that has OpenVINO:

```bash
python3 -m venv ~/ov && ~/ov/bin/pip install openvino numpy
HWNPU_OV_PYTHON=~/ov/bin/python HWNPU_OV_DEVICE=NPU hwnpu-chat --backend openvino --tier none --context 4096
```

`HWNPU_OV_DEVICE=NPU` runs on OpenVINO's NPU plugin, `CPU` (the default) on the CPU. On Windows, set `HWNPU_OV_PYTHON` to the path of `python.exe`.

## 7. All the checks at once (Windows)

`tools/intel/hwnpu-intel-check.ps1` (PowerShell 5.1) runs steps 4 and 6 and more in one go, each with `go run` in the clone and a time limit, and packs the logs into a zip under `%USERPROFILE%\hwnpu-intel-logs`: the machine and the driver, each command's output, its time and memory. A test that fails does not stop the others, and running it again with the same `-Out` skips the tests that passed.

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\intel\hwnpu-intel-check.ps1 -Level quick
```

`-Level quick` (about 30 minutes) checks the device, `cmd/report` and a small LFM3-1.7B chat. `standard` (the default, a few hours and about 25 GB of downloads) adds LFM3-1.7B with its defaults, hwnpu-agent, hwnpu-formats, PLaMo 3 Translate, PLaMo-Embedding, Gemma 4 E2B and Anima at 512 x 512. `full` adds images and audio in the chat, Qwen3.5, Anima at 1024 x 1024 and npuprobe. `-List` shows the tests, `-Only` and `-Skip` choose them.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `runtime library not found ... libze_loader.so.1` | Install `libze1` (step 1), or set `HWNPU_ZE_LOADER` to the loader's path. |
| `found /dev/accel/accel0 but cannot use it` | The device exists but Level Zero failed; the message carries the reason. |
| `permission denied` on `/dev/accel/accel0` | Add the user to `render` and log in again (step 1). |
| No device, `/dev/accel/accel0` missing | Check `sudo dmesg \| grep -i vpu` for firmware load errors; reinstall `intel-fw-npu` from the same release as the driver. |
| `MAPPED_INFERENCE_VERSION is NOT compatible` | Driver, compiler and firmware are from different releases. Reinstall all packages of one release. |
| `the driver has no compiler for OpenVINO IR` | `intel-driver-compiler-npu` is missing. Install it, or load precompiled blobs. |
| `ZE_extension_graph version ... is not supported` | The driver is too old for hwnpu (graph extension 1.5 or later is needed). Update the driver. |
| hwnpu-chat stops with `chain: part 0 has no output y` | The argument names the driver reports do not match the IR's. Please send the message and the output of `hwnpu-dev info`. |
| hwnpu-chat runs out of memory while loading | `--tier none` halves the graphs; `--format int4-g128` halves the weights. |
| Need more detail from Level Zero | Run with `ZE_ENABLE_LOADER_DEBUG_TRACE=1`. |
