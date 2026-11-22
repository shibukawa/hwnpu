# Architecture

[日本語](architecture.ja.md)

The vendor stacks hwnpu sits on, how the code is split, the rules that keep it free of cgo, how it is
tested, and how models are converted. The public API is in [api.md](api.md); each backend has its
own file ([Neural Engine](apple-neural-engine.md), [Core AI](apple-coreai.md), [Core ML](apple-coreml.md), [Intel](intel.md), [AMD](amd.md)).

## Vendor software stacks

| Vendor | OS | Path hwnpu takes | Backend name |
|---|---|---|---|
| Intel NPU (Core Ultra, Meteor Lake and later) | Linux, Windows | Level Zero loader (`libze_loader.so.1` / `ze_loader.dll`) + the NPU graph extension, through purego | `levelzero` |
| Intel (checks without the NPU) | any | OpenVINO IR run by OpenVINO itself (CPU plugin, or `HWNPU_DEV=ov.device=NPU`) | `openvino` (only with `HWNPU_BACKEND=openvino`) |
| AMD XDNA (Ryzen AI, Phoenix and later) | Linux | `amdxdna` kernel driver, ioctl and mmap through `golang.org/x/sys/unix` (not even purego) | `xdna` |
| AMD XDNA | Windows (Linux for tests) | XRT (`xrt_coreutil.dll`) through `native/xrtshim`, a small C library loaded at run time | `xrt` (on Linux only with `HWNPU_BACKEND=xrt`) |
| Apple Neural Engine | macOS | Core ML through `purego/objc` | `coreml` (default) |
| Apple Neural Engine | macOS 27 | Core AI through a Swift shim (`libcaishim.dylib`, embedded, loaded with purego) | `coreai` (`WithBackend("coreai")`, `HWNPU_BACKEND=coreai`) |

### Intel

```
app (Go, purego)
  └─ libze_loader.so.1 / ze_loader.dll            Level Zero loader (MIT)
       └─ libze_intel_npu.so.1 / (Windows DLL)    UMD, intel/linux-npu-driver (MIT)
            ├─ libnpu_driver_compiler.so          compiler in the driver: OpenVINO IR → NPU blob
            └─ /dev/accel/accelN                  KMD intel_vpu (drivers/accel/ivpu, GPL)
```

- The NPU extensions (`intel/level-zero-npu-extensions`) are all C ABI, reached through the
  versioned DDI table of `ze_graph_ext.h`. The extension name is versioned on old drivers
  (`ZE_extension_graph_1_2`) and plain on new ones (`ZE_extension_graph`); both are accepted.
- The firmware job format is internal to the UMD and not in the `ivpu_accel.h` UAPI, so direct
  ioctl is not practical. Level Zero is the only good path.
- `zeGraphCreate` takes a compiled blob (`ZE_GRAPH_FORMAT_NATIVE`) or OpenVINO IR
  (`ZE_GRAPH_FORMAT_NGRAPH_LITE`, compiled in the driver). The IR I/O description is a port of
  OpenVINO's `serializeIOInfo`; `--config` options come from `GraphSource.BuildFlags`.
- Driver, compiler and firmware versions must match ("MAPPED_INFERENCE_VERSION is NOT compatible
  with the ELF" otherwise). On Linux `/dev/accel/accel0` needs the `render` group.

### AMD

```
app (Go)
  └─ golang.org/x/sys/unix ioctl / mmap
       └─ /dev/accel/accelN    KMD amdxdna (drivers/accel/amdxdna, upstream since Linux 6.14)

official path, for reference: C++ app ─ libxrt_coreutil.so.2 + xrt_plugin.*-amdxdna (shim, Apache-2.0)
```

- The UAPI (`amdxdna_accel.h`) is small and stable: `CREATE_HWCTX / DESTROY_HWCTX / CONFIG_HWCTX /
  CREATE_BO / GET_BO_INFO / SYNC_BO / EXEC_CMD / WAIT_CMD / GET_INFO / SET_STATE / GET_ARRAY`.
- Work goes to a hardware context (a partition of columns) as `ert_start_kernel_cmd` packets in a
  command BO. Programs are xclbin (AXLF; the PDI overlay in `AIE_PARTITION`) plus control code
  (IRON's `insts.bin`), or a full ELF (AIE2P, Strix Point and later).
- XRT's C API is not usable from purego: hardware contexts and modules are C++ only and the remaining
  C calls are variadic. So Linux talks to the driver directly, porting XRT's shim (Apache-2.0); Windows,
  whose KMD interface is private, goes through XRT's C++ library via xrtshim. Details in [amd.md](amd.md).

### Apple

- The ANE's own framework (`AppleNeuralEngine.framework`, `_ANEClient`) is private. Core ML (and Core AI
  on macOS 27) are the public paths.
- `MLComputeUnitsCPUAndNeuralEngine` (macOS 13+) excludes the GPU; "Neural Engine only" cannot be asked for.
- `MLComputePlan` (macOS 14.4+) gives each op's preferred and supported devices before running. Strict
  mode uses it. It is an estimate, not a guarantee of where ops run.
- Core AI reports placement from its own compile cache (the MPSGraph delegate's manifest,
  `mps.fullyPlacedOnANE`). See [apple-coreai.md](apple-coreai.md).

### Qualcomm (not started)

Snapdragon X's Hexagon NPU has a C API in the QNN SDK (`QnnHtp.dll` / `libQnnHtp.so`,
`QnnInterface_getProviders`) that suits purego and would fit `GraphRunner`. Out of scope for now.

## Design policy

### What is shared and what is not

| Layer | Shared | Why |
|---|---|---|
| Device listing and opening | yes | hiding sysfs / loader / DLL differences is cheap |
| Device-visible memory (`Buffer`) | yes | "zero-copy `[]byte` view + explicit Sync" fits every backend |
| Completion (`Fence`) | yes | a `Wait` that `context.Context` can cancel |
| Submitting work | no, by capability | graph inference (`GraphRunner`: Intel, Apple) and kernel runs (`KernelRunner`: AMD) differ in arguments and in how they fail |
| Model format | no | OpenVINO IR, xclbin / ELF and mlpackage / aimodel do not convert into each other |

Intel is a graph machine (a whole network compiled by the driver, no public user-kernel API). AMD is a
kernel machine (IRON / mlir-aie / Peano kernels poured into the AIE tile array, no public driver
compiler). The API does not hide this.

Layering: drivers → hwnpu core (model and kernel execution) → helpers per use (chat, embeddings,
images, speech) in `internal/`.

### Why each backend

- **Intel = Level Zero (purego).** The only public path; C ABI with every struct by pointer.
- **AMD Linux = direct ioctl.** No dependencies, pure Go build. Cost: porting the shim and following
  changes in the command format.
- **AMD Windows = XRT through xrtshim.** The KMD interface is private; `xrt_coreutil.dll` is the only
  public entry and is C++.
- **Apple = Core ML (purego/objc), Core AI on macOS 27.** GPU excluded and placement checked first.
  The development machines are Macs, so the API is shaped there.

## Package layout

Module `github.com/shibukawa/hwnpu`.

| Path | What |
|---|---|
| `hwnpu.go`, `device.go`, `graph.go`, `kernel.go`, `options.go`, `float16.go` | Public API ([api.md](api.md)) |
| `backends.go`, `backends_darwin.go` | Register the backends compiled in (Level Zero, xrt, xdna, OpenVINO everywhere; Core ML, Core AI on darwin) |
| `models.go` | `ModelsYAML`: the embedded `models.yaml` |
| `modelstore/`, `npuinfo/` | Public packages ([api.md](api.md)) |
| `internal/types` | Value types shared by the public API and the backends |
| `internal/backend` | Backend interface and registry; `levelzero`, `openvino`, `amd` (amdxdna), `amdxrt` (XRT), `coreml`, `coreai` |
| `internal/dl`, `internal/ze` | purego / LoadLibrary wrapper; Level Zero and NPU-extension bindings |
| `internal/xdna`, `internal/xclbin`, `internal/aieelf`, `internal/ert`, `internal/accelfs` | amdxdna UAPI (`xdnasim`: fake driver), AXLF reader, full-ELF reader, ERT packets, `/dev/accel` listing |
| `internal/mlpackage`, `internal/milspec`, `internal/mlspec` | `.mlpackage` writer (MIL protobuf, MILBlob `weight.bin`); protos generated from coremltools |
| `internal/coreai/ailower`, `internal/coreai/mlirbc` | MIL → Core AI program; MLIR bytecode |
| `internal/ovlower`, `internal/ovir`, `internal/ovcpu` | MIL → OpenVINO IR; IR reader; IR on OpenVINO's CPU plugin |
| `internal/milcpu`, `internal/onnxrt`, `internal/sgemm`, `internal/f32`, `internal/mxfp4`, `internal/vecops` | CPU executors and inner loops |
| `internal/safetensors`, `internal/gguf`, `internal/torchzip`, `internal/onnx`, `internal/tflite` | Checkpoint readers |
| `internal/palette` | LUT weight fitting |
| `internal/llm` (+ `blocks`, `archs`, one package per architecture, `decode`, `chain`, `runner`, `paged`, `experts`, `iron`, `check`) | Converters and runners for language and other models |
| `internal/chat`, `internal/imagegen/*`, `internal/speech`, `internal/transcribe`, `internal/vision`, `internal/audio`, … | Per-use engines behind the commands |
| `internal/cache`, `internal/store`, `internal/power`, `internal/mem`, `internal/sysload` | Compile cache and model store, energy counters, page-aligned memory and NPU memory, load |
| `cmd/hwnpu`, `cmd/hwnpu-chat`, `cmd/hwnpu-image`, `cmd/hwnpu-audio`, `cmd/hwnpu-dev` | Commands ([cli.md](cli.md)) |
| `native/xrtshim`, `internal/backend/coreai/shim` | Shims loaded at run time (C for XRT, Swift for Core AI) |
| `tools/` | Python references (`tools/ref`), IRON exporters (`tools/amd`), Core ML / Core AI / OpenVINO helpers |

Dependencies of the root package: `github.com/ebitengine/purego` and `golang.org/x/sys` on Linux and
Windows; on darwin also `google.golang.org/protobuf`, `klauspost/compress` (zstd) and `golang.org/x/text`.
`npuinfo` uses only the standard library. The commands add more (kong, go-yaml, oto, …).

## purego / ABI rules

```go
package dl
func Open(candidates ...string) (*Library, error)      // tries each; ErrLibrary listing what was tried
func (l *Library) Bind(fptr any, symbol string) error  // RegisterLibFunc's panic as an error
func (l *Library) Sym(symbol string) (uintptr, error)
func BindAddr(fptr any, addr uintptr)                  // DDI table function pointers
```

- C never holds Go memory. Device-visible memory is allocated by the driver (`zeMemAllocHost`, BO mmap)
  or outside the Go heap (`internal/mem`).
- No structs by value. Go structs reproduce the C layout by hand, padding written out.
- No callbacks: Level Zero events are polled, amdxdna waits in an ioctl (or on the context's syncobj).
- Strings are passed as NUL-terminated `[]byte` kept alive with `runtime.KeepAlive` until the call
  returns; purego's automatic copy is not relied on.
- No float arguments. Typed `RegisterFunc` only, never `purego.SyscallN`.
- 64-bit only (`amd64 || arm64`).
- Linux: `purego.Dlopen(name, RTLD_NOW|RTLD_LOCAL)`. Windows: `windows.LoadLibraryEx(…,
  LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)` + `GetProcAddress`.
- With `CGO_ENABLED=0` on Linux, purego's fakecgo runs `dlopen`; fully static links do not work.
- Without purego (`npuinfo` on darwin): import only `dlopen`/`dlsym`/`dlerror` from libSystem with
  `//go:cgo_import_dynamic` and assembly trampolines, as `golang.org/x/sys/unix` does, and look the rest
  up at run time, so a missing private library is an error, not a binary that does not start.

## Testing and diagnostics

| Kind | What | Hardware |
|---|---|---|
| ABI golden | `unsafe.Sizeof` / `Offsetof` against values generated once from the C headers (`testdata/abi/levelzero_abi.c`, `testdata/abi/xdna_abi.c`: `ze_*`, `amdxdna_*`, `axlf`, `ert_*`) | no |
| ioctl numbers | `iowr(…)` against known constants (e.g. `DRM_IOCTL_AMDXDNA_CREATE_BO = 0xC0206443`) | no |
| xclbin / ELF readers | mlir-aie sample files as fixtures | no |
| Fake drivers | `testdata/zemock` (fake Level Zero, built when `cc` exists), `internal/xdna/xdnasim` (fake amdxdna in Go) | no |
| Fixtures | small models in `internal/fixtures` (regenerated with `gen_models.py`) | no |
| Converter | synthetic weights (`llm.NewSynthetic`), digests of program and `weight.bin`, fp64 forward written from `modeling_*.py`, `milcpu` against PyTorch | no |
| On device | Core ML / Core AI tests on a Mac; `hwnpu-dev report` (passthrough design, sample models) on each NPU; `HWNPU_TEST_PREBUILT` for prebuilt programs | yes |
| Power | the same model under `PowerLow` and `PowerPerformance`, `hwnpu-dev powerbench` (mJ/token) | yes |

All tests run with `CGO_ENABLED=0 go test ./...`.

Diagnostics: `hwnpu-dev info [-model M]` (devices, versions, library paths, placement of a model),
`hwnpu status` and `hwnpu issue` (machine, NPU state via `npuinfo`, processes, cache; no NPU wake
unless `--wake`), `go run ./npuinfo/cmd/npuinfo` (raw JSON). See [cli.md](cli.md).

## Conversion pipeline in Go

- Models are converted from Hugging Face's official checkpoint (bf16; `config.json` + safetensors or
  GGUF) straight to each NPU's format. No ONNX in between. Each target gets its own build because each
  quantizes differently. Published quantized checkpoints are used only when they can be rearranged
  without loss; nothing is re-quantized from a quantized source ([quantization](quantization.md)).
- Conversion happens once: prebuilt files are distributed, or a command converts on first use and
  keeps the result in the cache ([distribution](distribution.md)). Users only download and let the
  runtime compile (Core ML's mlmodelc, Intel's driver, Core AI's specialization).
- Writers: `internal/mlpackage` (MIL for Core ML), `internal/coreai/ailower` (the same MIL lowered
  to Core AI), `internal/ovlower` (the same MIL as OpenVINO IR for Intel). AMD programs are built
  with IRON (`tools/amd`, `internal/llm/iron`; [amd-llm.md](amd-llm.md)).
- Readers: `internal/safetensors`, `internal/gguf` (dequantizes Q4_0, Q4_1, Q8_0, Q4_K, Q5_K, Q6_K),
  `internal/torchzip`, `internal/onnx`, `internal/tflite`.

### Commands

| Command | Does |
|---|---|
| `hwnpu-dev convert` | Builds a model: architecture detected from `config.json` (`-arch` to force), `-backend coreml` (`.mlpackage`) or `intel` (OpenVINO IR + `out.parts.json`), `-target` (chip whose size limit decides the split; `none` never splits), `-layers N` (first N layers only), `-archs` (list the whitelist). Architecture parameters become flags (LFM3: `-post-scale`, `-conv`, `-attn`, `-hidden-out`) |
| `hwnpu-dev check` | Compares against the reference: embedding, each layer alone (fed the reference input), final norm and head, whole model. Exit 1 at the first layer over `-tol` (relative RMS, default 0.01). `-package` checks an existing `.mlpackage`'s logits |
| `tools/ref/ref_logits.py` | Writes PyTorch fp32 hidden states and logits (hooks on the decoder layers, since `output_hidden_states` often ends after the final norm; `hidden[0]` from a hook on `get_input_embeddings()`), `REF.json`, `--encoder` for encoders |

### Checking without an NPU

- `internal/milcpu` runs MIL programs in fp32 on the CPU (linear, matmul, elementwise, reductions,
  softmax, reshape/transpose/slice/concat/pad, conv1d, `gelu`, `layer_norm`, LUT and blockwise
  constexprs …). Purpose: does the graph compute what PyTorch does. `RoundF16` rounds each op's result
  to fp16 to imitate NE storage (not NE arithmetic). Speed is secondary (matmul is parallel).
- The same function written as different graphs (taps vs conv, with and without post-scale, per-layer vs
  whole) must agree. Tests also confirm a checker catches a deliberately broken layer.
- Numeric behavior of the NE itself is checked on a Mac by diffing the NE against Core ML's CPU per op
  ([apple-neural-engine.md](apple-neural-engine.md)).

## Model whitelist

Like llama.cpp, supported architectures are a whitelist. Each is a hand port of the forward pass of
Hugging Face's `modeling_*.py`: one `nn.Module` = one Go method. Reasons: NPU-specific rewrites
(post-scale, conv split into taps, gather avoided) are easy to put in, and nothing generic has to be
trusted. Steps and pass conditions are in `.claude/skills/port-architecture` and the
[new-model checklist](new-model-checklist.md).

- `internal/llm`: architecture-neutral parts. `Weights` (safetensors / GGUF by HF tensor names),
  `Options` (common settings, `NoHead` to output the hidden state before the final norm, `Start` =
  layer count builds only final norm + head), `Arch` (name, `model_type` / `architectures`, GGUF name
  map, `Params`, Intel format …), `Register` / `Detect` / `LoadConfig`.
- `Model.Tensors()` lists the tensors a build reads; `llm.NewSynthetic` makes deterministic random
  weights from it; `llm.Inventory` reports missing, mismatched and unused tensors against a checkpoint
  (unused tensors hint at a part not ported).
- `Model.Output()` (width, logits or not), `Model.Embedding()`, inputs listed by `Build`: the hooks
  encoders need. A program with `Start == 0` includes the embedding stage.
- `internal/llm/blocks`: shared layers (linear in a chosen weight format, `LinearBias`,
  `LinearPadded`, RMSNorm, `LayerNorm`, SwiGLU, RoPE, masks, `SDPA` / `Attention` / `SDPAGrouped`,
  `SiLU`, `Activation`). An architecture embeds `blocks.Graph` and writes only its own layers.
- `internal/llm/archs`: its import list is the whitelist.

| Package | Hugging Face architecture |
|---|---|
| `lfm3` | LFM3 (`lfm3`, `lfm3_vl`, `lfm3_omni`, `lfm3_audio` text models) |
| `lfm3vl`, `lfm3audio`, `lfm3gen` | LFM3-VL vision encoder, LFM3-Audio encoder, LFM3-Audio / Omni speech output |
| `lfm2moe` | LFM2 MoE |
| `gemma4` | Gemma 4 text model |
| `qwen3`, `qwen35` | Qwen3 dense; Qwen3.5 text model |
| `gptoss` | gpt-oss |
| `plamo`, `plamo3` | PLaMo bidirectional encoder (PLaMo-Embedding); PLaMo 3 |
| `bert` | BERT encoders (e.g. all-MiniLM-L6-v2) |
| `whisper` | Whisper |

Model families are described in [llm-models.md](llm-models.md), [llm-qwen.md](llm-qwen.md),
[moe.md](moe.md), [audio-models.md](audio-models.md). Image and vision models have their own
converters ([image-models.md](image-models.md), [vision-models.md](vision-models.md)).

## Milestones still open

Device listing, Core ML with strict mode, Intel (native blob and IR), power policy and compile cache,
AMD on Linux (xclbin and full ELF), and small LLMs on every NPU are done. Left:

| Item | State |
|---|---|
| Windows, Intel | Supported; where the driver installs `ze_loader.dll` is not verified |
| Windows, AMD | `xrt` backend tested on Linux against XRT 2.21; not yet run on Windows |
| AMD user-mode queues (UMQ) | not started |
| Qualcomm QNN | not started |
| Virtualized devices (VM, cloud) | not started (see Open) |

## Open

- **Intel power.** Energy comes from RAPL (`ScopePackage`, the whole CPU package, usually root). Whether
  the metric streamer gives the NPU alone is not checked.
- **Virtualization.** A client NPU is one PCIe device per SoC, so VMs and clouds cannot reach it today.
  AMD's shim has a `virtio` directory and the UAPI has a PF device type, which may mean a
  para-virtualized path is coming (a guess). Keep backend boundaries so a new backend can be added under
  `Device`.
- **purego fakecgo off glibc.** Whether `dlopen` works with `CGO_ENABLED=0` on musl / Alpine is not
  checked; static links are known not to work.
- **MLComputePlan is an estimate.** `Verified` on Core ML is not a guarantee of runtime placement.

## References

- Intel Level Zero NPU extensions: https://github.com/intel/level-zero-npu-extensions (`ze_graph_ext.h`, `ze_command_queue_npu_ext.h`)
- Intel NPU UMD: https://github.com/intel/linux-npu-driver (`docs/overview.md`); KMD UAPI `include/uapi/drm/ivpu_accel.h`
- OpenVINO NPU plugin: `src/plugins/intel_npu/src/compiler_adapter/src/{model_serializer,ze_graph_ext_wrappers,driver_compiler_adapter}.cpp`
- AMD KMD UAPI `include/uapi/drm/amdxdna_accel.h`, `Documentation/accel/amdxdna/amdnpu.rst`
- AMD XRT shim: https://github.com/amd/xdna-driver (`src/shim/{hwctx,hwq,buffer,device,pcidev,pcidrv}.cpp`, `src/shim/kmq/`)
- XRT headers: `xrt/detail/ert.h`, `xrt/detail/xclbin.h`, `xrt/xrt_kernel.h`, `xrt/xrt_hw_context.h`, `xrt/experimental/xrt_module.h`
- mlir-aie Windows host build: https://github.com/Xilinx/mlir-aie/blob/main/docs/buildHostWin.md
- Ryzen AI docs (xrt-smi on Windows): https://ryzenai.docs.amd.com/en/latest/xrt_smi.html
- purego: https://github.com/ebitengine/purego

## Not adopted

| What | Why |
|---|---|
| A generic importer (ONNX etc.) in the converter | NPU rewrites are hard to put into imported graphs; LLMs have few structural patterns, so hand ports are cheaper |
| ONNX as an intermediate format | Written straight from `config.json` + weights |
| Direct ioctl on Intel | The firmware job format is private to the UMD |
| XRT's C API from purego | C++-only contexts, deprecated kernel open, variadic calls |
| ONNX Runtime's EPs as a backend | EPs split graphs and send ops to the CPU silently, so strict mode cannot hold; ONNX Runtime is used only as a CPU executor (`internal/onnxrt`, float32, CPU EP) for models `models.yaml` places on the CPU |
| A separate `contrib/` module, `cmd/hwnpuinfo`, public `xdna/` and `levelzero/` helper packages | Not needed: the CPU runner lives in `internal/onnxrt`, diagnostics in `hwnpu-dev info` and `hwnpu status` |
| Re-quantizing published quantized checkpoints | Each target is quantized from bf16; a quantized checkpoint is used only when its format can be rearranged without loss |
| 32-bit platforms | All C structs are written for 64 bit |
