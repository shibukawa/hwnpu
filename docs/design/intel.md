# Intel NPU

[日本語](intel.ja.md)

Intel's NPU through Level Zero and its graph extension (`internal/ze`, `internal/backend/levelzero`), with programs written as OpenVINO IR by Go (`internal/ovlower`, `internal/ovir`). The plan for each model on Intel and the op table are in [../intel-npu-models.md](../intel-npu-models.md).

## State

- Real hardware: HP EliteBook 630 G11, Core Ultra 5 135U (Meteor Lake, NPU 3720), 32 GB, Windows 11 25H2, NPU driver 32.0.100.5540 (Level Zero 1.16, `ZE_extension_graph` 1.18, compiler 8.3). Run with `tools/intel/hwnpu-intel-check.ps1` (levels standard / full).
- Runs on it, all on the NPU: `report`, LFM3-1.7B (`--paged=-1 --context 4096`, and the default 262144 context in 7 parts; 17–25 tok/s with drafts), thinking answers, `hwnpu-chat agent` read and write, PLaMo 3 Translate 0.8B (ja→en and en→ja, 25–31 tok/s).
- Everything else (the vision, audio and image commands, Gemma 4, Qwen3, Qwen3.5, Whisper) is checked only on OpenVINO's CPU (below). Whether the NPU compiler accepts those graphs, their speed and their memory are open.

## Loading the library

1. `HWNPU_ZE_LOADER`, 2. `libze_loader.so.1` / `ze_loader.dll` on the normal search path, 3. known places (Linux `/usr/lib/x86_64-linux-gnu`, `/usr/local/lib`; Windows `%SystemRoot%\System32`, oneAPI's `bin`). On failure `ErrLibrary` lists the paths tried. `internal/dl` loads with purego (Unix) or `syscall.LoadLibrary` (Windows). No cgo.

## Init and device

```
zeInit(ZE_INIT_FLAG_VPU_ONLY)
zeDriverGet → zeDeviceGet → properties.type == ZE_DEVICE_TYPE_VPU
zeDriverGetExtensionProperties → "ZE_extension_graph" or "ZE_extension_graph_<n>..." (OpenVINO's rule)
zeDriverGetExtensionFunctionAddress → ze_graph_dditable_ext_t; likewise "ZE_extension_command_queue_npu" (optional)
zeContextCreate; pfnDeviceGetGraphProperties → graph extension version, compiler version, formats
```

- `CapCompile` only when `graphFormatsSupported` has `NGRAPH_LITE`.
- The DDI table is read as `[N]uintptr`; only entries up to the negotiated version are bound with `purego.RegisterFunc`. Target 1.20, minimum 1.5 (`Create2`, `ze_graph_desc_2_t`); below that `VersionError`. From 1.8, `GetProperties2`'s `initStageRequired` decides whether `GraphInitialize` is called directly (as OpenVINO).
- On Linux, `internal/accelfs` scans `/sys/class/accel` (`intel_vpu` / `amdxdna`), so "device but no library" is told from "no device". Windows has no sysfs: devices come from `zeDeviceGet` only.

| Extension version | Entries added |
|---|---|
| 1.0 | Create, Destroy, GetProperties, GetArgumentProperties, SetArgumentValue, AppendGraphInitialize, AppendGraphExecute, GetNativeBinary, DeviceGetGraphProperties |
| 1.1 | GraphGetArgumentMetadata, GetArgumentProperties2 |
| 1.2 | GetArgumentProperties3 |
| 1.3 | QueryNetworkCreate, QueryNetworkDestroy, QueryNetworkGetSupportedLayers |
| 1.4 | BuildLogGetString |
| 1.5 | Create2, QueryNetworkCreate2, QueryContextMemory |
| 1.6 | DeviceGetGraphProperties2 |
| 1.7 | GetNativeBinary2 |
| 1.8 | GetProperties2, GraphInitialize |
| 1.11 | CompilerGetSupportedOptions, CompilerIsOptionSupported |
| 1.12 | Create3, GetProperties3, BuildLogGetString2, BuildLogDestroy |
| 1.15 | SetArgumentValue2 |
| 1.16 | Evict |
| 1.20 | GetArgumentProperties4, GetArgumentNames |

## Graphs

- Native blob: `ze_graph_desc_2_t{format: NATIVE}`.
- OpenVINO IR: the same bytes as OpenVINO's `model_serializer.cpp` (ALL_WEIGHTS_COPY): compiler version, `numberOfInputData = 2`, xml size + xml, weights size + weights. `internal/ovir` also inserts the runtime info `is_new_api` and `use_indices_for_io_metadata`.
- Build flags follow OpenVINO's `serializeIOInfo` / `serializeConfig` (`--inputs_precisions=… --config KEY="VALUE"`), built by `levelzero.BuildFlags`. The supported-layers query gets only the `--config` part; its answer is `<name0><name1>…`.
- Build log: `pfnBuildLogGetString` with a NULL graph handle, into `BuildError.Log`.
- Compiled blobs are cached (`WithCacheDir`); a load from the cache skips the compiler.

Placement check on Intel (the NPU compiler never hands ops to another device):

| Case | Result |
|---|---|
| Compiled blob | `Verified`, `Method = "levelzero-native-blob"` |
| IR compiled | `Verified`, `Method = "levelzero-compiler"` |
| IR failed | Query supported layers: any unsupported → `PlacementError` (`Device = "unsupported"`, `Cause` = `BuildError` with the log), whatever the placement policy; none → `BuildError` |

## Arguments, memory, execution

- `AllocBuffer` = `zeMemAllocHost(align 4096)`, seen as `[]byte`; freed with `zeMemFree`. `SetArgumentValue` takes only Level Zero memory, so `Bind` takes only `*Buffer` (also right under cgo's pointer rules). The host reads it directly (`CapHostCoherent`); Sync does nothing.
- Argument metadata from `GetArgumentProperties3` → `ArgInfo` (shape, `DType`, direction). `ArgInfo.ModelDType` is the model's precision (an fp32 model may run in fp16).
- Execution: command queue (optional NPU extension for turbo) → command list → first `GraphInitialize` or `AppendGraphInitialize` → `AppendGraphExecute` → close → execute with a fence. `Fence.Wait` loops `zeFenceHostSynchronize(50 ms)` while watching `ctx.Done()`.
- Argument values are captured when the command is appended, so each request is re-recorded from `SetArgumentValue` (a lock per graph). `Bind` on a finished request is allowed (rebinding is what the paged attention chain does).
- C layouts (`ze_graph_desc_2_t`, `ze_graph_argument_properties_3_t`, …) are Go structs checked by golden tests against `testdata/abi/levelzero_abi.json` (sizes and offsets from the official headers; clang on macOS arm64 and gcc on Linux amd64 agree).
- `testdata/zemock/zemock.c` is a fake driver built with `cc` at test start and loaded through `HWNPU_ZE_LOADER`.

## Programs for Intel

- One converter for Apple and Intel: the architecture builds MIL, `internal/ovlower` lowers it op by op to IR (`out.xml` + `out.bin`, `out.partN.xml`, `out.parts.json` with `meta`). Unknown ops or inputs stop with an error. Every milcpu op lowers except `read_state` (probes only). Op table: [../intel-npu-models.md](../intel-npu-models.md).
- Where programs go: `<model>/intel/` (hwnpu-chat), extension `.intel` in an export directory. Converters take `Backend` (`-backend intel`, `hwnpu-image convert --backend intel`, `depth.Setup.Backend`, `sana.ConvertOptions.Backend`, `systemone.Setup.Backend`, `music.WriteProgram` / `llm.WriteIR`).
- Commands treat the machine as Intel through `chat.HereVendor`; `llm.ProgramBackend` takes the program form name (`intel`) as well as the backend name.
- Weight formats: the NPU cannot read lookup tables, so `llm.IntelFormat` maps `lut4` → `int4` and `lut6` / `lut8` → `int8`; lut weights are rejected with that hint. Arch overrides (`Arch.IntelFormat`, `llm.IntelFormatter`) choose int8 for LFM3, BERT, PLaMo-Embedding, PLaMo 3 ≤ 2B; int4 per row for PLaMo 3 8B (hidden ≥ 4096); SANA 1.5 1.6B's DiT stays fp16 (`sana.FormatsFor`: per-row int8 breaks the picture); MOSS's text model keeps 5 layers fp16. Details: [quantization](quantization.md).
- LLMs split at the attention (`llm.convertPaged`): parts `out.partN.xml` and the attention program `out.attn.xml` (built in MIL, lowered by `ovlower`). Intel's default LFM3 form: `lfm3-int8-grouped-p8192-c262144-q8.parts.json`. Engine side: [llm-engine](llm-engine.md).
- hwnpu-chat keeps two programs per model on Intel (8 and 64 tokens, `onePrograms` off), so weights are held twice; `prefill: off` in models.yaml keeps one (used for PLaMo 3 8B, 3.8 GiB at int4).
- The `.bin` of the 8- and 64-token programs is shared on disk where the file system can clone ([distribution](distribution.md)).

### Weight formats on Meteor Lake (`hwnpu-formats`, LFM3-1.7B, cache 4096, 8 tokens a call)

| Format | ms a call | Next token = fp16 | Perplexity |
|---|---|---|---|
| fp16 | 112.1 | — | 4.57 |
| int8 | 68.5 | 99 % | 4.56 |
| int4-g128 | 84.4 | 76 % | 6.13 |
| int4 (per row) | 37.9 | 63 % | 10.32 |

int8 stays the default: per-group int4 is slower than int8 here, per-row int4 is fast but not accurate enough.

## Checking without the NPU (OpenVINO's CPU)

- `HWNPU_BACKEND=openvino` (or `--backend openvino`, device `HWNPU_OV_DEVICE`) runs Intel's IR on OpenVINO's CPU plugin (`internal/backend/openvino`, Python via `HWNPU_OV_PYTHON`, `serve_ir.py` / `run_ir.py`). Same conversion, argument names and host code as the NPU; only the compiler and device differ. pip's openvino also works on a Mac.
- `serve_ir.py` and `run_ir.py` block the `openvino_telemetry` module (its side process read deleted scripts and failed).
- Opening the backend cleans `/dev/shm` buffer directories left by crashed processes (8.4 GB in a day).
- `hwnpu-dev irdiff A.mlpackage …`: lowers a Core ML package to IR and compares milcpu with OpenVINO's CPU on the same random inputs — finds ops that do not lower before writing an Intel path.
- Tests `TestIntelIR*` (lfm3, gemma4, lfm3vl, lfm3audio, plamo3, bert, plamo, qwen3, qwen35 fp16/int8, whisper; paged forms for gemma4 and lfm3): relative error ≤ 1e-3 against milcpu (paged: 2.8e-3 gemma4, 5.7e-4 lfm3). `TestImageOps`: ≤ 3e-4.
- Real checkpoints with `HWNPU_BACKEND=openvino` on the Ryzen AI machine's CPU agree with the CPU, AMD's NPU or milcpu: RTMO-s pose, RTMDet-m detect (boxes within 0.1 px), PP-OCRv6 small, SwiftF0 (f0 within 0.013 %), Basic Pitch (notes identical), YuNet, DA3-SMALL (depth corr 0.9994), MoGe-2 ViT-S (0.9998), EfficientSAM-S (same mask), Qwen3.5-0.8B int8 chat, LFM3-Audio-2B TTS → Whisper large-v3-turbo round trip, Magenta RT2 small (12 codes the same), Sana 600M 1024² int8, MOSS-SoundEffect v2.0 (16 programs, 18 s a step, not listened to), Anima (latent cosine 0.99985, PSNR 37.9 dB in f32; 25.5 dB with f16 inference).

## Memory limits and residency

- The driver keeps graph weights in memory the NPU can see. On the 135U, Anima 512² fp16 failed at `dit_blocks_24.xml` with `pfnGraphInitialize: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY` at about 4.6 GB held (text 1.2 GB + 6 DiT groups of 0.57 GB). Yet `hwnpu-dev probe resident` stacks 0.23 GiB programs to 16 GB without error, so the limit is not a plain total; per-program size or the real graphs' shape is still to be split (`resident-large`: 0.57 and 1.2 GiB programs).
- A load can also fail as the compiler's `std::bad_alloc` ("bad allocation"). `types.IsOutOfMemory` treats Level Zero's `OUT_OF_HOST_MEMORY` / `OUT_OF_DEVICE_MEMORY` and "bad allocation" / "bad_alloc" as out of memory.
- Anima on Intel (`anima.progEngine`): parts are slots; on out of memory, close other parts and load again. Order: text (once a prompt) first, then the DiT group used most recently (MRU: with 7 groups and 4 places, 4 reloads a step; LRU would be 7), then time, patch and final layers. The VAE stage makes room the same way. Tested with fake NPUs (`residency_test.go`); on the real NPU the reload first worked at 1024².
- `hwnpu-dev probe resident`: loads 0.25 GiB programs one by one (kept open, run once each) until a load fails or `-resident-max-gb` (60 % of RAM, ≤ 16); then runs all again; on out of memory closes the oldest and retries. Flags `-resident-hold open|close`, `-resident-formats int8`, `-resident-size a,b`, `-resident-run=false` (load only), `-resident-cycle`. On Windows it records the process's private memory (`K32GetProcessMemoryInfo`) after each load; `maxRSS` returns the peak working set. The child reports each program as an `npuprobe-progress` line and each stage on stderr, so a crash keeps what was loaded.
- One run ended with the child killed by 0xc0000005 inside the driver while loading. If out of memory can be a crash rather than an error, the close-and-reload above does not help, and a self-imposed limit below the crash total is needed.

### Windows job time limit (TDR)

- Windows stops a GPU/NPU job past its time limit (TDR, 2 s by default) and the device is lost (`zeFenceHostSynchronize: ZE_RESULT_ERROR_DEVICE_LOST`). Anima 1024² with 4 DiT blocks a program hit it (estimated 2–4 s a group).
- `--group` default 0 = auto, `anima.DefaultGroup`: Core ML and Core AI 4; Intel halves from 4 until blocks × patches ≤ 4096 (512²: 4, 640²: 2, 768² and 1024²: 1). `g` is in the export name. On DEVICE_LOST (`types.IsDeviceLost`) with group ≥ 2 the error suggests halving `--group`.
- g4 and g1 exports give the same picture (latent difference 3.6 % relative, rounding between groups).

## Windows

- The same code runs through `ze_loader.dll` on the PATH (`ZE_ENABLE_LOADER_DEBUG_TRACE=1` shows the load path). Checked on the 135U above.
- Disk sharing of IR uses ReFS / Dev Drive block cloning where present ([distribution](distribution.md)).

## Open

- Whether the NPU compiler accepts the ops added for the non-chat models (DepthToSpace, ConvolutionBackpropData, Clamp, MaxPool, Elu, …) and the large graphs (Sana's Gemma 2 6 layers int8 ~0.5 GB, EfficientSAM's 4096-token attention); speed; whether Sana and MOSS fit the 135U's memory.
- Where the 4.6 GB failure comes from (program size or graph shape, not the total); whether the 0xc0000005 crash is out of memory.
- Anima on the NPU after g1: the reload after "bad allocation" (text and a few blocks a prompt expected); int8 Anima (`image-512-int8`) against fp16.
- Cost of re-recording command lists for attention over chunks (full-attention layers × chunks used, each step).
- Real-NPU runs of Gemma 4, LFM3-VL / Audio, Qwen3.5, Whisper, d1, PLaMo 3 8B int4 (two programs or one).
- A size limit like Apple's 1 GiB: the 2 GiB of `-target intel` is a placeholder. Compile times of large graphs.
- Whether the compiler can share weights between the 8- and 64-token graphs (today each holds its own).
- Speed on Lunar Lake and later (only Meteor Lake measured); whether per-group int4 behaves differently there.
- Running two models at once (the driver manages the queues; not measured).
- Chroma, SigLIP 2 (label) and FLUX have only IRON graphs (Python), no Go MIL graph; Qwen-Image has a Go graph but writes no IR yet. They need a port to `internal/llm/blocks` first.
- A Level Zero "loader bypass" mode (load the driver DLL and read its DDI tables) if some Windows setups ship no loader — not needed so far.
- `ZE_GRAPH_FLAG_INPUT_GRAPH_PERSISTENT` to skip copying a page-aligned blob; `zeMutableCommandList` if re-recording costs show.

## Not adopted

- int4 (per row or per group) as Intel's default: see the Meteor Lake table (int4 per row 63 % next-token agreement; LFM3 22/52 in the first CPU checks).
- `-idle` release on Intel: reloading Intel's programs takes tens of seconds to minutes; only AMD (IRON) releases ([amd](amd.md)).
- LRU for Anima's DiT groups: MRU reloads 4 groups a step against 7.
