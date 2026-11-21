# Public Go API

[日本語](api.ja.md)

The packages other modules can import: `github.com/shibukawa/hwnpu` (devices, buffers, graphs,
kernels), `modelstore` (model files) and `npuinfo` (what the OS says about NPUs). The API is not frozen
yet (`USAGE_LIBRARY.md` says "planned"); model engines stay in `internal/` until their shape is settled.
How-to examples: [../../USAGE_LIBRARY.md](../../USAGE_LIBRARY.md).

The one decision that shapes the API: share devices, buffers and completion; split submitting work by
capability (`GraphRunner` for compiled graphs, `KernelRunner` for kernel binaries). See
[architecture.md](architecture.md).

## Devices

```go
type Vendor uint8      // VendorUnknown, VendorIntel, VendorAMD, VendorApple
type Capability uint32 // CapGraph, CapCompile, CapKernel, CapHostCoherent, CapStrictVerify, CapPowerControl, CapPowerMeter

type DeviceInfo struct {
	Index    int
	Vendor   Vendor
	Backend  string // "levelzero", "openvino", "xdna", "xrt", "coreml", "coreai"
	ID       string // backend-specific, used by OpenDevice
	Name     string
	Path     string // device node, empty when none
	Caps     Capability
	Driver   string
	Firmware string
}

func Enumerate(ctx context.Context) ([]DeviceInfo, error)  // every backend compiled in; failures joined, others still listed
func Open(ctx context.Context, opts ...Option) (*Device, error) // first match; ErrNoDevice (joined with listing errors)
func OpenDevice(ctx context.Context, info DeviceInfo, opts ...Option) (*Device, error)

func (d *Device) Info() DeviceInfo
func (d *Device) Close() error // close graphs, requests and buffers first
func (d *Device) AllocBuffer(size int, opts ...BufferOption) (*Buffer, error)
func (d *Device) Graph() (GraphRunner, error)   // ErrUnsupported without CapGraph
func (d *Device) Kernel() (KernelRunner, error) // ErrUnsupported without CapKernel
func (d *Device) PowerPolicy() PowerPolicy
func (d *Device) PowerSample() (PowerSample, error) // ErrUnsupported without CapPowerMeter
```

Capabilities by backend (from the providers):

| Backend | Caps |
|---|---|
| `coreml` | Graph, Compile, HostCoherent, StrictVerify (when `MLComputePlan` exists) |
| `coreai` | Graph, Compile, HostCoherent, StrictVerify |
| `levelzero` | Graph, HostCoherent, StrictVerify, PowerControl; Compile with the driver compiler; PowerMeter with readable RAPL |
| `openvino` | Graph, Compile, HostCoherent |
| `xdna` | Kernel, PowerControl; PowerMeter with the driver's sensor or the APU's package power (amdgpu hwmon) |
| `xrt` | Kernel; PowerMeter where available |

Registration order: Core ML before Core AI on darwin (Core ML is the default backend for the library;
`hwnpu-chat` on a Mac uses Core AI only, see [cli.md](cli.md)); `xrt` before `xdna`, listing itself
outside Windows only with `HWNPU_BACKEND=xrt`; `openvino` only with `HWNPU_BACKEND=openvino`.

## Buffers and completion

```go
func (b *Buffer) Bytes() []byte                  // zero-copy view, invalid after Close
func (b *Buffer) Size() int
func (b *Buffer) DeviceAddress() (uint64, bool)  // AMD: the BO's address (host buffer: user VA via PASID; instruction buffer: device heap)
func (b *Buffer) SyncToDevice(off, n int) error  // after CPU writes, before Submit; no-op with CapHostCoherent
func (b *Buffer) SyncFromDevice(off, n int) error // after Wait, before CPU reads
func (b *Buffer) Close() error

func WithBufferKind(k BufferKind) BufferOption   // BufferDefault, BufferInstructions (AMD control code, from the device heap)

type Fence interface {
	Wait(ctx context.Context) error // on cancel the submission keeps running; no resubmit until it finishes
	Done() (bool, error)
}
```

- Intel and Apple are host coherent: Sync is a no-op.
- AMD is not coherent. On x86 Sync flushes cache lines in user space (CLFLUSH, as XRT does); elsewhere it
  uses the driver's `SYNC_BO`.

## GraphRunner (Intel, Apple)

```go
type GraphSource struct {
	Format     GraphFormat // GraphFormatNative (vendor blob) or GraphFormatOpenVINOIR (needs CapCompile)
	Path       string      // file or directory (Core ML: .mlpackage, .mlmodel, .mlmodelc)
	Blob       []byte      // native blob in memory
	XML, Bin   []byte      // OpenVINO IR
	BuildFlags string      // e.g. `--inputs_precisions="0:fp16" --config NPU_PLATFORM="4000"`
	Function   string      // one function of a multi-function ML Program (macOS 15+); functions share weights
	Units      string      // Core ML: "" Neural Engine (+CPU), "cpu", "gpu"; HWNPU_COREML_UNITS overrides
}

type GraphRunner interface {
	Compile(ctx context.Context, src GraphSource) (*Graph, error) // strict mode: *PlacementError
}

type ArgInfo struct {
	Index      int
	Kind       ArgKind // ArgInput, ArgOutput
	Name       string
	Dims       []int
	DType      DType   // of the bound buffer: F16, F32, F64, BF16, I8, U8, I16, I32, I64
	ModelDType DType   // declared by the model; differs when the device converts (fp32 run as fp16 on Intel)
	Layout     string
	Bytes      int
	PaddedDims []int   // shape actually run when padded to reach the NPU; callers still use Dims/Bytes
}

func (g *Graph) Args() []ArgInfo          // inputs first, then outputs; the order is the Bind index
func (g *Graph) Arg(name string) (ArgInfo, bool)
func (g *Graph) AllocInput(index int) (*Buffer, error) // memory the backend runs from directly (Core AI); else AllocBuffer
func (g *Graph) Cached() bool             // loaded from the WithCacheDir cache
func (g *Graph) Placement() PlacementReport
func (g *Graph) NewRequest() (*Request, error)
func (g *Graph) Close() error

func (r *Request) Bind(index int, b *Buffer) error // buffer ≥ ArgInfo.Bytes, dense in DType/Dims
func (r *Request) BindName(name string, b *Buffer) error
func (r *Request) Submit() (Fence, error)
func (r *Request) Run(ctx context.Context) error   // Submit + Wait
func (r *Request) ResetState() error                // zero the model's states (Core ML MLState: KV caches)
func (r *Request) Close() error
```

- A request is used by one goroutine at a time.
- `AllocInput` is for inputs that are large and change little between runs (an LLM's caches): with
  Core AI a run then copies nothing. The buffer can be bound to any argument of that shape, in any
  graph of the device.
- Several shapes of one model (prompt and decode programs) share weights as functions of one ML
  Program (`Function`). On Intel each shape is its own graph.
- `EncodeF16` / `DecodeF16` convert between `[]float32` and little-endian fp16 bytes.

## KernelRunner (AMD)

```go
type ProgramSource struct {
	Xclbin []byte // xclbin contents
	ELF    []byte // or a full ELF (AIE2P), as IRON and aiecc write for Strix Point and later
	Path   string // or a file of either kind (told apart by its contents)
}

type KernelRunner interface {
	Load(ctx context.Context, src ProgramSource) (*Program, error)
}

func (p *Program) Kernels() []string // a Program = one hardware context
func (p *Program) Kernel(name string) (*Kernel, error)
func (p *Program) Close() error

func (k *Kernel) Name() string
func (k *Kernel) Args() []KernelArg // Index, Name, Type (C type), Offset/Size in the register map, Buffer
func (k *Kernel) NewRun() (*Run, error)

func (r *Run) SetBuffer(i int, b *Buffer) *Run
func (r *Run) SetBufferAt(i int, b *Buffer, off, n int) *Run // device gets b's address + off
func (r *Run) SetUint32(i int, v uint32) *Run
func (r *Run) SetUint64(i int, v uint64) *Run
func (r *Run) SetBytes(i int, v []byte) *Run
func (r *Run) Err() error                 // first setter error
func (r *Run) Submit() (Fence, error)     // sync inputs first
func (r *Run) Run(ctx context.Context) error
func (r *Run) Scratchpad() []byte         // full ELF: the ctrl scratchpad written before Submit, or nil
func (r *Run) UseScratchpad(b *Buffer, off, n int) error // read parameters from device-written memory
func (r *Run) Close() error
```

- A run can be submitted again after it finishes, with the same or new arguments.
- IRON's usual call `kernel(opcode, bo_instr, instr_count, bo_in, bo_out)` becomes
  `k.NewRun()` → `SetUint64(0, 3)`, `SetBuffer(1, instr)` (a `BufferInstructions` buffer),
  `SetUint32(2, words)`, `SetBuffer(3, in)`, `SetBuffer(4, out)` → `Run(ctx)`.

## Strict mode

```go
type Placement uint8 // PlacementStrict (default), PlacementPrefer

type PlacementReport struct {
	Verified bool          // the backend could check
	Method   string        // e.g. "coreml-compute-plan"
	Reason   string        // why not verified
	NPUOps   int
	Ops      []OpPlacement // ops not on the NPU: Name, Type, Device ("cpu", "gpu", "unknown")
}
func (r PlacementReport) OnNPU() bool // Verified && len(Ops) == 0

type PlacementError struct {
	Report PlacementReport
	Cause  error // e.g. the compiler error when the NPU could not compile at all
}
```

`PlacementStrict` rejects a graph that is not verified to run entirely on the NPU, including one whose
placement cannot be verified; `errors.Is(err, ErrNotOnNPU)` holds. `PlacementPrefer` keeps the graph and
records the result in `Graph.Placement`.

| Backend | How placement is verified |
|---|---|
| Core ML | `MLComputePlan`: every op's preferred device must be the Neural Engine. An estimate, not a runtime guarantee |
| Core AI | its compile cache: the MPSGraph delegate's manifest (`mps.fullyPlacedOnANE`; method `coreai-cache-manifest`) |
| Level Zero | supported-layer query (`levelzero-query-network`); the NPU plugin does not fall back, so a successful compile means it runs on the NPU |
| OpenVINO | verified only with `HWNPU_OV_DEVICE=NPU` |
| AMD | xclbin / ELF are NPU-only: always satisfied |

## Power

```go
type PowerPolicy uint8 // PowerDefault, PowerLow, PowerBalanced, PowerPerformance
type PowerScope string // ScopeNPU ("npu"), ScopePackage ("package": cores, GPU and NPU)

type PowerSample struct {
	Time    time.Time
	EnergyJ float64 // cumulative, monotonic for one meter
	Watts   float64 // instantaneous when the meter reports it (AMD sensors), else 0
	Scope   PowerScope
	Source  string  // e.g. "rapl:intel-rapl:0"
}
func AveragePower(a, b PowerSample) float64 // (b.EnergyJ - a.EnergyJ) / dt
```

| Backend | `PowerPolicy` maps to | Meter |
|---|---|---|
| Level Zero | Low: low queue priority + "background" workload type; Balanced: normal priority; Performance: high priority + turbo (turbo and workload type need the extension) | RAPL, package scope, usually root |
| xdna | hardware context QoS priority (Low / Normal / High) | driver sensor, or amdgpu package power |
| Core ML / Core AI | none (no CapPowerControl) | none in the API; see `npuinfo` (IOReport) and `hwnpu-dev powerbench` |

More in [memory-and-power.md](memory-and-power.md).

## Errors

```go
var (
	ErrNoDevice    // no NPU found
	ErrUnsupported // not supported by this device/backend
	ErrLibrary     // runtime library not found (wraps the paths tried)
	ErrNotOnNPU    // model does not run entirely on the NPU (*PlacementError)
	ErrClosed      // already closed
)
type DriverError  struct{ Backend, Op string; Code int64; Msg string } // ze_result_t / errno
type BuildError   struct{ Log string; Err error }                      // compile failure with the build log
type VersionError struct{ Component, Have, Need string }
```

## Options

| Option | Effect |
|---|---|
| `WithVendor(Vendor)`, `WithBackend(string)`, `WithIndex(int)` | Pick the device in `Open` |
| `WithPlacement(Placement)` | Default `PlacementStrict` |
| `WithPowerPolicy(PowerPolicy)` | Ignored without CapPowerControl |
| `WithCacheDir(dir)` | Store compiled graphs and reuse them. Key: the model, the build flags, the driver / compiler (or OS) version. Compiling costs power, so it is not repeated each start |

Environment: `HWNPU_BACKEND` (backend for the commands; makes `openvino` / `xrt` list),
`HWNPU_ZE_LOADER` (path of the Level Zero loader), `HWNPU_COREML_UNITS` and `HWNPU_COREAI_UNITS`
(move programs off the Neural Engine, e.g. `cpu` to separate engine errors from model errors).

## modelstore

`github.com/shibukawa/hwnpu/modelstore` finds and fetches model files the way hwnpu's commands do.
Layers, in order:

| # | Layer | Set by |
|---|---|---|
| 0 | this machine's cache (always written back to) | `Options.Root` (`HWNPU_CACHE`, else `hwnpu` in the user cache directory) |
| 1 | sources: `file://`, `http(s)://` (WebDAV or any read-only server), `s3://`; `?writeback=true` stores what later layers give | `Options.Sources` (`HWNPU_MODEL_SOURCE`, `HWNPU_MODEL_SOURCE_1`, …) |
| 2 | bundled models, read only, used in place (hard link, else symlink) | `Options.Bundle` (a `configdir.ConfigDir`), `BundleFolder` |
| 3 | Hugging Face | unless `Options.HuggingFace` is false (`HWNPU_MODEL_HUGGING_FACE`) |

- A repository's files are those its `hwnpu-lock.json` names, of one commit (the model's pin, else the
  first lock after the local cache, else the local one, else Hugging Face's `main`). Files live under
  `original/OWNER/NAME` of the root (`Store.Dir`).
- The layers are a process setting: `Open` sets them for all of hwnpu's code in the process.
- API: `Open`, `File(ctx, repo, name)`, `Weights(ctx, repo)`, `Dir`, `Layers`, `NotFound(err)`;
  disk: `Usage`, `Limits` (`Options.Limits`), `DiskSpace`, `PlanPrune` / `Prune`, `Remove`,
  `PlanClean` / `Clean` (sources), `ErrNoRoom`; memory class per model: `Memory`, `SetMemory`, `Memories`.

Design: [distribution.md](distribution.md), [../cache-sharing-design.md](../cache-sharing-design.md).

## npuinfo

`github.com/shibukawa/hwnpu/npuinfo` reads what the OS and drivers tell about NPUs: devices, the
processes using them, how busy they are, power, memory. Activity Monitor and the usual Linux tools show
none of this.

```go
info, err := npuinfo.GetWithContext(ctx, npuinfo.Options{Wake: false, SampleInterval: 0, SkipCaches: false})
// info.Linux / info.Windows / info.Darwin (only the running OS), info.Errors (partial failures)
```

- Standard library only, nothing else from hwnpu, so it can become its own module (gopsutil etc.). Types
  build on every OS so a viewer can decode another machine's JSON. `go run ./npuinfo/cmd/npuinfo`
  writes JSON. Per-OS fields are in [../../npuinfo/README.md](../../npuinfo/README.md).
- OS-specific structs as the OS gives them; a common API comes later.
- **Reading must not wake the NPU.** amdxdna's `GET_INFO` / `GET_ARRAY` resume a sleeping NPU and keep
  it up for the 5 s autosuspend delay, so they run only while `power/runtime_status` is active (unless
  `Options.Wake`). Opening the device, sysfs (not PCI config space) and `/proc` do not wake it. On macOS,
  40 IOReport reads left the ANE's power state at 0.
- `SampleInterval` (default 500 ms) is the gap between two reads of rate counters (IOReport, Windows
  counters); below 0 skips them on macOS.

What it reads (measured on a Ryzen AI 9 365, Linux 7.0, firmware 1.1.2.64, and an M1, macOS 27.2.1):

| | Linux (amdxdna) | macOS |
|---|---|---|
| Per process | `GET_ARRAY(HW_CONTEXT_ALL)` without root: PID, columns, commands submitted / completed, state (`gops`, `fps`, `heap_usage` stay 0) | `proc_pid_rusage(RUSAGE_INFO_V6).ri_neural_footprint` (not in `phys_footprint`); other users' processes need root |
| NPU memory | sum of `/proc/PID/maps` lines of `/dev/accel/accel0` (LFM3 q4: 1635 MiB; not in RSS or VmPin) | the footprint above (234 MiB for a 0.23 GiB lut4 program) |
| Busy, power | amdgpu `gpu_metrics` v3: `average_ipu_activity` (93 % in decode), `average_ipu_power` (1026 mW), NPU reads | IOReport (private, no root): `PMP/Energy Counters/ANE` (mJ), `AMC Stats` ANE RD/WR bytes, ANE interrupts; all 26 power domains as `soc_power_w`, all DRAM clients as `dram_bytes_per_sec` |
| Other | AIE version and shape, clocks, power mode, TOPS and task limits, per-column AIE status (raw), telemetry (raw), last async error | I/O Registry ANE classes (`H11ANEIn`, load-balancer clients), `CurrentPowerState`, ANE kext and framework versions, system ANE processes, thermal pressure, Low Power Mode |
| Not available | fdinfo `drm-*` (not implemented by amdxdna); sensor query ENODEV on that BIOS | M1's `Energy Model` channels are all 0 (dropped) |

Windows code exists (PnP ComputeAccelerator devices, D3DKMT adapters including compute-only, every
performance counter set whose name contains GPU, NPU or Neural) but has only been cross-built.

## Library `ai` for applications (v0)

hwnpu is becoming a set of Go libraries for running models on an NPU; `hwnpu-chat`, `hwnpu-image` and
`hwnpu-audio` become demos of them (names unchanged). Design and tasks: [../app-api.md](../app-api.md),
[../app-api-tasks.md](../app-api-tasks.md); usage and rules: [../developer-guide.md](../developer-guide.md).

| Package | Content | State |
|---|---|---|
| `ai` | `App` registers the models an app uses; `Use[T]` is a typed handle (`stt.Get(ctx, rt)` returns `*speech.Transcriber`); `Plan` (what is missing, download size, memory, licenses; two stages: download/convert, then compile on the NPU); `Runtime` loads only from the prepared store | v0 |
| `ai/speech`, `ai/text` | Transcribe, speak, loudness VAD, WAV; translate, chat. Each package owns its models' preparation and loading (`internal/airole.Role`); `ai` imports none of them, so an app links only what it uses | v0 |
| `ai/flow` | Chunk streams (`Stream[T]`, `iter.Seq2`) and the network between models | v0 |
| `ai/cli` (+ `konghelper`) | Subcommands for an app: `models`, `models download`, `models prepare`, `status` (shared with `hwnpu status` through `internal/npustatus`) | v0 |
| `ai/vision`, `ai/imagegen`, `ai/sound` | Depth, faces, detection…; image generation; music, sound effects, pitch | not yet |

- Preparation calls the same functions as the commands' first run (`transcribe.Prepare`,
  `speech.Prepare`, `chat.Prepare`, their AMD forms) into the same store, so apps and commands share it.
- Line-up per machine memory: a use picks the largest model for one model alone; an app running several
  sets `use.At(gb, name)` / `use.As(gb)` (`samples/interpreter`: translation 2B from 16 GB, 8B from
  32 GB). At run time a store entry for another model or memory class is `ErrNotPrepared`.
  `Plan.UsableGB` / `Plan.Fits` compare the sum loaded at once with the memory usable for models.
- Joining models, by when the next stage may start: after all of the previous (`Last`, `Use.Release`
  before loading a big model); as pieces arrive (`Async`, `Latest`, `Hold`, `Tee`); on a beat
  (`Metronome`, `Pace`, `Live`). Declared as a network: `flow.Source` → nodes → `flow.Sink` with
  `a.To(b, edge)` (edges: none, `flow.Keep(n)`, `flow.Newest()`), program nodes `flow.Each` /
  `Route` / `Both` / `With`, model nodes (`speech.TranscribeNode`, `text.TranslateNode`, …), one call
  per request with `ai.NewCall(...).Then(...)`. Clock tools are tested with `testing/synctest`.
- Live input: the samples module (`samples/`, its own `go.mod`) takes the microphone from
  [hwcamera](https://github.com/shibukawa/hwcamera); `capture.Mic` buffers on its own goroutine
  (without it, speech during transcription was lost).

**The NPU is one queue** (Ryzen AI 9 365, whisper-large-v3-turbo + plamo-3-translate-8b +
LFM3-Audio-2B-TTS in one process): calls interleave but the total equals the sum (transcribe 1.38 s and
translate 3.34 s alone, 4.67 s together; all three 14.7 s against a sum of 14.8). Whether a chain keeps
up in real time is the sum of each stage's NPU time: TTS takes 2.6 s per second of audio, so spoken
output falls behind a continuous speaker; subtitles alone are 2.1 s (2B) or 3.7 s (8B) per utterance.
`samples/interpreter`: 13.3 s of Japanese to 8.5 s of English speech in 31.6 s, the output transcribed
back word for word.

**Parts meet at text, not vectors.** Hidden states are not compatible across models even with the same
tokenizer. Vectors connect only where the bridge was trained together (LFM3-Audio TTS reading the text
model's hidden state, multimodal towers, Anima's text adapter).

Each model records its "values as a part" (inputs and outputs, resident size, second load time,
longest command, streaming) per the [new-model checklist](new-model-checklist.md).

## Open

- `ai`: a residency manager (memory budget, submission order), nodes with three or more inputs, letting
  the library decide when to release models, per-edge measurement, `ai/vision` / `ai/imagegen`, moving
  the commands onto `ai`; `ai` not yet run on a Mac (Core AI) or Intel, nor `text.Chat` on hardware, nor
  a real microphone. Real parallelism by column split (AMD) not verified.
- `npuinfo`: run on Windows (Intel and AMD); check IOReport channel names on M2 and later; decode AIE
  status and telemetry; a common API; a browser viewer.
- The library API is marked planned; names may change.

## Not adopted

| What | Why |
|---|---|
| One interface hiding graph vs kernel execution | Intel and AMD differ at the root; hiding it would make both awkward |
| `CompileSet` (compile several sources at once, sharing weights) | Weight sharing is done by multi-function ML Programs (`GraphSource.Function`) |
| `Graph.NativeBlob` for app-side caching | `WithCacheDir` keeps compiled results inside hwnpu |
| `WithLibraryPath`, `HWNPU_ACCEL_DEVICE`, `WithTurbo`, `WithWorkloadType`, `WithQoS` | `WithPowerPolicy` maps to turbo, workload type and QoS; `HWNPU_ZE_LOADER` names the loader |
| Passing hidden states between separately trained models | Not compatible; needs training per pair |
| Publishing per-use interfaces now | Wait for a second working combination so the API is not frozen too early |
