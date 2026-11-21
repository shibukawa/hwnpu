# Using hwnpu as a Go library (planned)

[日本語](USAGE_LIBRARY.ja.md)

> **Planned**: the Go library API is not settled yet. Names and arguments may change without notice.

How to use hwnpu's Go API, and notes for each NPU. For the commands (hwnpu-chat, hwnpu-image, hwnpu-audio, hwnpu-dev), start from [README.md](README.md).

## Basics

```go
dev, err := hwnpu.Open(ctx)
if err != nil { ... }
defer dev.Close()

gr, _ := dev.Graph()
g, err := gr.Compile(ctx, hwnpu.GraphSource{Path: "model.mlpackage"})
if errors.Is(err, hwnpu.ErrNotOnNPU) {
	// err is a *hwnpu.PlacementError. The ops placed off the NPU are in Report.Ops
}
defer g.Close()

in, _ := g.Arg("x")
out, _ := g.Arg("y")
x, _ := dev.AllocBuffer(in.Bytes)
y, _ := dev.AllocBuffer(out.Bytes)
hwnpu.EncodeF16(x.Bytes(), input)

req, _ := g.NewRequest()
req.BindName("x", x)
req.BindName("y", y)
err = req.Run(ctx)
hwnpu.DecodeF16(result, y.Bytes())
```

## Power and cache

```go
dev, err := hwnpu.Open(ctx,
	hwnpu.WithPowerPolicy(hwnpu.PowerLow), // Intel: low priority + background
	hwnpu.WithCacheDir(cacheDir),          // save compiled results and reuse them
)

before, err := dev.PowerSample() // ErrUnsupported on devices whose energy cannot be read
// ... inference ...
after, _ := dev.PowerSample()
fmt.Println(hwnpu.AveragePower(before, after), "W", after.Scope) // "package" is the whole CPU package
```

- The power policy works only on devices with `CapPowerControl`. Core ML has no power setting
- On Intel, energy is read from RAPL on Linux. It is the value of the whole CPU package, not of the NPU alone, and usually needs root
- The cache key holds the model, the build flags and the versions of the driver and compiler (macOS on Apple), so a change in any of them rebuilds it

## Models' files (modelstore)

`github.com/shibukawa/hwnpu/modelstore` finds and fetches the files of the models as hwnpu's commands do: this machine's cache, then the sources (`HWNPU_MODEL_SOURCE`: folders, WebDAV and HTTP, S3), then the models the application installs with it, then Hugging Face (docs/cache-sharing-design.md 7).

```go
app := configdir.New("", "myapp") // github.com/shibukawa/configdir v1.1.0
s, err := modelstore.Open(modelstore.Options{
	Bundle: &app, // the installer's models/ in the app's resource folder (nil: none)
	// Root: "" (HWNPU_CACHE, else hwnpu in the user cache directory), Sources: nil (the environment's)
})
config, err := s.File(ctx, "LiquidAI/LFM3-1.7B", "config.json") // a local path
weights, err := s.Weights(ctx, "LiquidAI/LFM3-1.7B")            // model.safetensors (or its shards)
```

- The settings are the process's: `Open` sets them for everything in it
- `Options.Files` are definition files of the app's own (models.yaml's format: models, uses, aliases), read over hwnpu's, the machine's and the user's
- The bundled models are in the layout of a source (`original/OWNER/NAME/` with `hwnpu-lock.json`) and are used in place (linked, not copied)
- `Options.Limits` (nil: `HWNPU_CACHE_MAX_SIZE`, `HWNPU_CACHE_MIN_FREE`) bounds the cache: a download that would leave the disk with less than `MinFree` (10 GB unless set) fails with `modelstore.ErrNoRoom`; with a `MaxSize`, the units least recently used are removed first to make room. `s.Usage()`, `s.PlanPrune(o)`, `s.Prune(o)` and `s.Remove(units)` are what `hwnpu cache` does

```go
s, err := modelstore.Open(modelstore.Options{Limits: &modelstore.Limits{MaxSize: 100e9, MinFree: 20e9}})
u, err := s.Usage()                                                     // u.Units: kind, bytes, last used
gone, err := s.Prune(modelstore.PruneOptions{OlderThan: 30 * 24 * time.Hour}) // not used for 30 days
```

## Apps that combine models (ai)

`github.com/shibukawa/hwnpu/ai` is a set of packages for an app to register the models it uses, download and compile them ahead of time, and run them on the NPU (v0: its shape will still change; how to use it and the open design questions are in [docs/developer-guide.md](docs/developer-guide.md), the plan in [docs/app-api.md](docs/app-api.md), in Japanese). It prepares models the way hwnpu-chat, hwnpu-image and hwnpu-audio do on their first run, in the same cache, so models prepared by the commands are used as they are.

```go
// Register the models the app uses. Names are models, families or uses of models.yaml (hwnpu-chat list)
var (
	stt = speech.Transcribe("transcribe")
	tr  = text.Translate("translate").
		At(16, "plamo3translate:2b"). // the 2B on machines of 16 GB and up
		At(32, "plamo3translate:8b")  // the 8B on machines of 32 GB and up
	tts = speech.Speak("speech")

	app = ai.NewApp("interpreter", stt, tr, tts)
)

// Prepare: look at the plan (it downloads nothing), then download, convert and compile for the NPU
plan, err := app.Plan(ctx)
fmt.Println(plan.DownloadBytes(), plan.MemoryGB(), plan.Licenses())
err = plan.Run(ctx, func(p ai.Progress) { fmt.Println(p.Use, p.Stage, p.Message) })

// Run: no network, no conversion. A model that is not prepared is ai.ErrNotPrepared
rt, err := app.Open(ctx)
defer rt.Close()
t, err := stt.Get(ctx, rt) // *speech.Transcriber
res, err := t.Transcribe(ctx, pcm16k, speech.TranscribeOptions{})
```

| Package | What it has |
|---|---|
| `ai` | `App` (registration), `Use` (a typed handle: `Get`, `Release`, and the line-up by the machine's memory, `At` and `As`), `Plan` (planning and preparing), `Runtime` |
| `ai/speech` | Speech to text (Whisper), text to speech (LFM3-Audio), a VAD by loudness, reading and writing WAV |
| `ai/text` | Translation (PLaMo 3 Translate), chat |
| `ai/flow` | Passing data between models: `Stream`, a stream of chunks, and its tools |
| `ai/aicli` | kong commands to put in the app's command line: `models` (the list, `download`, `prepare`), `status` |

- **A line-up by the machine's memory**: a use's name picks the largest model the machine runs when it runs alone. An app that runs several at once can keep its main model large and the others small with `At(gb, name)` (another model on machines of at least that memory) and `As(gb)` (pick as a machine of that memory would). `APP models` shows the models picked, what they take running together, and the machine's memory
- **One at a time**: `use.Release(rt)` unloads a model, so that large models used one after another (a picture's generator, then its depth) are not in memory together
- **Preparing on the first run**: `app.Open(ctx, ai.WithAutoPrepare(fn))` prepares a model that is not ready at its first `Get`, as the commands do. That may download gigabytes
- **Licenses**: a model whose license asks to be accepted fails with `*ai.LicenseError` unless `plan.AcceptLicenses` is set. Show `plan.Licenses()` and get the user's agreement before setting it

### Joining models (ai/flow)

Every model has a batch form (one value in, one out) and a stream form (`...Stream`). A stream is an `iter.Seq2`, read with `range`; it stops at a `break`, an error, or the cancel of its context. Stages are joined in three ways, by when the next one may start.

| Joined | Tools |
|---|---|
| When the one before has finished (a picture, then its depth) | The batch calls, `flow.Last`, `flow.Of` |
| As the chunks come (translate and speak each sentence, follow a camera) | `flow.Async` (stages overlap), `flow.Latest` (drop stale chunks, keep the newest), `flow.Hold` (keep the newest at hand), `flow.Tee` |
| On the beat (audio in time for a player, frames for a screen) | `flow.Metronome` (a tick a beat), `flow.Pace` (hand a chunk over at its time), `flow.Live` (a recording at a microphone's pace) |

```go
utterances := speech.NewVAD(speech.VADOptions{}).Detect(ctx, mic) // mic: flow.Stream[[]float32] at 16 kHz
texts := t.TranscribeStream(ctx, utterances, speech.TranscribeOptions{})
en := translator.TranslateStream(ctx, texts, text.TranslateOptions{To: "en"})
audio := speaker.SpeakStream(ctx, flow.Async(ctx, en, 4), speech.SpeakOptions{})
for c, err := range audio { ... } // c.Data: audio at 24 kHz
```

### The app's command line (ai/aicli)

```go
var cli struct {
	Run    RunCmd       `cmd:"" default:"withargs" help:"..."`
	Models aicli.Models `cmd:"" help:"The models the app uses: list, download, prepare."`
	Status aicli.Status `cmd:"" help:"This machine's NPU and what uses it."`
}
k := kong.Parse(&cli, aicli.Bind(ctx, app))
k.FatalIfErrorf(k.Run())
```

`APP models` lists the models and what they take, `APP models download` downloads and converts them (no NPU is needed), and `APP models prepare` also loads each once to compile it for the NPU. `APP status` is `hwnpu status`. A working example is [samples/interpreter](samples/interpreter), which interprets recorded speech into speech in another language.

- So far it has the models of speech and text. Those of pictures (depth, faces, generation and the others) are only in the commands yet
- It was run on AMD's NPU (Ryzen AI 9 365) only. On a Mac and on Intel the commands do the same preparing and loading, but they were not run through `ai`

## AMD (running kernels directly)

AMD's NPU runs kernels built with IRON / mlir-aie (an xclbin and an instruction stream), not models.

```go
kr, err := dev.Kernel() // ErrUnsupported on anything but AMD
prog, err := kr.Load(ctx, hwnpu.ProgramSource{Path: "final.xclbin"})
k, err := prog.Kernel("MLIR_AIE")

instr, _ := dev.AllocBuffer(len(insts), hwnpu.WithBufferKind(hwnpu.BufferInstructions))
in, _ := dev.AllocBuffer(n)
out, _ := dev.AllocBuffer(n)
copy(instr.Bytes(), insts)
instr.SyncToDevice(0, len(insts)) // AMD's NPU is not cache coherent
in.SyncToDevice(0, n)

run, _ := k.NewRun()
err = run.SetUint64(0, 3).SetBuffer(1, instr).SetUint32(2, uint32(len(insts)/4)).
	SetBuffer(3, in).SetBuffer(4, tmp).SetBuffer(5, out).
	Run(ctx)
out.SyncFromDevice(0, n)
```

- The argument indices, offsets and types are read from the xclbin's metadata (`k.Args()`)
- It needs read and write access to `/dev/accel/accelN` (the `render` group). It assumes a Linux kernel of 6.14 or later (amdxdna)
- The power policy sets the priority (QoS) of the hardware context. The power mode of the whole device is root only, so it is not changed
- Power is read from the NPU's sensor (an instantaneous value), or when there is none, from the power of the whole package the amdgpu driver reports (PPT, no root needed). From the first `PowerSample` it is read every 50 ms and integrated into energy. The NPU sensor exists only in AMD's DKMS driver; with the stock Linux 7.0 kernel the value is the whole package's (`Scope` is `package`)

On Strix Point and later (XDNA2), the full ELF IRON builds (`design.elf`, which runs a whole graph in one hardware context and one command) can be loaded too. The kernel name has the form `main:sequence`, and all arguments are buffers. Values for each call are written to `run.Scratchpad()`.

## Notes for Apple

- Core ML cannot be told "the Neural Engine only". hwnpu allows only the CPU and the Neural Engine, and checks with the estimate of `MLComputePlan` (macOS 14.4 or later) that every op is placed on the Neural Engine. It is an estimate, so it does not guarantee the placement at run time
- The Neural Engine assumes fp16. fp32 models and models that are too small are placed on the CPU. Convert to an fp16 ML Program with coremltools
- A model that is too small can be put on the Neural Engine by padding it along the batch with `tools/coreml/ane_pad.py`. Small models can get slower, though, so read [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md) before using it
- `hwnpu-chat` and `hwnpu-image` run on Core AI alone on a Mac (macOS 27 is needed; Core ML is not used, and `hwnpu-image --cpu` runs a Core ML export on the CPU interpreter for checking). On macOS 27 the other commands' converted models also run on Core AI (`--backend coreai` or `HWNPU_BACKEND=coreai`; Core ML is the default). The results and speed are the same; the difference is that Core AI keeps the compiled results until the disk runs low (Core ML compiles again on later runs once they go over about 4 GiB). The commands it works with and the costs are in section 5 of [INSTALL_APPLE.md](INSTALL_APPLE.md)

## Notes for Intel

- It needs the Intel NPU driver (`intel-level-zero-npu`, `intel-driver-compiler-npu`, `intel-fw-npu`) and the Level Zero loader (`libze1`). The loader's location can also be given with the environment variable `HWNPU_ZE_LOADER`
- On Linux it needs read and write access to `/dev/accel/accel0` (the `render` group)
- `GraphSource{Path: "model.xml"}` passes OpenVINO IR (the `.bin` next to it is read too), and `GraphSource{Blob: ...}` a compiled blob. Extra options for the compiler go in `BuildFlags` in the form `--config KEY="VALUE"`
- A model with ops the NPU compiler does not support fails with `ErrNotOnNPU` whatever the placement policy, and the ops at fault and the build log are returned
- Converted to OpenVINO IR for Intel's NPU are hwnpu-chat's models (LFM3 and its VL, Audio and d1, Gemma 4, Qwen3.5, Qwen3, PLaMo 3, PLaMo-Embedding), Anima, `hwnpu-audio`'s transcribe (Whisper), tts, generate (Magenta RealTime 2, MOSS-SoundEffect), pitch and notes, and `hwnpu-image`'s depth (DA3, MoGe-2), face-detect (YuNet), segment (EfficientSAM's encoder), detect, pose and ocr. Chroma, FLUX.2 klein and SigLIP 2 (label) are on AMD's NPU and Apple's Neural Engine only (not Intel yet). Only LFM3, PLaMo 3 Translate, PLaMo-Embedding and Anima have run on the hardware; the others are checked from conversion to results on OpenVINO's CPU (`HWNPU_BACKEND=openvino`, "Telling problems apart with OpenVINO" in [INSTALL_INTEL.md](INSTALL_INTEL.md)) ([docs/design.md](docs/design.md) 16.166)

## Diagnostics

```bash
go run ./cmd/hwnpu-dev info -model internal/fixtures/coreml/linear_relu_fp16.mlpackage
```

## Tests

```bash
CGO_ENABLED=0 go test ./...
```

- With a C compiler (`cc`), the tests build a fake Level Zero driver (`testdata/zemock`) and check the Intel backend
- The test models are in `internal/fixtures` and can be regenerated with `gen_models.py` (it needs numpy, and coremltools for the Core ML ones)
- The reference values of the Level Zero structs can be regenerated with `testdata/abi/levelzero_abi.c`, and those of amdxdna, xclbin and ERT with `testdata/abi/xdna_abi.c`
- The AMD backend is tested with a fake driver written in Go (`internal/xdna/xdnasim`) that mimics the behavior of Linux's amdxdna driver
