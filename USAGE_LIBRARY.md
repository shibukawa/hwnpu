# Using BonsaiAI as a Go library (planned)

[日本語](USAGE_LIBRARY.ja.md)

> **Planned**: the Go library API is not settled yet. Names and arguments may change without notice.

How to use BonsaiAI's Go API, and notes for each NPU. For the commands (bonsaiai-chat, bonsaiai-image, bonsaiai-audio, bonsaiai-dev), start from [README.md](README.md).

## Basics

```go
dev, err := bonsaiai.Open(ctx)
if err != nil { ... }
defer dev.Close()

gr, _ := dev.Graph()
g, err := gr.Compile(ctx, bonsaiai.GraphSource{Path: "model.mlpackage"})
if errors.Is(err, bonsaiai.ErrNotOnNPU) {
	// err is a *bonsaiai.PlacementError. The ops placed off the NPU are in Report.Ops
}
defer g.Close()

in, _ := g.Arg("x")
out, _ := g.Arg("y")
x, _ := dev.AllocBuffer(in.Bytes)
y, _ := dev.AllocBuffer(out.Bytes)
bonsaiai.EncodeF16(x.Bytes(), input)

req, _ := g.NewRequest()
req.BindName("x", x)
req.BindName("y", y)
err = req.Run(ctx)
bonsaiai.DecodeF16(result, y.Bytes())
```

## Power and cache

```go
dev, err := bonsaiai.Open(ctx,
	bonsaiai.WithPowerPolicy(bonsaiai.PowerLow), // Intel: low priority + background
	bonsaiai.WithCacheDir(cacheDir),       // save compiled results and reuse them
)

before, err := dev.PowerSample() // ErrUnsupported on devices whose energy cannot be read
// ... inference ...
after, _ := dev.PowerSample()
fmt.Println(bonsaiai.AveragePower(before, after), "W", after.Scope) // "package" is the whole CPU package
```

- The power policy works only on devices with `CapPowerControl`. Core ML has no power setting
- On Intel, energy is read from RAPL on Linux. It is the value of the whole CPU package, not of the NPU alone, and usually needs root
- The cache key holds the model, the build flags and the versions of the driver and compiler (macOS on Apple), so a change in any of them rebuilds it

## Models' files (modelstore)

`github.com/shibukawa/bonsaiai/modelstore` finds and fetches the files of the models as BonsaiAI's commands do: this machine's cache, then the sources (`BONSAIAI_MODEL_SOURCE`: folders, WebDAV and HTTP, S3), then the models the application installs with it, then Hugging Face (docs/cache-sharing-design.md 7).

```go
app := configdir.New("", "myapp") // github.com/shibukawa/configdir v1.1.0
s, err := modelstore.Open(modelstore.Options{
	Bundle: &app, // the installer's models/ in the app's resource folder (nil: none)
	// Root: "" (BONSAIAI_CACHE, else bonsaiai in the user cache directory), Sources: nil (the environment's)
})
config, err := s.File(ctx, "LiquidAI/LFM3-1.7B", "config.json") // a local path
weights, err := s.Weights(ctx, "LiquidAI/LFM3-1.7B")            // model.safetensors (or its shards)
```

- The settings are the process's: `Open` sets them for everything in it
- `Options.Files` are definition files of the app's own (models.yaml's format: models, uses, aliases), read over BonsaiAI's, the machine's and the user's
- The bundled models are in the layout of a source (`original/OWNER/NAME/` with `bonsaiai-lock.json`) and are used in place (linked, not copied)
- `Options.Limits` (nil: `BONSAIAI_CACHE_MAX_SIZE`, `BONSAIAI_CACHE_MIN_FREE`) bounds the cache: a download that would leave the disk with less than `MinFree` (10 GB unless set) fails with `modelstore.ErrNoRoom`; with a `MaxSize`, the units least recently used are removed first to make room. `s.Usage()`, `s.PlanPrune(o)`, `s.Prune(o)` and `s.Remove(units)` are what `bonsaiai cache` does

```go
s, err := modelstore.Open(modelstore.Options{Limits: &modelstore.Limits{MaxSize: 100e9, MinFree: 20e9}})
u, err := s.Usage()                                                     // u.Units: kind, bytes, last used
gone, err := s.Prune(modelstore.PruneOptions{OlderThan: 30 * 24 * time.Hour}) // not used for 30 days
```

## Apps that combine models (ai)

`github.com/shibukawa/bonsaiai/ai` is a set of packages for an app to register the models it uses, download and compile them ahead of time, and run them on the NPU (v0: its shape will still change; how to use it and the open design questions are in [docs/developer-guide.md](docs/developer-guide.md), the plan in [docs/app-api.md](docs/app-api.md), in Japanese). It prepares models the way bonsaiai-chat, bonsaiai-image and bonsaiai-audio do on their first run, in the same cache, so models prepared by the commands are used as they are.

```go
// Register the models the app uses. Names are models, families or uses of models.yaml (bonsaiai-chat list)
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
| `ai/vision` | The depth of a picture (Depth Anything 3, MoGe-2), face detection (YuNet), and where a head is before the camera, from its face |
| `ai/imagegen` | Drawing and editing pictures (FLUX.2 klein): from a prompt, from a prompt and reference pictures |
| `ai/flow` | Passing data between models: `Stream`, a stream of chunks, and its tools |
| `ai/cli` | Commands for the app's command line: `models` (the list, `download`, `prepare`), `status`. It depends on no command line library (`cli.List`, `cli.Run(ctx, app, args)` and the others) |
| `ai/cli/konghelper` | The adapter of those commands for kong |

- **A line-up by the machine's memory**: a use's name picks the largest model the machine runs when it runs alone. An app that runs several at once can keep its main model large and the others small with `At(gb, name)` (another model on machines of at least that memory) and `As(gb)` (pick as a machine of that memory would). `APP models` shows the models picked, what they take running together, and the machine's memory
- **One at a time**: `use.Release(rt)` unloads a model, so that large models used one after another (a picture's generator, then its depth) are not in memory together. `app.InTurns(depth, artist)` says so to the app: its plan (`APP models`) counts the larger alone, and loading one releases the other
- **Prepare first, then run**: the Runtime runs only models prepared and compiled for the program (`APP models prepare`, `plan.Run`); `konghelper.Open(ctx, app)` fails before a command starts, naming what is missing. Nothing downloads or compiles while the app runs
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

The joins can also be written as a structure (a network): nodes of models and nodes of the program joined with `To` and given to the app, which loads the models and runs it. What runs once for a request is written with `ai.NewCall(...).Then(...)` and `Do`. See sections 6.6 and 6.7 of [docs/developer-guide.md](docs/developer-guide.md).

```go
mic := flow.Source[[]float32]("mic").From(stream)
hear := speech.TranscribeNode(stt, speech.TranscribeOptions{})
trans := text.TranslateNode(tr, text.TranslateOptions{To: "en"})
log := flow.Tap("heard", func(c flow.Chunk[flow.Text]) { fmt.Println(c.Data.Text) }) // a node of the program
out := flow.Sink[flow.Text]("out").Do(show)

app := ai.NewApp("subtitles") // its models are taken from the network
app.From(mic).To(speech.VADNode(speech.VADOptions{})).To(hear).To(log).To(trans).To(out)
err := app.Run(ctx)
```

### The app's command line (ai/cli)

The commands are functions of `ai/cli`, which depends on no command line library. Call `cli.List(ctx, app, cli.ListOptions{})` and the others from commands of your own, or hand the arguments over to `cli.Run(ctx, app, os.Args[1:])`. For an app that uses kong there is the adapter `ai/cli/konghelper`.

```go
var cmds struct {
	Run    RunCmd            `cmd:"" default:"withargs" help:"..."`
	Models konghelper.Models `cmd:"" help:"The models the app uses: list, download, prepare."`
	Status konghelper.Status `cmd:"" help:"This machine's NPU and what uses it."`
}
k := kong.Parse(&cmds, konghelper.Bind(ctx, app))
k.FatalIfErrorf(k.Run())
```

`APP models` lists the models and what they take (with what they take together and what this machine has for models), `APP models download` downloads and converts them (no NPU is needed), and `APP models prepare` also loads each once to compile it for the NPU. `APP status` is `bonsaiai status`. The working examples are [samples/interpreter](samples/interpreter), which interprets speech, from the microphone or a recording, into speech in another language, and [samples/parallax](samples/parallax), which estimates the depth of a picture, follows the viewer's head with the camera, and draws the picture as if it had depth (its window is Ebitengine), and [samples/layers](samples/layers), which parts a picture into layers by depth, draws what each layer hid, and writes a PSD. The samples are a module of their own (`samples/go.mod`) and take the microphone and the camera from [hwcamera](https://github.com/shibukawa/hwcamera), which is not a dependency of the library.

- So far it has the models of speech and text, the depth and the faces of pictures, and drawing and editing pictures (FLUX.2 klein). The other picture models (Anima), object and pose detection and the others are only in the commands yet
- It was run on AMD's NPU (Ryzen AI 9 365) and on a Mac (M3, Core AI): the three samples, the camera and the window. On Intel the commands do the same preparing and loading, but they were not run through `ai`
- On Apple's Neural Engine FLUX.2 klein draws from reference pictures at 1024 x 1024 (one of 1024, and none, one or two of 512) and 512 x 512 (one or two of 512), in the same export as drawing from a prompt (`Generator.EditSizes`)

## AMD (running kernels directly)

AMD's NPU runs kernels built with IRON / mlir-aie (an xclbin and an instruction stream), not models.

```go
kr, err := dev.Kernel() // ErrUnsupported on anything but AMD
prog, err := kr.Load(ctx, bonsaiai.ProgramSource{Path: "final.xclbin"})
k, err := prog.Kernel("MLIR_AIE")

instr, _ := dev.AllocBuffer(len(insts), bonsaiai.WithBufferKind(bonsaiai.BufferInstructions))
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

- Core ML cannot be told "the Neural Engine only". BonsaiAI allows only the CPU and the Neural Engine, and checks with the estimate of `MLComputePlan` (macOS 14.4 or later) that every op is placed on the Neural Engine. It is an estimate, so it does not guarantee the placement at run time
- The Neural Engine assumes fp16. fp32 models and models that are too small are placed on the CPU. Convert to an fp16 ML Program with coremltools
- A model that is too small can be put on the Neural Engine by padding it along the batch with `tools/coreml/ane_pad.py`. Small models can get slower, though, so read [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md) before using it
- `bonsaiai-chat` and `bonsaiai-image` run on Core AI alone on a Mac (macOS 27 is needed; Core ML is not used, and `bonsaiai-image --cpu` runs a Core ML export on the CPU interpreter for checking). On macOS 27 the other commands' converted models also run on Core AI (`--backend coreai` or `BONSAIAI_BACKEND=coreai`; Core ML is the default). The results and speed are the same; the difference is that Core AI keeps the compiled results until the disk runs low (Core ML compiles again on later runs once they go over about 4 GiB). The commands it works with and the costs are in section 5 of [INSTALL_APPLE.md](INSTALL_APPLE.md)

## Notes for Intel

- It needs the Intel NPU driver (`intel-level-zero-npu`, `intel-driver-compiler-npu`, `intel-fw-npu`) and the Level Zero loader (`libze1`). The loader's location can also be given with the environment variable `BONSAIAI_ZE_LOADER`
- On Linux it needs read and write access to `/dev/accel/accel0` (the `render` group)
- `GraphSource{Path: "model.xml"}` passes OpenVINO IR (the `.bin` next to it is read too), and `GraphSource{Blob: ...}` a compiled blob. Extra options for the compiler go in `BuildFlags` in the form `--config KEY="VALUE"`
- A model with ops the NPU compiler does not support fails with `ErrNotOnNPU` whatever the placement policy, and the ops at fault and the build log are returned
- Converted to OpenVINO IR for Intel's NPU are bonsaiai-chat's models (LFM3 and its VL, Audio and d1, Gemma 4, Qwen3.5, Qwen3, PLaMo 3, PLaMo-Embedding), Anima, `bonsaiai-audio`'s transcribe (Whisper), tts, generate (Magenta RealTime 2, MOSS-SoundEffect), pitch and notes, and `bonsaiai-image`'s depth (DA3, MoGe-2), face-detect (YuNet), segment (EfficientSAM's encoder), detect, pose and ocr. FLUX.2 klein and SigLIP 2 (label) are on AMD's NPU and Apple's Neural Engine only (not Intel yet). Only LFM3, PLaMo 3 Translate, PLaMo-Embedding and Anima have run on the hardware; the others are checked from conversion to results on OpenVINO's CPU (`BONSAIAI_BACKEND=openvino`, "Telling problems apart with OpenVINO" in [INSTALL_INTEL.md](INSTALL_INTEL.md)) ([docs/design/intel.md](docs/design/intel.md))

## Diagnostics

```bash
go run ./cmd/bonsaiai-dev info -model internal/fixtures/coreml/linear_relu_fp16.mlpackage
```

## Tests

```bash
CGO_ENABLED=0 go test ./...
```

- With a C compiler (`cc`), the tests build a fake Level Zero driver (`testdata/zemock`) and check the Intel backend
- The test models are in `internal/fixtures` and can be regenerated with `gen_models.py` (it needs numpy, and coremltools for the Core ML ones)
- The reference values of the Level Zero structs can be regenerated with `testdata/abi/levelzero_abi.c`, and those of amdxdna, xclbin and ERT with `testdata/abi/xdna_abi.c`
- The AMD backend is tested with a fake driver written in Go (`internal/xdna/xdnasim`) that mimics the behavior of Linux's amdxdna driver
