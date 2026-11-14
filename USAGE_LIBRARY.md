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
- `hwnpu-image` runs on Core AI alone on a Mac (macOS 27 is needed; Core ML is not used, and `--cpu` runs a Core ML export on the CPU interpreter for checking). On macOS 27 the other commands' converted models also run on Core AI (`--backend coreai` or `HWNPU_BACKEND=coreai`; Core ML is the default). The results and speed are the same; the difference is that Core AI keeps the compiled results until the disk runs low (Core ML compiles again on later runs once they go over about 4 GiB). The commands it works with and the costs are in section 5 of [INSTALL_APPLE.md](INSTALL_APPLE.md)

## Notes for Intel

- It needs the Intel NPU driver (`intel-level-zero-npu`, `intel-driver-compiler-npu`, `intel-fw-npu`) and the Level Zero loader (`libze1`). The loader's location can also be given with the environment variable `HWNPU_ZE_LOADER`
- On Linux it needs read and write access to `/dev/accel/accel0` (the `render` group)
- `GraphSource{Path: "model.xml"}` passes OpenVINO IR (the `.bin` next to it is read too), and `GraphSource{Blob: ...}` a compiled blob. Extra options for the compiler go in `BuildFlags` in the form `--config KEY="VALUE"`
- A model with ops the NPU compiler does not support fails with `ErrNotOnNPU` whatever the placement policy, and the ops at fault and the build log are returned
- Converted to OpenVINO IR for Intel's NPU are hwnpu-chat's models (LFM3 and its VL, Audio and d1, Gemma 4, Qwen3.5, Qwen3, PLaMo 3, PLaMo-Embedding), Anima, Sana, `hwnpu-audio`'s transcribe (Whisper), tts, generate (Magenta RealTime 2, MOSS-SoundEffect), pitch and notes, and `hwnpu-image`'s depth (DA3, MoGe-2), face-detect (YuNet), segment (EfficientSAM's encoder), detect, pose and ocr. Chroma, SigLIP 2 (label) and Qwen-Image are on AMD's NPU and Apple's Neural Engine only (not Intel yet). Only LFM3, PLaMo 3 Translate, PLaMo-Embedding and Anima have run on the hardware; the others are checked from conversion to results on OpenVINO's CPU (`HWNPU_BACKEND=openvino`, "Telling problems apart with OpenVINO" in [INSTALL_INTEL.md](INSTALL_INTEL.md)) ([docs/design.md](docs/design.md) 16.166)

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
