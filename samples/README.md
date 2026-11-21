# Samples of the ai package

[日本語](README.ja.md)

Small apps built on hwnpu's [`ai`](../USAGE_LIBRARY.md) package: each declares the models it uses, prepares them for this machine's NPU, and joins them with the program's own steps. They run on AMD's NPU (Ryzen AI) and on Apple's Neural Engine (macOS 27, Core AI).

| Sample | What it does | Models | Download (Apple / AMD) |
|---|---|---|---|
| [interpreter](interpreter/README.md) | Interprets speech, from the microphone or a recording, into speech in another language | Whisper, PLaMo 3 Translate, LFM3-Audio TTS | 11.4 GB / 6.6 GB |
| [parallax](parallax/README.md) | Shows a picture as if it were deep, turning with your head before the camera | Depth Anything 3, YuNet, FLUX.2 klein | 16.5 GB / 14.9 GB |
| [layers](layers/README.md) | Parts a picture into layers by depth, draws what each layer hid, and writes a Photoshop document | Depth Anything 3, FLUX.2 klein | 16.5 GB / 14.9 GB |
| [backdrop](backdrop/README.md) | Blurs or replaces what is behind you in the camera's picture, as a video call does | MediaPipe Selfie Segmentation (on the CPU) | 0.5 MB |

The downloads are for a machine of 16 GB (the models are picked by the machine's memory) and are made once: the models stay in hwnpu's cache, shared by the samples and the commands. `go run ./SAMPLE models` prints what this machine would download. parallax and layers share their models: running both downloads no more than one.

## Running one

The samples are a Go module of their own, built against the hwnpu of this checkout (`replace` in [go.mod](go.mod)). Run them from this directory:

```bash
cd samples
go run ./interpreter models            # the models, what they take to download and to run, and the network
go run ./interpreter models prepare    # download, convert and compile them ahead (or pass --prepare to a run)
go run ./interpreter talk.wav -o out.wav
```

Every sample has the same `models` (`list`, `download`, `prepare`) and `status` commands. The models are kept in hwnpu's cache and shared with the commands (`hwnpu-chat`, `hwnpu-image`, `hwnpu-audio`): what one prepared, the others use.

- The first run of a model downloads it and compiles it for the NPU: minutes, on Apple up to an hour for FLUX.2 klein's edit programs (prebuilt programs are to take this off). Later runs start in seconds.
- On Apple, Core AI keeps the compiled programs per process name: `go run` and a binary you built are compiled for separately. Build once (`go build -o bin/ ./...`) and run the binaries.
- The models are picked for this machine's memory; `models` shows what they take running together.

## What they need

- An NPU hwnpu runs: AMD Ryzen AI (Linux) or a Mac with Apple Silicon on macOS 27. See [INSTALL_AMD.md](../INSTALL_AMD.md) and [INSTALL_APPLE.md](../INSTALL_APPLE.md).
- The microphone and the camera come from [hwcamera](https://github.com/shibukawa/hwcamera), without cgo. On a Mac the terminal app needs permission for them (System Settings › Privacy & Security); the first use asks.
- parallax's and backdrop's windows are [Ebitengine](https://ebitengine.org). On Linux it needs the X11/OpenGL libraries Ebitengine builds with; without a display, build with `-tags headless` (parallax's `render` and `track` still work).
- backdrop's model runs on the CPU with ONNX Runtime's shared library (the system's, or the one `HWNPU_ONNXRUNTIME` names), as YuNet does on an AMD machine. It was run on the AMD machine only.

## Layout

| Directory | What it is |
|---|---|
| `interpreter/`, `parallax/`, `layers/`, `backdrop/` | The samples (each a `main` package) |
| `internal/capture` | The microphone and the camera as streams of `ai/flow` (hwcamera) |
| `internal/layering` | Parting a picture into layers by depth, shared by layers and parallax |

hwcamera and Ebitengine are dependencies of this module only, not of the hwnpu library.
