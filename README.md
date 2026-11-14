# hwnpu

[日本語](README.ja.md)

Commands, written in Go without cgo, that run the NPU built into the CPU (a Go library is planned). The aim is to run small models on little power.

- In the default strict mode, a model that does not run on the NPU alone is an error before it runs. Nothing falls back to the CPU or the GPU silently
- The only dependencies are [purego](https://github.com/ebitengine/purego) and golang.org/x/sys. The user's machine needs no Python

| Platform | Backend | Status |
|---|---|---|
| macOS (Apple Silicon) | Core ML (and Core AI on macOS 27) | Done. Checked on M1 and M3 |
| Linux / Windows (Intel Core Ultra) | Level Zero (OpenVINO IR) | Done. Only some models run on the hardware so far |
| Linux (AMD Ryzen AI) | amdxdna ioctl (exports from IRON / mlir-aie) | Done. Checked on a Ryzen AI 9 365 (Strix Point) |
| Windows (AMD Ryzen AI) | XRT (xrtshim) | Done. Untested on Windows |
| WSL2 | none | Not supported (the NPU is not visible) |

## Try it

```bash
go run ./cmd/hwnpu-chat "What is the capital of Japan?"   # the first run downloads the model and converts it for the NPU
go run ./cmd/hwnpu-chat list                               # the models, and which one this machine picks
go run ./cmd/hwnpu-dev report -o report.md                 # check that hwnpu runs this machine's NPU
```

Setup: [Apple](INSTALL_APPLE.md), [Intel](INSTALL_INTEL.md), [AMD](INSTALL_AMD.md).

## Commands

| Command | What it does | Guide |
|---|---|---|
| `hwnpu-chat` | Chat, translation, a coding agent (`agent`), an OpenAI compatible server (`serve`), embedding search (`search`), typed judgments (`systemone`) | [CHAT.md](CHAT.md); AMD: [USAGE_LLM_AMD.md](USAGE_LLM_AMD.md) |
| `hwnpu-image` | Image generation (Anima, Sana, Chroma1), depth, faces, objects, poses, text (OCR), labels, cutouts | [USAGE_IMAGE.md](USAGE_IMAGE.md) |
| `hwnpu-audio` | Music and sound effects, speech, transcription, pitch, notes | [USAGE_AUDIO.md](USAGE_AUDIO.md) |
| `hwnpu-dev` | Checks, NPU measurements (`probe`), model conversion and comparison | [USAGE_DEV.md](USAGE_DEV.md), [PROBE.md](PROBE.md) |

`-m` takes a model's name (`-m gemma-4-E2B-it`), a family (`-m gemma4`) or a use (`-m translate`); a family or a use picks the model that fits the machine's memory. The models come from the embedded [models.yaml](models.yaml), and a file of your own adds or replaces them ([docs/model-table.md](docs/model-table.md)).

## As a library (planned)

The Go library API is planned and not settled yet. Its current shape is in [USAGE_LIBRARY.md](USAGE_LIBRARY.md).

## Documents

- [MODEL.md](MODEL.md): the recommended model for each use and amount of memory
- [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md): the data types each NPU computes in, and how models are prepared
- [CLOUD_AI_COMPATIBILITY.md](CLOUD_AI_COMPATIBILITY.md): which features of cloud AI APIs hwnpu can take over
- [docs/design.md](docs/design.md): design notes and measurements (Japanese)

## Tests

```bash
CGO_ENABLED=0 go test ./...
```

## License

[Apache-2.0](LICENSE). Model weights follow their own licenses.
