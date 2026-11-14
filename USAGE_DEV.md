# Developer commands (hwnpu-dev)

[日本語](USAGE_DEV.ja.md)

Commands to check that things work, measure, and convert and check models.

## Check report

It checks whether hwnpu runs on the NPU at hand and writes the result as Markdown that can be pasted as is. The test models are embedded in the program, so it runs from anywhere in the repository.

```bash
go run ./cmd/hwnpu-dev report -o report.md
```

- For each device found, it compiles the models for that backend in strict mode, compares the results with fp32 references and measures the run time
- It also shows how loading the Level Zero loader went, which tells why an Intel NPU is not found
- `-json report.json` writes JSON too. `-n` sets the number of measurement runs, `-power low` and the like the power policy, and `-cache DIR` the compile cache
- On AMD, `-xclbin final.xclbin -insts insts.bin` runs a design built from mlir-aie's `programming_examples/basic/passthrough_dmas` and checks its output (the arguments in order: opcode, instr, ninstr, input, unused, output)
- It exits with code 2 when something fails, and 1 when no NPU is found

On macOS, `go run ./cmd/hwnpu-dev powerbench` measures the energy of one call of a Core ML model with powermetrics. It asks for the sudo password once at the start. See [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md) for details.

## Converting and checking models

`go run ./cmd/hwnpu-dev convert` converts Hugging Face checkpoints (safetensors or GGUF) to `.mlpackage` for the Neural Engine without Python. It supports the following. The steps for porting are in `.claude/skills/port-architecture`, and how to choose source models and the Neural Engine's limits are in [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md).

- LFM3 (including judge models such as d1-2B), LFM3-VL's vision encoder, LFM3-Audio / LFM3-Omni's audio encoder and audio output
- Gemma 4 (gemma-4-E2B-it)
- Qwen3.5 (the text part of Qwen3.5-4B and others; checked on the CPU only), Qwen3 (Qwen3-0.6B and 1.7B; `hwnpu-chat -m qwen3`)
- BERT style encoders, PLaMo-Embedding-1B
- PLaMo 3 (PLaMo 3 Translate 2B; checked on the CPU and OpenVINO's CPU only; `hwnpu-chat -m plamo-3-translate-2b` translates)
- Whisper (whisper-large-v3-turbo; the audio encoder is `-arch whisper-encoder`, the cross attention keys and values `-arch whisper-cross`)

Other commands:

- `hwnpu-dev check`: runs a converted graph on the CPU in fp32 and checks it layer by layer against a reference from the Hugging Face implementation (`tools/ref/ref_logits.py`). No Mac is needed
- `hwnpu-dev generate`: generates text with a converted causal LM (the tokenizer is Go too). Converted with `-cache N` it has a KV cache and can write long texts (LFM3-1.7B on an M3 with a cache of 1024: 20.6 ms a token)
- `hwnpu-dev formats`: compares the accuracy and speed of each weight format (int8, int4-g128, int4) on the NPU

## Measuring the NPU (probe)

`go run ./cmd/hwnpu-dev probe` measures the NPU with models it generates. It builds in Go a model of the same shape as an LLM generating one token, and first measures the speed of each format (fp16, int8, lut4, int4 on Apple; fp16, int8, int4 on Intel) with 0.5B and 1B weights (`bench`). Then it grows the model and narrows down by bisection where it falls off the NPU, loads several models just under that point at once, and checks whether the limit is per model or in total (`limit`). It needs no Python, no model files and no root, and runs without cloning with the one line below. The results are written to `npuprobe-<machine>-<time>.md`.

```bash
go run github.com/shibukawa/hwnpu/cmd/hwnpu-dev@latest probe
```

With `bench` or `limit` it does only one of them. To also measure the energy per token on Apple, use `go run ./cmd/hwnpu-dev powerbench -decode 8,16 -gpu` (it needs sudo). The steps for each machine, how to read the results and the results so far are in [PROBE.md](PROBE.md).

To try it on a Windows machine without Go, take along an executable built on a machine with Go.

```bash
GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build -o hwnpu-dev.exe ./cmd/hwnpu-dev
```

## Diagnostics

```bash
go run ./cmd/hwnpu-dev info -model internal/fixtures/coreml/linear_relu_fp16.mlpackage
```
