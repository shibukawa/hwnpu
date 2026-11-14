# Running LLMs on the AMD NPU

[日本語](USAGE_LLM_AMD.ja.md)

How to run Gemma 4 E2B / E4B, LFM3 and its multimodal models, PLaMo 3 Translate and Qwen3.5 on the AMD NPU. How to use the commands themselves (chat, server, agent) is in [CHAT.md](CHAT.md), and how to set up IRON is in [INSTALL_AMD.md](INSTALL_AMD.md).

## Export and run

IRON (amd/IRON) is used once as a compiler on the developer's machine. `tools/amd/iron_export_lm.py` writes an export, and hwnpu runs it with Go alone (the user's machine needs neither Python nor XRT).

```bash
python tools/amd/iron_export_lm.py gemma4 ~/models/gemma-4-E2B-it ~/models/gemma-4-E2B-it.amd-npu2 --max-seq-len 4096 --prefill-chunk 1024
go run ./cmd/hwnpu-chat -m ~/models/gemma-4-E2B-it.amd-npu2
go run ./cmd/hwnpu-chat agent -m ~/models/gemma-4-E2B-it.amd-npu2 "Summarize README.md"
```

When there is no Apple or Intel NPU but there is an AMD NPU, `hwnpu-chat` (`chat`, `agent`, `serve`) does not need an export directory in `-m`. It picks the export of the `-m` model (LFM3-1.7B by default) from `~/models/NAME.amd-npu2-q4`, or else `~/models/NAME.amd-npu2` (`go run ./cmd/hwnpu-chat -m gemma-4-E2B-it`). Uses and families pick by memory, as on the other NPUs (`-m translate`, `-m chat`). On loading, it prints for each part of the export (`decode`, `prompt`) either `all on the NPU` (green) or, when a step runs on the host, `NOT all on the NPU` (red).

When there is no export, it works like the first run on the other NPUs, which downloads and converts the model. The first run downloads the checkpoint to `~/.cache/hwnpu/models/NAME` (only the files the export reads), exports it with IRON to `~/models/NAME.amd-npu2-q4` (`-q6` for Qwen3.5 0.8B and 2B), then runs it. How to export is written in `amd_export` in models.yaml (`lfm3 --max-seq-len 12288 --q4` and so on). Exporting needs the IRON environment (mlir-aie, Peano, Python 3.12; step 2 of [INSTALL_AMD.md](INSTALL_AMD.md)). Pass that virtual environment in `HWNPU_IRON_ENV`, or a script that sets it up (one that bash sources); without it, the python3 in PATH is used. Without the environment, it stops and says what to install. The export's output stays in `~/models/NAME.amd-npu2-q4.log`. Models without `amd_export` (Qwen3-0.6B, the embedding models) do not run on the AMD NPU (the RUNS FROM column of `hwnpu-chat list`).

```bash
HWNPU_IRON_ENV=~/mlir-aie/ironenv go run ./cmd/hwnpu-chat -m translate "桜が満開です。"
```


## Gemma 4 E4B

Gemma 4 E4B (gemma-4-E4B-it) is exported with the same `gemma4` (context 4096, chunks of 1024; the export is 15 GB). Decode runs at 3.6 tokens a second (about 60 % of E2B), and it reads 1024 tokens from position 3072 in 8.2 seconds.

## LFM3-1.7B

IRON has no application for LFM3-1.7B, so hwnpu's `tools/amd/iron_lfm3.py` is used (`lfm3`). The state of the convolution layers is kept at each boundary between prompt chunks, and hwnpu copies the state at those positions into marks and rolls back to them. Up to a context of 12288 and chunks of 2048, the sliding window layers do not become rings, so rewinding and the prompt store work. With `--q4`, the MLP and output matrices are 4 bit, the attention and short convolution projections 8 bit, and the embeddings stay on the host. On a Ryzen AI 9 365 this cuts pinned memory from 3.75 to 1.64 GiB and raises decode from 8.7 to 11.2 tokens a second (1.34 → 1.05 J a token). The KL against bf16 was 0.036 nats (the top token matches 92 %) (section 13 of [docs/amd-npu-models.md](docs/amd-npu-models.md)). Prompt speed does not change: it reads the first 2048 tokens in 2.4 seconds and 2048 tokens from position 10240 in 12.5 seconds (because attention reads the whole context, including what is outside the window). To keep bf16, leave out `--q4`.

```bash
python tools/amd/iron_export_lm.py lfm3 ~/.cache/hwnpu/models/LFM3-1.7B ~/models/LFM3-1.7B.amd-npu2-q4 --max-seq-len 12288 --prefill-chunk 2048 --q4
go run ./cmd/hwnpu-chat -m ~/models/LFM3-1.7B.amd-npu2-q4
go run ./cmd/hwnpu-chat agent -m ~/models/LFM3-1.7B.amd-npu2-q4 "Make go test pass"
```

## LFM3-VL, LFM3-Audio, LFM3-Omni

LFM3-VL-2B, LFM3-Audio-2B and LFM3-Omni-2B are also exported with hwnpu's IRON applications (`lfm3vl`, `lfm3audio`, `lfm3omni`). LFM3-VL stretches images to 512 × 512 and reads them (256 tokens), and LFM3-Audio reads up to 41 seconds of audio (it does not answer with audio). LFM3-Omni does both. `--q4` makes only the main model 4 bit and 8 bit (the towers stay bf16). Pinned memory goes from 4.12 → 2.00 GiB for VL, 4.14 → 2.03 for Audio and 4.51 → 2.39 GiB for Omni.

```bash
python tools/amd/iron_export_lm.py lfm3vl ~/models/LFM3-VL-2B ~/models/LFM3-VL-2B.amd-npu2-q4 --max-seq-len 12288 --prefill-chunk 2048 --q4
python tools/amd/iron_export_lm.py lfm3audio ~/models/LFM3-Audio-2B ~/models/LFM3-Audio-2B.amd-npu2-q4 --max-seq-len 12288 --prefill-chunk 2048 --q4
python tools/amd/iron_export_lm.py lfm3omni ~/models/LFM3-Omni-2B ~/models/LFM3-Omni-2B.amd-npu2-q4 --max-seq-len 12288 --prefill-chunk 2048 --q4
go run ./cmd/hwnpu-chat serve -m ~/models/LFM3-VL-2B.amd-npu2-q4 --listen 127.0.0.1:8080
```

## Images and audio with Gemma 4 E2B

To have Gemma 4 E2B read images and audio too, export it with its towers (the vision and audio encoders) (`gemma4mm`; up to one image and one audio clip a prompt). Pass them with `image_url` (a data URL) and `input_audio` (WAV) of the OpenAI compatible API.

```bash
python tools/amd/iron_export_lm.py gemma4mm ~/models/gemma-4-E2B-it ~/models/gemma-4-E2B-it.mm.amd-npu2 --max-seq-len 4096 --prefill-chunk 1024
go run ./cmd/hwnpu-chat serve -m ~/models/gemma-4-E2B-it.mm.amd-npu2 --listen 127.0.0.1:8080
```

## Memory and disk

NPU buffers are pinned memory and do not show in the process's RSS (1.8 GB for Gemma 4 E2B in 4 bit). `hwnpu-chat` (`chat`, `agent`, `serve`) unloads an AMD NPU export from the device and gives its memory back while no request comes for `--idle` (5 minutes by default), and loads it again on the next request (1 to 2 seconds for E2B). The conversation goes on from where it was, copied to the host. `--idle 0` never unloads. Exports with images and audio do not load the towers' weights until the first image or audio comes (2.82 → 1.83 GB for Gemma 4 E2B in 4 bit, 51 MiB more than the text only export; 0.7 seconds at the first image or audio; section 16.81 of [docs/design.md](docs/design.md)), so the same export also serves for text alone.

Several exports of the same model (text only and with images and audio, 4 bit and bf16, and so on) hold the same files (4.4 GiB of `ple.bin` for Gemma 4 E2B, for example). When the disk holding the models has a file system that can share copies, such as Btrfs or XFS (reflink), or ReFS or Dev Drive on Windows, hwnpu replaces files, and the starts of files, that hold the same bytes as a neighboring export with such shared copies (reflinks) the first time it reads an export. The disk then holds only one of them (15.2 → 8.2 GiB for two exports of Gemma 4 E2B; they stay ordinary files). On ext4 and NTFS it does nothing (section 16.79 of [docs/design.md](docs/design.md); not yet tried on Windows hardware).

## PLaMo 3 Translate

Preferred Networks' translation model PLaMo 3 Translate 2B (pfnet/plamo-3-translate-2b) is also exported with hwnpu's `tools/amd/iron_plamo3.py` (`plamo3`). hwnpu-chat translates what you type (text that contains Japanese into English, anything else into Japanese; `--to LANG` sets the target language, `--from` the source language, `--form plain|polite` the style of Japanese), and keeps the translations so far in the prompt as history. In the OpenAI compatible API, set `target_lang`, `source_lang`, `form`, `context` and `glossary` in `chat_template_kwargs` (as in vLLM); a system message becomes the reference context (`context`). With `--q4`, the MLP and output matrices are 4 bit, attention 8 bit, and the embeddings stay on the host. This cuts pinned memory from 5.96 to 2.16 GiB and raises decode from 7.3 to 11.6 tokens a second (Ryzen AI 9 365, around position 2600). 36 of 44 translated sentences were the same as bf16, and the rest were correct rewordings (section 13 of [docs/amd-npu-models.md](docs/amd-npu-models.md)). To keep bf16, leave out `--q4` (decode at 7.7 tokens a second, 6.6 at position 8191; a chunk of 1024 tokens takes 2.2 seconds from position 0 and 10.5 seconds from 7168). The Gemma 4 and LFM3 exports have `--q4` too.

```bash
python tools/amd/iron_export_lm.py plamo3 ~/models/plamo-3-translate-2b ~/models/plamo-3-translate-2b.amd-npu2-q4 --max-seq-len 8192 --prefill-chunk 1024 --q4
go run ./cmd/hwnpu-chat -m ~/models/plamo-3-translate-2b.amd-npu2-q4
go run ./cmd/hwnpu-chat -m ~/models/plamo-3-translate-2b.amd-npu2-q4 --to zh-hans
```

## Qwen3.5

Qwen3.5 (0.8B, 2B, 4B) is exported with hwnpu's `tools/amd/iron_qwen35.py` (`qwen35`). In 4 bit, 0.8B and 2B break down on long English text, so their MLP and output matrices are 6 bit (`--mlp q6 --head q6`; the attention and GDN projections are 8 bit, and the embeddings stay on the host. Pinned memory is 1.04 GiB for 0.8B and 1.97 GiB for 2B). Adding `--verify 3` also exports a version that checks the next token and 2 guesses in one call, and hwnpu-chat then generates while checking guesses taken from the prompt and the answer (speculative decoding; the logits are the same as one token at a time, so at temperature 0 the text is the same too). On a Ryzen AI 9 365, 0.8B went from 82 ms a token to 72 ms on code and 62 ms on Japanese explanations (section 16.117 of [docs/design.md](docs/design.md)). `HWNPU_IRON_DRAFTS=0` turns the guesses off.

```bash
python tools/amd/iron_export_lm.py qwen35 ~/models/Qwen3.5-0.8B ~/models/Qwen3.5-0.8B.amd-npu2-q6 --q4 --mlp q6 --head q6 --verify 3
go run ./cmd/hwnpu-chat -m ~/models/Qwen3.5-0.8B.amd-npu2-q6
```

## Phoenix and Hawk Point (NPU1)

Phoenix and Hawk Point (NPU1) have no full ELF, so export with `--device npu1 --cols 4`. Each version then becomes "an xclbin per step", as in IRON (1095 steps for Gemma 4 E2B's decode, 2 xclbins), and hwnpu makes the instruction stream for each call from the program included in the export. This is untested on NPU1 hardware. The same kind of export made for NPU2 with `--per-step` runs correctly on a Ryzen AI 9 365, decoding at 2.6 tokens a second (less than half of the 6 with full ELF, because a token needs 1095 submissions). On NPU1, the xclbins of the versions (6 in all) use up the limit of 6 hardware contexts.

```bash
python tools/amd/iron_export_lm.py gemma4 ~/models/gemma-4-E2B-it ~/models/gemma-4-E2B-it.amd-npu1 --max-seq-len 4096 --prefill-chunk 1024 --device npu1 --cols 4
```

## Speed

On a Ryzen AI 9 365, decode runs at 5.5 to 6 tokens a second and prompts at about 260 tokens a second (bf16 weights). Prompt chunks of 1024 tokens are fastest (20 % faster than IRON's default of 2048), and even with a context of 4096 tokens the sliding window layers do not become rings, so rewinding and the prompt store work. How it works and its limits are in [docs/amd-npu-models.md](docs/amd-npu-models.md).
