# Audio commands (hwnpu-audio)

[日本語](USAGE_AUDIO.ja.md)

`hwnpu-audio` generates music and sound effects (`generate`), reads text aloud (`tts`), transcribes (`transcribe`), and finds pitch (`pitch`) and notes (`notes`), on the NPU. Each downloads its model on the first run and converts it by itself. `hwnpu-audio list` shows the models to choose from.

On a Mac, hwnpu-audio runs on the Neural Engine through Core AI alone, so it needs macOS 27 (on macOS 26 or earlier it stops with an error that says so). Each model is converted for Core AI (exports `NAME.coreai` of `.aimodel`s in the model's directory, as hwnpu-chat's and hwnpu-image's), and Core AI keeps the compiled programs in its own cache: a second start loads in 0.1 seconds (docs/design.md 16.196). `--cpu` runs a Core ML export with the CPU interpreter, for checking without the NPU.

On AMD's NPU (Ryzen AI 300 and later) each model runs from an export of IRON's (`tools/amd`), which the first run makes (IRON's environment as [INSTALL_AMD.md](INSTALL_AMD.md) says) unless a source has it, and puts in the sources written back to: transcription, speech and sound effects on the NPU, music's decoder on the NPU and its language model on the CPU. `pitch` and `notes` run on the CPU.

## Music (Magenta RealTime 2)

`go run ./cmd/hwnpu-audio generate "disco funk"` uses Magenta RealTime 2 small to make 48 kHz stereo music in the style a text names, and plays it on the speakers as it goes (`-o out.wav` writes a file). On an M1 it runs about 1.4 times faster than real time. The weights are CC-BY 4.0, so when you use them, credit "Magenta RealTime 2 by Google DeepMind".

- The style of a recording: `song.wav`
- A mix of styles: `"jazz@2" "bossa nova"`
- MIDI notes: `--notes melody.mid`
- Continuing a recording: `--continue intro.wav`
- Play and write a file at once: `--play -o out.wav`; stream to standard output: `-o -`

On AMD's NPU SpectroStream's decoder runs as the graph of `tools/amd/iron_magenta.py`, and the language model and MusicCoCa on the CPU. The first run exports the decoder with IRON (about 15 seconds) unless a source has it (`magenta-realtime-2-small.amd-npu2`). On a Ryzen AI 9 365 it makes music at about 0.3 times real time (4 seconds in 13 seconds: the decoder 1.6 seconds and the language model 1.6 seconds a second of music; all on the CPU 28 seconds), at about 2.7 GB.

## Sound effects (MOSS-SoundEffect)

The same `generate` with `-m sfx` uses MOSS-SoundEffect v2.0 (Apache-2.0) to make 48 kHz sound effects of up to 30 seconds from a description in English or Chinese. The text encoder (Qwen3-1.7B), the DiT (1.3B) and the DAC decoder all run on the NPU; on an M1, the model card's 100 steps take about 4 minutes for 10 seconds of sound. By the machine's memory, 16 GB or more picks the model with the text encoder in fp16, and 8 GB the lut8 model (`-m moss-soundeffect:8gb` or `:16G` picks one explicitly).

```bash
go run ./cmd/hwnpu-audio generate -m sfx "A dog barking loudly in a park."
go run ./cmd/hwnpu-audio generate -m sfx --duration 20s --steps 50 "Rain on a tin roof."   # --steps 50 takes half the time
```

On AMD's NPU the text encoder, the DiT and the DAC decoder run as the graphs of `tools/amd/iron_moss_sfx.py` and `iron_moss_dac.py` (the DiT's GEMMs in bfp16, the text encoder in bf16, or in 8 bits for `moss-soundeffect:8g`; the decoder's Snake by a float32 kernel that takes sin by its series, as the version for Intel's NPU does). The first run exports them with IRON (about 30 seconds once IRON's kernels are built) unless a source has the export (`moss-soundeffect-v2.0.amd-npu2`). On a Ryzen AI 9 365 a step takes 2.8 seconds and the decoder 1.7 seconds for 5 seconds of sound (5 seconds in 25 steps in 1 minute 6 seconds; the card's 100 steps about 4 minutes 40 seconds), at about 5 GB.

## Reading aloud (LFM3-Audio-2B)

`go run ./cmd/hwnpu-audio tts --lang ja -o out.wav こんにちは` reads text aloud with LFM3-Audio-2B (`-m speech`). Details are in section 5 of [CHAT.md](CHAT.md).

On AMD's NPU it runs the export of `tools/amd/iron_lfm3speech.py`: the text model (8 bit, its final hidden state out), the depthformer's MaskGIT pass and the codec decoder, each a program on the NPU; the host samples the codes and runs the inverse STFT. The first run exports it with IRON (about a minute once IRON's kernels are built) unless a source has it (`lfm3-audio-2b-tts.amd-npu2`). On a Ryzen AI 9 365 it speaks at about 0.43 times real time (5.1 seconds of Japanese in 11.8 seconds: the text model 107 ms and the depthformer's three passes 72 ms a frame of 80 ms), at about 2.8 GB.

## Transcription (Whisper)

`go run ./cmd/hwnpu-audio transcribe speech.wav` transcribes speech with Whisper large-v3-turbo. It downloads the checkpoint on the first run and converts it. On an M3 it reads a 30 second window in 0.6 seconds, and transcribed 65 seconds of Japanese in 4 seconds. `-l ja`, `--translate` and `--timestamps` work.

On AMD's NPU (Ryzen AI 300 and later) it runs the two graphs of `tools/amd/iron_whisper.py`, the encoder of a window and the decoder of one token, which keeps its KV cache on the NPU. The first run downloads the checkpoint and exports it with IRON (about 20 seconds once IRON's kernels are built; IRON's environment as [INSTALL_AMD.md](INSTALL_AMD.md) says, `HWNPU_IRON_ENV`), unless a source has the export (`whisper-large-v3-turbo.amd-npu2`). On a Ryzen AI 9 365 a window's encoder takes 0.84 seconds and the decoder 51 tokens a second: 88 seconds of English speech in 9 seconds, at about 2.1 GB (the NPU's pinned memory and the process's). `-m DIR` runs an export's directory.

## Pitch and notes (SwiftF0, Basic Pitch)

`go run ./cmd/hwnpu-audio pitch voice.wav` gives the pitch of a voice or instrument every 16 ms with SwiftF0, and `go run ./cmd/hwnpu-audio notes --midi song.mid song.wav` writes notes to MIDI with Basic Pitch. Both models are distributed as ONNX, and `internal/onnx` turns their static parts into NPU programs. On AMD machines they run on the CPU (as in the object detection section of [USAGE_IMAGE.md](USAGE_IMAGE.md), ONNX Runtime is used when it is installed).
