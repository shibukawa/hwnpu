# Audio commands (hwnpu-audio)

[日本語](USAGE_AUDIO.ja.md)

`hwnpu-audio` generates music and sound effects (`generate`), reads text aloud (`tts`), transcribes (`transcribe`), and finds pitch (`pitch`) and notes (`notes`), on the NPU. Each downloads its model on the first run and converts it by itself. `hwnpu-audio list` shows the models to choose from.

On a Mac, hwnpu-audio runs on the Neural Engine through Core AI alone, so it needs macOS 27 (on macOS 26 or earlier it stops with an error that says so). Each model is converted for Core AI (exports `NAME.coreai` of `.aimodel`s in the model's directory, as hwnpu-chat's and hwnpu-image's), and Core AI keeps the compiled programs in its own cache: a second start loads in 0.1 seconds (docs/design.md 16.196). `--cpu` runs a Core ML export with the CPU interpreter, for checking without the NPU.

## Music (Magenta RealTime 2)

`go run ./cmd/hwnpu-audio generate "disco funk"` uses Magenta RealTime 2 small to make 48 kHz stereo music in the style a text names, and plays it on the speakers as it goes (`-o out.wav` writes a file). On an M1 it runs about 1.4 times faster than real time. The weights are CC-BY 4.0, so when you use them, credit "Magenta RealTime 2 by Google DeepMind".

- The style of a recording: `song.wav`
- A mix of styles: `"jazz@2" "bossa nova"`
- MIDI notes: `--notes melody.mid`
- Continuing a recording: `--continue intro.wav`
- Play and write a file at once: `--play -o out.wav`; stream to standard output: `-o -`

## Sound effects (MOSS-SoundEffect)

The same `generate` with `-m sfx` uses MOSS-SoundEffect v2.0 (Apache-2.0) to make 48 kHz sound effects of up to 30 seconds from a description in English or Chinese. The text encoder (Qwen3-1.7B), the DiT (1.3B) and the DAC decoder all run on the NPU; on an M1, the model card's 100 steps take about 4 minutes for 10 seconds of sound. By the machine's memory, 16 GB or more picks the model with the text encoder in fp16, and 8 GB the lut8 model (`-m moss-soundeffect:8gb` or `:16G` picks one explicitly).

```bash
go run ./cmd/hwnpu-audio generate -m sfx "A dog barking loudly in a park."
go run ./cmd/hwnpu-audio generate -m sfx --duration 20s --steps 50 "Rain on a tin roof."   # --steps 50 takes half the time
```

## Reading aloud (LFM3-Audio-2B)

`go run ./cmd/hwnpu-audio tts --lang ja -o out.wav こんにちは` reads text aloud with LFM3-Audio-2B (`-m speech`). Details are in section 5 of [CHAT.md](CHAT.md).

## Transcription (Whisper)

`go run ./cmd/hwnpu-audio transcribe speech.wav` transcribes speech with Whisper large-v3-turbo. It downloads the checkpoint on the first run and converts it. On an M3 it reads a 30 second window in 0.6 seconds, and transcribed 65 seconds of Japanese in 4 seconds. `-l ja`, `--translate` and `--timestamps` work.

## Pitch and notes (SwiftF0, Basic Pitch)

`go run ./cmd/hwnpu-audio pitch voice.wav` gives the pitch of a voice or instrument every 16 ms with SwiftF0, and `go run ./cmd/hwnpu-audio notes --midi song.mid song.wav` writes notes to MIDI with Basic Pitch. Both models are distributed as ONNX, and `internal/onnx` turns their static parts into NPU programs. On AMD machines they run on the CPU (as in the object detection section of [USAGE_IMAGE.md](USAGE_IMAGE.md), ONNX Runtime is used when it is installed).
