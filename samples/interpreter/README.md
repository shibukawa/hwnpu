# interpreter

[日本語](README.ja.md) · [All samples](../README.md)

Interprets speech into another language, as speech: from the microphone, or from a recording. Japanese speech becomes English, any other language Japanese (or the language you ask for).

```
mic → vad → transcribe → heard → commands → translate → said → speech → out
```

`transcribe`, `translate` and `speech` are models on the NPU (Whisper large-v3-turbo, PLaMo 3 Translate, LFM3-Audio TTS). The others are steps of the program: `vad` cuts the sound into utterances by its loudness, `heard` and `said` print what passes, and `commands` catches spoken commands instead of translating them. The speaker runs behind the others, so the next utterance is transcribed and translated while one is spoken.

## Run

```bash
cd samples
go run ./interpreter models prepare            # download, convert and compile the models (about 11 GB)
go run ./interpreter talk.wav -o english.wav   # a recording: Japanese speech to English speech
go run ./interpreter --mic -o english.wav      # the microphone, until Ctrl-C or "終わり"
go run ./interpreter say こんにちは -o hello.wav  # a text, translated and spoken
```

| Command | What it does |
|---|---|
| `interpreter FILE.wav` | Interprets a recording (a WAV file of any rate and channels). `--live` gives it at the pace a microphone would and shows each step's delay |
| `interpreter --mic` | Interprets what the microphone hears. `--seconds 30s` stops after that long |
| `interpreter mics` | The audio inputs, numbered: `--device 2` takes the second |
| `interpreter say TEXT` | One call of the translator and the speaker |
| `interpreter models` | The models picked for this machine, what they take, and the network |
| `interpreter status` | This machine's NPU and what uses it |

Options: `--to es` (the language to interpret into), `-l ja` (the language of the speech, instead of finding it in each utterance), `--voice male`, `--no-commands`.

### Spoken commands

| Say | What happens |
|---|---|
| 「変換、スペイン語」 / "switch to Spanish" | The next utterances are interpreted into Spanish |
| 「終わり」 / "that's all" | The interpreter ends once what it heard is spoken |

## Download

| Model | Apple | AMD |
|---|---|---|
| Whisper large-v3-turbo (transcribe) | 1.6 GB | 1.5 GB |
| PLaMo 3 Translate 2B (translate) | 5.8 GB | 2.4 GB |
| LFM3-Audio 2B TTS (speech) | 4.0 GB | 2.8 GB |
| **Total** | **11.4 GB** | **6.6 GB** |

For a machine of 16 GB; `interpreter models` prints this machine's. On a Mac the downloads are checkpoints converted here; on AMD they are exports for the NPU.

## What it takes

- Memory: about 7.5 GB running together. The translator is PLaMo 3 Translate 2B on machines of 16 GB and up (`interpreter models` shows why).
- Speed: a 5-second Japanese sentence became 4.9 s of English speech in 5.3 s on an M3 (from a recording, the models loaded).
- On a Mac the terminal app needs microphone permission; macOS asks at the first `--mic`.

The network is declared in [main.go](main.go) with `ai/flow`; [docs/developer-guide.md](../../docs/developer-guide.md) walks through it.
