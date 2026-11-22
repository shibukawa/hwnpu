# voice-agent

[日本語](README.ja.md) · [The samples](../README.md)

A small voice assistant that answers aloud when called by its name, "コダマ" (kodama: "コダマ、今何時？", "ねえコダマ、明日の天気は？"). It tells the time and the date, sets timers, reads the weather ([Open-Meteo](https://open-meteo.com)), the news headlines (NHK's RSS) and Wikipedia; other questions the chat model answers itself. It speaks Japanese.

```
mic → vad → transcribe → name? → shortcut (time, timers) ──────────→ speech → speaker
                                └→ chat model ⇄ tools → sentences ─┘
```

The models run on the NPU: Whisper large-v3-turbo (transcription), LFM3-1.7B (chat with tools) and Supertonic 3 (speech, OpenRAIL-M).

What is general is in the library: the conversation with its tool calls is [`ai/agent`](../../ai/agent), hearing, speaking and answering soon [`ai/agent/voice`](../../ai/agent/voice). The sample has its own system prompt and what it knows of the user (prompt.go), its tools (tools.go, timers.go), the requests answered without the model (shortcut.go) and its commands (main.go): another voice assistant is the same packages with other tools and another prompt.

## Running it

```bash
cd samples
go run ./voice-agent models prepare          # once: download (about 7.5 GB), convert, compile
go run ./voice-agent                         # listen to the microphone until Ctrl-C
go run ./voice-agent --place 大阪 --user 太郎さん --about "経済のニュースに関心がある"
go run ./voice-agent run talk.wav            # a recording, at a microphone's pace
go run ./voice-agent ask 明日の天気は？       # one request as text, answered aloud
go run ./voice-agent ask --text 今日のニュース # answered in text (the chat model alone is loaded)
```

- Say "コダマ、..." to ask. "コダマ" alone is answered with a chime, and the next words are the request.
- For 8 seconds after it has spoken (`--follow`), the next request needs no name.
- The time, the date and timers ("3分のタイマー", "ラーメンのタイマーを止めて", "タイマーあと何分？", "今何時？") and "会話をリセットして" are answered without the model (`--no-shortcuts` sends them to it).
- The conversation is forgotten after 5 minutes without a request (`--reset`), and only the last few turns are kept (`--history`), so the prompt stays short and quick to read.
- `--place` (where the user is: the weather's place by default), `--user` (how to call them) and `--about` (a sentence or two about them) go into the system prompt, which does not change while it runs and so stays in the model's cache.

## How it answers soon

What happens is in `Listener` of `ai/agent/voice`:

1. A pause of 250 ms (`--pause`) starts the transcription, before the VAD ends the utterance (after 700 ms of silence). When the speaker does not go on, that transcript is the utterance's, and most of Whisper's time is gone.
2. The chat model starts on it too (`--speculate`, on by default). Nothing is said and no tool runs before the utterance has ended. When the speaker goes on, the turn is cancelled and the conversation is as it was.
3. The time, the date and timers need no model.
4. The answer is spoken a piece at a time as the chat model writes it, the first piece cut short at a comma. Supertonic 3 speaks at about 80 times real time on the Neural Engine (under 0.1 s a sentence), so making the speech is hardly a wait.
5. Phrases said often ("ちょっと待ってね", said before a tool that goes to the network) are synthesized once and kept on disk. The chime and the alarm are tones, and a timer's "…の時間です" is synthesized when it is set.
6. The system prompt and the tools are read into the model's cache at the start and after a reset.

The log tells how long after the end of the speech the transcript came and the answer's first sound was played.

## Limits

- Supertonic 3 is under OpenRAIL-M: its use restrictions (the model card's) hold for what is made with it too.
- By default the microphone is not heard while the assistant speaks, nor for 0.3 s after. With `--barge-in` it is, and its name stops it, but without headphones it hears itself, which also slows its speech.
- The name is found in the transcripts: every utterance heard goes through Whisper.
- The weather and the news come from services on the network: Open-Meteo's geocoding finds the place (with 市, 都, 府 or 県 added, the most populous place), and the news comes from the RSS feed of `--feed`.
