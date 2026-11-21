# Developer guide: building apps with hwnpu (the `ai` packages)

[日本語](developer-guide.ja.md)

> **v0**: this API is a first version and its shape will still change. This document is both a guide to using it and material for reviewing its design at this stage. The design questions are numbered in [section 10](#10-design-questions), to be referred to ("D4: option B").

hwnpu is a set of Go libraries that run AI models on the NPU. `hwnpu-chat`, `hwnpu-image` and `hwnpu-audio` are demos that try its features one at a time. An app registers several models, prepares them ahead of time, and runs them joined together.

- There now: speech to text, translation, chat, text to speech, a VAD by loudness, the ways of joining models, commands for an app's command line
- Not there yet: pictures (depth, faces, objects, generation), music and sound effects, managing what stays loaded, a local server ([section 11](#11-not-there-yet))
- It was run on AMD's NPU (Ryzen AI 9 365) only. On a Mac and on Intel it calls the same preparing and loading as the commands, but it was not run through this library there
- The whole plan is in [app-api.md](app-api.md) and the list of work in [app-api-tasks.md](app-api-tasks.md) (both in Japanese)

## 1. The whole picture

```
register                 plan                 prepare                     run
ai.NewApp(uses...)  →  app.Plan(ctx)   →   plan.Run(ctx, fn)     →    app.Open(ctx) → use.Get(ctx, rt)
the models the app      what is missing,     download, convert,          loads from the prepared
uses                    bytes to download,   compile for the NPU         store alone (no network)
                        memory, licenses
```

| Package | What it is for |
|---|---|
| `github.com/shibukawa/hwnpu/ai` | `App` (registration), `Use[T]` (a typed handle), `Plan` (planning and preparing), `Runtime` (running), `Store` (where things are kept) |
| `.../ai/speech` | Speech to text `Transcribe`, text to speech `Speak`, a VAD by loudness, reading and writing WAV |
| `.../ai/text` | Translation `Translate`, chat `Chat` |
| `.../ai/flow` | Passing data between models: `Stream` and its tools, the network (nodes and `Connect`) |
| `.../ai/aicli` | kong commands for the app's command line: `models`, `status` |

**Words**

- **Use**: the name of a purpose, such as `transcribe` or `translate`. Uses are models.yaml's `categories`; each picks a model that fits the machine's memory. `hwnpu-chat list` shows them
- **Family and tag**: a line of models and one of them, as in `plamo3translate:2b`
- **Role**: what a model runs as (`transcribe`, `translate`, `chat`, `speak`). The function of a package of models (`speech.Transcribe` and the others) decides it
- **Store**: where the models' files and programs are. By default it is the commands' cache (`~/.cache/hwnpu` and the like)
- **Target**: which NPU the programs are for: `coreai` (Apple), `intel`, `amd-xdna2`

## 2. The smallest example

A program that reads a WAV file and writes its text.

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/speech"
)

// the models the app uses are kept in package variables
var (
	stt = speech.Transcribe("transcribe")
	app = ai.NewApp("dictation", stt)
)

func main() {
	ctx := context.Background()

	// prepare: download what is missing, convert it, compile it for the NPU (nothing from the second run on)
	err := app.Prepare(ctx, func(p ai.Progress) {
		fmt.Fprintln(os.Stderr, p.Use, p.Stage, p.Message)
	})
	if err != nil {
		log.Fatal(err)
	}

	// run
	rt, err := app.Open(ctx)
	if err != nil {
		log.Fatal(err)
	}
	defer rt.Close()
	t, err := stt.Get(ctx, rt) // *speech.Transcriber
	if err != nil {
		log.Fatal(err)
	}
	raw, err := os.ReadFile(os.Args[1])
	if err != nil {
		log.Fatal(err)
	}
	pcm, err := speech.ReadWAV(raw, speech.TranscribeRate) // mono, 16 kHz
	if err != nil {
		log.Fatal(err)
	}
	res, err := t.Transcribe(ctx, pcm, speech.TranscribeOptions{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(res.Text)
}
```

A working example that joins several models is [samples/interpreter](../samples/interpreter): it interprets speech, from the microphone or a recording, into speech in another language. The samples are a module of their own (`samples/go.mod`) and take the microphone and the camera from [hwcamera](https://github.com/shibukawa/hwcamera), which is not a dependency of the library ([section 6.5](#65-live-inputs-the-microphone-and-the-camera)).

```
cd samples
go run ./interpreter models prepare
go run ./interpreter --mic -o english.wav      # from the microphone, until Ctrl-C or "that's all"; "switch to Spanish" changes the language
go run ./interpreter talk.wav -o english.wav   # from a recording
```

## 3. Registering models

### 3.1 `Use` and `App`

```go
var (
	stt = speech.Transcribe("transcribe")          // ai.Use[*speech.Transcriber]
	tr  = text.Translate("plamo3translate:2b")     // ai.Use[*text.Translator]
	tts = speech.Speak("speech")                   // ai.Use[*speech.Speaker]

	app = ai.NewApp("interpreter", stt, tr, tts)
)
```

- A name is one of models.yaml's: a model (`whisper-large-v3-turbo`), a family with a tag (`plamo3translate:2b`), a family (`plamo3translate`) or a use (`translate`)
- A `Use[T]` is a value. It carries the type `T` that `Get` returns, so nothing has to be cast when a model is taken out
- Instead of listing the models, give the app its network (section 6.6) with `app.Connect`: the models of its nodes are then the app's
- `app.Use(more...)` adds more later. A `Use` of the same name and role registered twice stays as it was first registered
- A name that does not fit the role (`speech.Transcribe("chat")`) is an error of `Plan` and of `Get`

### 3.2 A line-up by the machine's memory

A use's name picks the largest model the machine runs **when it runs alone** (`translate` is PLaMo 3 Translate 8B, 8.2 GB, from 16 GB machines). An app that runs several at once wants its main model large and the others small and fast, so it writes its line-up when it registers.

```go
tr := text.Translate("translate").      //  8 GB: the pick of the definitions (the 0.8B)
	At(16, "plamo3translate:2b").       // 16 GB and up: the 2B
	At(32, "plamo3translate:8b")        // 32 GB and up: the 8B

helper := text.Chat("chat").As(8)       // on every machine, pick as an 8 GB machine would
```

| | Meaning |
|---|---|
| `use.At(gb, name)` | On machines of at least gb GB of memory the use is `name`. Several may be given; the one of the most memory the machine has counts. A machine below all of them takes the use's own name |
| `use.As(gb)` | Pick as a machine of gb GB would: both which model, and which of the model's programs (models.yaml's `variants`) |
| `app.Memory(gb)` | Plan the whole app as for a machine of gb GB (to prepare for smaller machines). The tiers of `At` count for that memory too |

- `At` and `As` return a new `Use`. The value they are called on does not change
- What the user sets works as it does for the commands: `HWNPU_MEMORY` (calls the machine's memory something else), `hwnpu memory MODEL GB` (by model), `HWNPU_MODEL_ALLOWLIST` (which models may be used)
- The model picked and why are in `plan.Items[i].Model` and `.Why`. `plan.MemoryGB()` is what the models take together and `plan.MachineGB()` the machine's memory

### 3.3 Definitions of the app's own

An app can add a definitions file of its own (in models.yaml's format): its own models, its own uses (with `picks` by memory), aliases.

```go
app.Store = ai.Store{Models: modelstore.Options{Files: []string{"myapp-models.yaml"}}}
```

```yaml
version: 1
categories:
  interpreter-translate:
    picks:
      - {machine: 8, model: plamo-3-translate-0.8b}
      - {machine: 16, model: plamo-3-translate-2b}
```

This and `At` of section 3.2 say the same thing, one in code and one in a file ([D5](#10-design-questions)).

## 4. Preparing

### 4.1 The plan

```go
plan, err := app.Plan(ctx)   // downloads nothing
for _, it := range plan.Items {
	fmt.Println(it.Use, it.Model, it.Ready, it.Compiled, it.Download, it.MemoryGB, it.Why)
}
plan.DownloadBytes()   // bytes to download (at most)
plan.MemoryGB()        // memory of all the models running together
plan.MachineGB()       // the machine's memory
plan.Licenses()        // each model's license (id, class, URL, whether it asks to be accepted, whether it was)
plan.Target()          // coreai, intel, amd-xdna2; "" when no NPU here runs the models
```

- `Ready`: the files and programs are in the store, and the model loads without the network
- `Compiled`: the model was loaded once on this machine's NPU (its programs are compiled)
- `Download` is models.yaml's `bytes`, the most there is to download. When a source has the converted programs, the checkpoint is not downloaded

### 4.2 Preparing has two stages

```go
plan.NoCompile = true                       // download and convert only
err = plan.Run(ctx, func(p ai.Progress) {   // p.Stage: prepare, compile, done
	...
})
```

| Stage | What it does | NPU |
|---|---|---|
| Download and convert | Takes the files from the sources (this machine's cache, `HWNPU_MODEL_SOURCE*`, models bundled with the app, Hugging Face) and converts them into programs. When a source has the converted programs, it only takes them | Not needed |
| Compile | Loads each model once. Compiling for Apple's Neural Engine takes from seconds to minutes on the first load (100 s for Gemma 4), and what it makes belongs to the machine | Needed |

- A `plan.Run` that was cut off goes on from where it stopped when called again
- `Progress.Message` is a line of the log of a download or a conversion. There are no byte counts yet ([D15](#10-design-questions))
- `app.Prepare(ctx, fn)` is `Plan` and then `Run`
- What was made also goes to the sources that are written back to (`?writeback=true`). `Run` waits for that before it returns

### 4.3 Licenses

When a model's license asks to be accepted (`License.Accept`), `plan.Run` returns a `*ai.LicenseError`. The app shows `plan.Licenses()`, gets the user's agreement, sets `plan.AcceptLicenses = true`, and calls `Run`. The acceptance is kept in the user's settings folder (`licenses.json`), shared with the commands.

### 4.4 The store, and shipping what was prepared

```go
app.Store = ai.Store{
	Models:   modelstore.Options{Root: cacheDir, Sources: []string{"https://models.example.com/hwnpu"}},
	Prepared: []string{filepath.Join(resources, "prepared")},
}
```

- The zero value of `Store` is the commands' store. The app uses what the commands prepared, and the commands use what the app prepared
- `Store.Models` is [`modelstore.Options`](../USAGE_LIBRARY.md): where the cache is, the sources, the models bundled with the app, the cache's limits, the app's definitions files
- What was prepared is written into `store.json` at the root of the cache (by the NPU's target, then by `ROLE:NAME`)
- `Store.Prepared` are folders with a copy of a prepared cache's `store.json` and the files it names. They are read only and used where they are. A use found in one loads without being prepared
- The store's settings (where the cache is, the sources, the definitions) are one per process for now. Two `App`s with different `Store`s cannot be used at once in a process ([D8](#10-design-questions))

## 5. Running

```go
rt, err := app.Open(ctx)            // ai.WithLog(w), ai.WithAutoPrepare(fn), ai.WithAcceptedLicenses()
defer rt.Close()

t, err := stt.Get(ctx, rt)          // loads at the first call (under a second to tens of seconds); the same one after
...
err = stt.Release(rt)               // unloads it and frees its memory; the next Get loads it again
rt.Loaded()                         // the models held now: use, model, memory, how long loading took
```

- **It touches no network and converts nothing.** A use that is not prepared is `ai.ErrNotPrepared`. With `ai.WithAutoPrepare(fn)` the first `Get` prepares it, as the commands do; that may download gigabytes
- **A store whose record is not what is picked now counts as not prepared.** When the use was prepared as another model (the app's line-up changed, or the machine's memory setting did), the old one is not loaded in its place
- **Loads run one at a time.** Each model runs one request at a time (others wait). Different models may be called at the same time
- **An app that uses its models one after another** unloads them with `Release`: a picture's generator, then its depth, with only the larger of the two in memory. Nothing unloads models by itself yet ([D9](#10-design-questions)). This is another level than the swapping of stages that image generation does inside ([section 9.5](#95-release-and-image-generation-swapping-its-stages))
- **Models can be called at the same time, but the NPU is one queue.** Whether a pipeline keeps up in real time is decided by what its stages' times add up to ([section 9](#9-constraints-and-rules))
- A model's `OnNPU()` reports whether its programs were verified to be on the NPU (`Speaker` does not have it yet)

### 5.1 Errors

| Error | When | What to do |
|---|---|---|
| `ai.ErrNotPrepared` | The use is not prepared, is prepared as another model, or its files are gone | `plan.Run`, or `WithAutoPrepare` |
| `ai.ErrBundleTarget` | What `Store.Prepared` has is for another NPU alone, or in a format that is not read | Ship what was prepared for that NPU, or prepare on the machine |
| `*ai.LicenseError` | A license that asks to be accepted was not | Get the agreement, then `plan.AcceptLicenses` |
| `flow.ErrEmpty` | A stream ended without a chunk (`Held.Wait`) | — |
| `Get` of a `Use` that is not registered | It is not one of the `app`'s | Add it to `ai.NewApp` or `app.Use` |

Check them with `errors.Is` and `errors.As`.

## 6. Joining models (`ai/flow`)

### 6.1 Chunks and streams

```go
type Chunk[T any] struct {
	Data       T
	Start, Len time.Duration // where it is from the start of the stream, and how long it lasts
	Final      bool          // the last chunk of a segment
}

type Stream[T any] = iter.Seq2[Chunk[T], error]

for c, err := range stream {
	if err != nil { return err }
	use(c.Data)
}
```

- A stream is a Go iterator. It is read with `range` and stops at a `break`, an error, or the cancel of its context. Nothing comes after an error
- **A stage is a function from streams to a stream.** Stages are joined by passing one's result to the next. A stage goes no further than its reader reads (unless `flow.Async` is between them)
- **Times are the stream's own.** Audio counts them in samples, so a late microphone or a jittery network does not move them. Only `Metronome`, `Pace` and `Live` look at the wall clock
- **`Final`** ends a segment: the last chunk of an utterance's audio, a settled sentence, a finished picture. In a stream of text, the chunks before a final one are "the text so far", each replacing the one before (captions while someone speaks, an answer being generated)
- Audio is `[]float32` (mono, −1 to 1) and text is `flow.Text{Text, Lang}`. The sampling rate is not in the chunk: a transcriber and a VAD read 16 kHz (`speech.TranscribeRate`), and a speaker gives `Speaker.Rate()` (24 kHz) ([D11](#10-design-questions))

### 6.2 Three ways of joining

They differ in when the next stage may start.

**(1) When the one before has finished.** The next stage needs the whole result (a picture, then its depth).

```go
img, err := gen.Generate(ctx, prompt)      // a batch call: one value in, one out
if err := genUse.Release(rt); err != nil { ... }   // a large model is unloaded first
est, err := depthUse.Get(ctx, rt)
d, err := est.Estimate(ctx, img)
```

`flow.Last(stream)` waits for all of a stage that gives a stream, and `flow.Of(value, length)` gives a whole value to a stage that reads one. (`gen` and `est` are of packages that are not there yet.)

**(2) As the chunks come.** The next stage starts on each chunk while the one before makes the next.

```go
utterances := speech.NewVAD(speech.VADOptions{}).Detect(ctx, mic)   // the speech alone
texts := stt.TranscribeStream(ctx, utterances, speech.TranscribeOptions{})
en := tr.TranslateStream(ctx, texts, text.TranslateOptions{To: "en"})
audio := tts.SpeakStream(ctx, flow.Async(ctx, en, 4), speech.SpeakOptions{})
for c, err := range audio { ... }
```

| Tool | What it does | Drops chunks? |
|---|---|---|
| `flow.Async(ctx, s, n)` | Reads `s` in a goroutine of its own, up to n chunks ahead. Stages overlap (the second sentence is translated while the first is spoken) | No. With n chunks waiting, the stage before waits |
| `flow.Latest(ctx, s)` | Reads `s` in a goroutine of its own and gives its reader the newest chunk only. Chunks that came while the reader was busy are dropped (a camera's frames into a detector that takes longer than a frame) | Yes (never the last chunk, nor the error) |
| `flow.Hold(ctx, s)` | Reads `s` in a goroutine of its own and keeps the newest chunk: `Latest()`, `Wait(ctx)`, `Stop()` | Overwrites |
| `flow.Tee(s, n)` | One stream to n readers (the same transcript to captions and to a translation) | No. What a slower reader has not read waits in memory |

**(3) On the beat.** The output is due at fixed times (audio in time for a player, frames for a screen).

```go
// draw 60 times a second, with the newest position of the head the detector found
head := flow.Hold(ctx, faces.DetectStream(ctx, flow.Latest(ctx, camera)))
defer head.Stop()
for tick, err := range (flow.Metronome{Period: time.Second / 60, Skip: true}).Ticks(ctx) {
	if err != nil { break }
	pos, _ := head.Latest()
	draw(mesh, pos.Data, tick.Data.At)
}

// a chunk of audio every 2 seconds: the tick comes 500 ms before its beat, to make the chunk in
for tick, err := range (flow.Metronome{Period: 2 * time.Second, Lead: 500 * time.Millisecond}).Ticks(ctx) {
	if err != nil { break }
	chunk := generate(tick.Data.N)          // tick.Data.Late > 0: the stage is not keeping up
	player.At(tick.Data.At, chunk)
}
```

(`faces`, `camera`, `draw`, `generate` and `player` are made up. The metronome, `Hold` and `Latest` are there.)

| Tool | What it does |
|---|---|
| `flow.Metronome{Period, Lead, Skip}.Ticks(ctx)` | A stream of a `Tick{N, At, Late, Skipped}` a beat. The first tick comes at once and its beat is `Lead` later. When the reader is late: without `Skip` every beat is given, late (an output that must have no holes); with `Skip` the beats that are over are dropped and the present one is given (an output that shows the present) |
| `flow.BPM(bpm)` | The period of bpm beats a minute |
| `flow.Pace(ctx, s, ahead)` | Passes each chunk on at the time of its `Start` (the first one at once). `ahead` is how far ahead of the clock chunks may pass (a player's buffer). A late chunk passes at once; none is dropped |
| `flow.Live(ctx, s)` | Passes a recorded stream on in real time, as a microphone would (each chunk at the time of its end): a file standing in for a microphone or a camera |

### 6.3 The other tools

| Tool | What it does |
|---|---|
| `flow.Of(v, len)`, `flow.From(vs...)` | A stream from values |
| `flow.Collect(s)`, `flow.Last(s)` | Read a stream to its end: all the chunks, the last chunk's data |
| `flow.Map(s, fn)` | A stage from a function (times and `Final` are kept) |
| `flow.Finals(s)` | The final chunks only (the settled sentences) |
| `flow.Rechunk(s, rate, n)` | Cut audio again into chunks of n samples |
| `flow.Sentences(s)`, `flow.SplitSentences(text)` | Cut text into sentences (Japanese and Western sentence ends) |
| `flow.Samples(n, rate)` | How long n samples last |
| `speech.Chunks(pcm, rate, n)` | An array of audio as a stream of chunks of n samples |

### 6.4 Rules

- A stream given to `Async`, `Latest` or `Hold` must stop when ctx is cancelled. Their goroutine ends before the stream they return does (read to the end, `break`, cancel). While the stream before is waiting for its next chunk, it ends only when that chunk comes
- A stage of your own passes an error chunk on and ends, and returns at once when `yield` returns false

```go
func Upper(s flow.Stream[flow.Text]) flow.Stream[flow.Text] {
	return func(yield func(flow.Chunk[flow.Text], error) bool) {
		for c, err := range s {
			if err != nil {
				yield(c, err)
				return
			}
			c.Data.Text = strings.ToUpper(c.Data.Text)
			if !yield(c, nil) {
				return
			}
		}
	}
}
```

- Cancelling is ctx. There is no way yet to interrupt (stop speaking when the other person starts) ([D13](#10-design-questions))

### 6.5 Live inputs: the microphone and the camera

The `ai` packages have no microphone, camera or speaker. An input is anything that is a `flow.Stream`: made from a file or from a device, the stages after it are the same. The samples use [hwcamera](https://github.com/shibukawa/hwcamera) (each operating system's camera and audio input API without cgo), with the little code that makes streams of it in `samples/capture`.

```go
// samples/capture: the microphone (hwcamera/audio) as a stream
mic := capture.Mic(ctx, capture.MicOptions{Rate: speech.TranscribeRate}) // flow.Stream[[]float32]

// from a recording; the stages after it are the same
mic := speech.Chunks(pcm, speech.TranscribeRate, speech.TranscribeRate/10)
mic = flow.Live(ctx, mic) // to pace it as a microphone would
```

**A live input does not wait.** A stream is pulled by its reader, so while the stages after it are busy nobody reads the input, and the sound or the frames of that time are lost at the device. What to do about it is decided right after the input.

| Input | What to do |
|---|---|
| Sound (all of it must be heard) | Read it in a goroutine of its own and keep it: `flow.Async(ctx, mic, n)`. `capture.Mic` does this inside and reports through `MicOptions.Lost` when sound was lost all the same. The most it keeps is `MicOptions.Backlog` (a minute by default). **That is not a delay**: sound is passed on as soon as it is read, and nothing waits while the stages after it keep up |
| Video (old frames are of no use) | Pass on the newest only: `flow.Latest(ctx, camera)` |

- Seen on the machine: joined without keeping, the sound of the seconds in which the first utterance was transcribed and translated was lost; the second utterance lost its beginning and the third was gone. With keeping, all three came through
- `flow.Live` (a recording at a microphone's pace) only waits when its reader is late, and loses nothing. A recording does not show this difference
- Stopping the microphone and stopping the stages after it are two things. `capture.Mic` ends (without an error) when its ctx is cancelled, and the stages after it work off what was heard, so they are given a context of their own (`samples/interpreter` turns the microphone off at the first Ctrl-C and ends at the second)
- The camera (for the pseudo-3D sample): a frame of hwcamera in the driver's memory needs its `Release`, so frames are made pictures in Go's memory (`WithFormat(hwcamera.RGBA)`) before they go into a stream, and pass through `flow.Latest`. It is not written yet
- Outputs (a speaker, a screen) follow the same idea: the library has none ([D19](#10-design-questions))

### 6.6 Declaring the network

The way of writing so far joins streams with functions. The joins are in the code, and the library does not know them. A **network** is the same joins written as a structure: nodes joined with `Connect` and given to the app, which loads the models and runs it.

```go
var (
	stt = speech.Transcribe("transcribe")
	tr  = text.Translate("translate").At(16, "plamo3translate:2b").At(32, "plamo3translate:8b")
	tts = speech.Speak("speech")
)

mic   := flow.Source[[]float32]("mic")                             // an input: its stream comes later
vad   := speech.VADNode(speech.VADOptions{})                       // a node without a model
hear  := speech.TranscribeNode(stt, speech.TranscribeOptions{})    // nodes of models
trans := text.TranslateNode(tr, text.TranslateOptions{To: "en"})
say   := speech.SpeakNode(tts, speech.SpeakOptions{})
out   := flow.Sink[[]float32]("out").Do(play)                      // an output

// nodes of the program: they see what passes, change it, and decide whether it goes on
heard := flow.Tap("heard", func(c flow.Chunk[flow.Text]) { log.Println(c.Data.Text) })
commands := flow.Each("commands", func(c flow.Chunk[flow.Text], emit func(flow.Chunk[flow.Text])) error {
	switch c.Data.Text {
	case "that's all":
		return flow.Stop                                           // end here: the nodes before stop, those after finish
	case "switch to Spanish":
		trans.Update(func(o *text.TranslateOptions) { o.To = "es" }) // change another node's settings; pass nothing on
		return nil
	}
	emit(c)                                                        // pass it on as it is
	return nil
})

app := ai.NewApp("interpreter")     // its models are taken from the network's nodes
app.Connect(mic)
mic.Connect(vad)
vad.Connect(hear)
hear.Connect(heard)
heard.Connect(commands)
commands.Connect(trans)
trans.Connect(say, flow.Keep(4))    // how an edge passes chunks on: it keeps up to 4
say.Connect(out)

mic.From(capture.Mic(ctx, capture.MicOptions{}))   // the input's stream, given for the run
err := app.Run(ctx)                                 // loads the models, then starts reading the input
```

**The kinds of nodes**

| Node | Made with | What it is |
|---|---|---|
| Input | `flow.Source[T](name)`, `.From(stream)` | A stream. Declared by its name, given its stream before the run |
| Model | `speech.TranscribeNode(use, opts)`, `speech.SpeakNode`, `text.TranslateNode` | The model's stream form (`...Stream`). It has settings, which can be changed while it runs (`node.Update(func(o *Options))`) |
| Model (your own) | `ai.Node(use, fn)`, `ai.NewModelNode(use, opts, fn)` | `fn(ctx, model, in)` returns a stream: for a use of a model that its package does not have |
| Program | `flow.Each(name, fn, end...)` | `fn(c, emit)` is called with each chunk that comes. What it gives to `emit` goes on, nothing, one chunk or several. A node that keeps text in a variable and emits at the end of a sentence is written so. `end` is called when the input has ended, to emit what was kept |
| Program (looking only) | `flow.Tap(name, fn)` | Passes everything on as it is and shows it to `fn` (a log, captions on a screen, a counter) |
| Program (the whole stream) | `flow.Func(name, fn)` | `fn(ctx, in)` returns a stream: the "stage of your own" of section 6.4 as a node |
| The VAD, without a model | `speech.VADNode(opts)` | Finds the utterances |
| Output | `flow.Sink[T](name)`, `.Do(fn)` | `fn` for each chunk |

**Joining**

- `a.Connect(b)` compiles only when a's output type is b's input type
- **An output can go to several nodes** (one transcript to captions and to a translation). Each gets every chunk, and a slow reader does not hold the others back
- **A node has one input.** Connecting a second panics. A node with several inputs (a renderer that takes depth and the head's position) cannot be written yet ([D21](#10-design-questions))
- How an edge passes chunks on is `Connect`'s second argument

| Passing | Meaning | Where |
|---|---|---|
| (none) | Passed when the node after reads. While it is busy, the node before waits | Ordinary edges |
| `flow.Keep(n)` | Read in a goroutine of its own, up to n chunks kept. The node before does not wait. Nothing is dropped | Before the speaker (the next text is translated while it speaks) |
| `flow.Newest()` | The newest chunk only. What came while the node after was busy is dropped | A camera's frames into a detector |

**Running**

- `app.Connect(sources...)` gives the network to the app. The nodes' own `Connect`s may come before or after it
- **The app's models are taken from the network's nodes.** `app.Plan` and `APP models` find them without their being listed in `ai.NewApp(name, uses...)`. `APP models` shows the network too
- `app.Run(ctx)` is `Open`, loading the models, running the network and `Close`. On a `Runtime` that is open, `rt.Run(ctx)`
- **All the models are loaded before an input is read.** No sound of the microphone is lost in the seconds of loading
- How it ends: `nil` once every input has ended and the last chunk has gone through; an error with the node's name (`*flow.NodeError`) when a node fails; ctx's error at its cancel
- **A node ending it from inside**: when the function of `flow.Each` returns `flow.Stop`, its node ends there. The nodes before it stop (the microphone is closed), the nodes after it work off what they were given, and `Run` returns `nil`
- One model in two places is two nodes (`speech.TranscribeNode(stt, optsL)` and `speech.TranscribeNode(stt, optsR)`). There is one model, and its calls go through one after another (section 9.3)

**And the stream functions**: they stay. A node's inside is stream functions, and an edge's way of passing is `flow.Async` and `flow.Latest`. The network is a layer that puts them together and runs them. Calling a model once a request (a server, batch work), and anything that needs finer control, is written with `Get` and the stream functions as before.

## 7. The models

Every model has a batch form and a stream form.

### 7.1 `ai/speech`

| | Batch | Stream |
|---|---|---|
| `Transcriber` (Whisper) | `Transcribe(ctx, pcm, TranscribeOptions) (Transcript, error)`. Speech longer than 30 seconds is read in windows | `TranscribeStream(ctx, Stream[[]float32], opts) Stream[Text]`. Reads the chunks up to each final one together and gives their text as a final chunk |
| `Speaker` (LFM3-Audio) | `Speak(ctx, text, SpeakOptions) (Audio, error)` | `SpeakStream(ctx, Stream[Text], opts) Stream[[]float32]`. Speaks each final text and gives a chunk of audio, timed by the output's samples |
| `VAD` (by loudness) | `Segments(pcm) []Span` | `Detect(ctx, Stream[[]float32]) Stream[[]float32]`. Passes the speech alone, with `Final` at the end of each utterance |

- `TranscribeOptions{Language, Translate, MaxTokens}`: an empty language is found in each 30 seconds. `Translate` writes English
- `SpeakOptions{Language, Voice, Seed, MaxFrames}`: `SpeakStream` takes a chunk's `Text.Lang` when the language is empty, and one seed for the whole stream (the voice stays the same)
- `VADOptions`: 12 dB above the background noise, 600 ms of silence to end an utterance, 250 ms at least, 200 ms of padding, 25 s at most (all can be set). It uses no model (`speech.NewVAD`; nothing to register)
- Not there yet: provisional captions while someone speaks, speech that comes out as it is made, a VAD by a model
- `speech.ReadWAV(data, rate)`, `speech.WriteWAV(w, pcm, rate)`

### 7.2 `ai/text`

| | Batch | Stream |
|---|---|---|
| `Translator` (PLaMo 3 Translate) | `Translate(ctx, text, TranslateOptions) (string, error)` | `TranslateStream(ctx, Stream[Text], opts) Stream[Text]`. Translates each final text, with the 4 pairs before it as its history |
| `Chatter` (a chat model) | `Complete(ctx, []Message, ChatOptions) (string, error)` | `CompleteStream(ctx, msgs, opts) Stream[Text]`. "The answer so far" after each token as provisional chunks, then the whole answer as the final one |

- `TranslateOptions{To, From, Form, Context, History, Glossary, MaxTokens}`: an empty `To` is English for a text with Japanese characters and Japanese for any other
- `ChatOptions{MaxTokens, Think, ThinkBudget, Temperature}`
- Not there yet: tool calls, images and audio as input, answers in a given form (JSON Schema), embeddings

## 8. The app's command line (`ai/aicli`)

```go
var cli struct {
	Run    RunCmd       `cmd:"" default:"withargs" help:"..."`
	Models aicli.Models `cmd:"" help:"The models the app uses: list, download, prepare."`
	Status aicli.Status `cmd:"" help:"This machine's NPU and what uses it."`
}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	k := kong.Parse(&cli, aicli.Bind(ctx, app))
	k.FatalIfErrorf(k.Run())
}

// the app's own commands may take ctx and app too
func (c *RunCmd) Run(ctx context.Context, app *ai.App) error { ... }
```

| Command | What it does |
|---|---|
| `APP models` | The models the app uses: their state (missing; ready, not compiled; compiled; bundled), bytes to download, memory, license, what they take together and the machine's memory, why the app's line-up picked them. `--json` |
| `APP models download` | Downloads and converts. No NPU is needed. `--accept-license`, `-q` |
| `APP models prepare` (`compile`) | As download, then loads each model once to compile it |
| `APP status` | `hwnpu status`: the device, its driver, how busy it is, its power, the processes that use it, the cache. `-w 2s`, `--json`, `--wake` |

```
$ interpreter models
interpreter's models, for this machine's NPU (amd-xdna2)

USE         ROLE        MODEL                   STATE     DOWNLOAD  MEMORY  LICENSE
transcribe  transcribe  whisper-large-v3-turbo  compiled  -         2.2 GB  mit
translate   translate   plamo-3-translate-2b    missing   -         2.7 GB  plamo-community-license (conditional)
speech      speak       LFM3-Audio-2B-TTS       compiled  -         2.9 GB  lfm-open-1.0 (conditional)

Their checkpoints are here or come from the sources; about 7.8 GB running together, of this machine's 16 GB. ...
translate: plamo3translate:2b on machines of 16 GB and up, by the app; ...
```

## 9. Constraints and rules

There is one NPU, its memory is pinned, and its programs have fixed shapes. Much of an app's design follows from that. The numbers say where they come from: **"measured" means on a Ryzen AI 9 365 (AMD, XDNA 2) on 2026-11-21, through this library**; the others are records of design.md and MODEL.md. Nothing was measured through this library on a Mac or on Intel yet.

### 9.1 The NPU is one queue

**Different models can be called at the same time. It is not faster.** The time for all of them is the sum of their times alone.

| Call (measured, AMD) | Alone | Called at the same time |
|---|---|---|
| Speech to text (13.3 s of audio) | 1.4 s | 3.1 s with the speaker |
| Translation (8B, 3 sentences) | 3.3 s | 3.8 s with the speaker |
| Text to speech (3.9 s of speech) | 10.1 s | 11.4 s with the transcriber, 13.4 s with the translator |
| All three | 14.8 s added up | 14.7 s until all are done |

- The calls take turns in small pieces (the transcription above ended 3.1 s into the speaker's call). A call does not wait for the whole of another
- **Nothing orders the calls yet.** There is no way to say that audio goes first or that background work waits ([D10](#10-design-questions)). How long a short call waits in the middle of heavy work (a step of image generation takes one to several seconds) was not measured
- AMD: each program of a model has a hardware context, and the firmware shares the time between them (design.md 16.94). A switch costs about 0.5 ms
- Apple: switching programs costs 15 to 35 ms each time (design.md 16.36), which counts for a model that calls several programs a frame (the speaker). How several models of one process behave when called at once was not measured
- Intel: how models behave when run at once is not measured (design.md 16.94)

### 9.2 Can the models work together in real time?

"Running at the same time" and "keeping up in real time" are two things. The NPU being one queue, a pipeline keeps up only **when the time its stages take on the NPU adds up to less than a second for each second of real time**. Over that, what waits grows.

| Stage (measured, AMD) | Time on the NPU |
|---|---|
| Speech to text (whisper-large-v3-turbo) | About 1 to 1.4 s an utterance (up to 30 s; the window is always 30 s, so a short one takes little less) |
| Translation (plamo-3-translate-8b / 2b) | About 2.7 s / about 1.1 s for an utterance of one or two sentences |
| Text to speech (LFM3-Audio-2B-TTS) | About 2.6 s for a second of speech (about 0.4 times real time) |

An example: interpreting someone who speaks for 4 s and pauses for 1 s (an utterance every 5 s).

| Set-up | NPU time an utterance | Of 5 s | What happens |
|---|---|---|---|
| Captions only (speech to text + the 2B) | About 2.1 s | 0.4 | Keeps up. 2 to 3 s behind the end of an utterance |
| Captions only (the 8B) | About 3.7 s | 0.7 | Keeps up, with little to spare |
| With the translation spoken | About 13.6 s (10 s of it the speaker) | 2.7 | Does not keep up. It falls further behind with every utterance |

The delays measured (the recording paced as a microphone, counted from the end of each utterance; the `+N.Ns` that `samples/interpreter --live` prints):

| | First utterance | Third utterance |
|---|---|---|
| Text | +1.6 s | +5.4 to 5.7 s |
| Translation (8B / 2B) | +4.3 / +2.7 s | +8.7 / +7.9 s |
| Spoken translation (8B / 2B) | +20 / +14 s | +24 / +19 s |

- The +1.6 s of the text is 0.6 s of silence that ends an utterance (`VADOptions.Hangover`) and about a second of transcription
- Later utterances are later because they share the NPU with the speech of the one before
- Records on a Mac (alone): Whisper reads a 30 s window in 0.6 s on an M3; PLaMo 3 Translate on an M3 runs at 53 (0.8B), 17 (2B) and 10 to 11 (8B) tokens a second; the speaker at about 0.5 times real time in lut8 on an M3 (design.md 16.31, an old measurement). The speaker does not reach real time there either

**What to choose when the sum is over 1**

| Case | Design |
|---|---|
| Old input is of no use (a camera's frames into a detector) | Drop: `flow.Latest`. The detector sees only the present, and skips frames |
| The output's times are fixed (drawing) | On the beat: `flow.Metronome` and `flow.Hold`. A model's result replaces the one before when it comes |
| All of it is needed, late is fine (transcribing a recording, a translation to read later) | Keep: `flow.Async`. What waits grows, and nothing is lost |
| All of it is needed and late is not fine (a conversation) | **Take turns** (not listening until the answer is spoken; listening only while a key is held). Or make it lighter: smaller models (`use.At`, `use.As`), shorter outputs, fewer stages (captions instead of speech) |

- Even under 1, the delay is "until the utterance ends" plus each stage's time. Provisional captions while someone speaks, and speech that plays as it is made, are not there yet
- There is no API yet for an app to measure what the sum is on its machine ([D20](#10-design-questions)). For now, measure each stage alone as above and add them up

### 9.3 Several inputs into one model

- **A model runs one request at a time.** Calls made at the same time go through one after another
- **Calls are independent.** `Transcribe`, `Translate`, `Speak` and `Complete` do not depend on the call before (a translation's history and a conversation's messages are given by the caller each time). So several inputs can take turns through one model
- **Left and right audio**: inputs are mono (a transcriber and a VAD read mono at 16 kHz). Stereo is split into two mono streams, each with its own VAD, into the same transcriber. Measured: two clips of 13.3 s at the same time take 2.8 s (twice the 1.4 s of one). Both count in the sum of section 9.2
- **Loading the same model twice is not faster**: the NPU being one queue, the total is the same and only the memory doubles (loading two was not measured)
- **Hardly anything takes several inputs in one call (a batch).** Programs have fixed shapes: one 30 s window for speech to text, one sequence for chat and translation, one text at a time for embeddings. The exceptions are of pictures: image generation draws a few together (`hwnpu-image --batch`), and labels are read 8 texts at a time
- A chat model skips reading again what continues the conversation before (or one of the two long ones before that); the result is the same. With more conversations taking turns through one model, that saving is lost (it is only slower)

### 9.4 Memory

**The NPU's buffers are pinned memory that cannot be swapped out.** Over what there is, other applications are pushed to swap. Models loaded together add up (nothing is shared).

| The machine's memory | 8 GB | 16 GB | 24 GB | 32 GB | 48 GB | 64 GB and up |
|---|---|---|---|---|---|---|
| About what is left for models (MODEL.md) | 3 GB | 9 GB | 15 GB | 22 GB | 35 GB | 50 GB |

- Compare `plan.MemoryGB()` (the models loaded together) with `plan.UsableGB()` (the table above). `plan.Fits()` and `APP models` say whether it is over
- An example (the interpreter, AMD's figures): 2.2 GB for speech to text and 2.9 GB for the speaker, plus 1.0 (0.8B), 2.7 (2B) or 6.5 GB (8B) for the translator. The 8B does not fit a 16 GB machine (about 9 GB)
- **How memory is counted**: on Apple the Neural Engine's share is not in the process's memory (its footprint); it is the kernel's wired memory, readable for the whole machine only. On AMD the NPU's buffers are not in the process's RSS either. models.yaml's `memory` is both together

The limits known by chip:

| Chip | Constraint | Source |
|---|---|---|
| M1 (the Pro, Max and Ultra are taken to be alike) | With over 2.8 GiB of programs loaded on the Neural Engine together, every call slows down about 12 times. Large models (PLaMo 3 Translate 8B and others) are not picked by a use (`not_on: [M1]`) | `m1HoldBytes` in `internal/chat/model.go`, PROBE.md |
| M2 and later | No such limit was found (3.65 GiB at full speed on an M2) | The same |
| Apple, all | The Neural Engine's cache of compiled programs holds about 4 GiB. What is over it is compiled again at every start (minutes). Compiling is one serial queue for the whole machine | design.md 16.99, 16.229 |
| AMD (XDNA 2) | No limit of memory of the NPU's own was found (its buffers are host memory). The models share the DRAM's bandwidth (about 55 GB/s), so those that read large weights slow each other down | design.md 16.94 |
| Intel | On a Core Ultra 5 135U, Anima in fp16 failed with `OUT_OF_HOST_MEMORY` at about 4.6 GB. The limit of a program's weights is not known | design.md 16.144, intel-npu-models.md |

### 9.5 `Release`, and image generation swapping its stages

Both keep things from being in memory together, **at different levels**.

| | Image generation swapping its stages | `use.Release(rt)` |
|---|---|---|
| What | The stages inside one model (text encoder → DiT → VAE), and the groups of blocks inside the DiT | A use (a model), whole |
| Who does it | The pipeline itself (on by default) | The app calls it |
| The next use | Each stage is loaded again by itself | The next `Get` loads all of it again |
| Where it is now | `internal/imagegen` (only the commands use it) | `ai` |

Image generation's swapping is built differently for each NPU (figures of the records).

- **AMD**: the three stages never run together, so each is on the NPU only while it runs. For Anima at 1024², the peak goes from 14.2 to 5.0 GiB; loading costs a few seconds a picture. The DiT reads each group's weights into one bank (pinned memory 7 × 464 → 464 MiB; `--resident-dit` keeps them all)
- **Apple (Core AI)**: a group of the DiT is loaded when its turn comes, run, closed, and the next one is loaded ahead. For Anima-Turbo the Neural Engine holds 0.57 instead of 2.53 GiB, and a picture takes the same 49 s (M1). For Chroma it is 2.2 to 2.6 instead of 8.3 GiB, and swapping is the faster
- **Intel**: holds what fits, and closes what was not used lately to load again when it does not

- When image generation comes into `ai/imagegen`, it brings this swapping as it is, on by default. An app then calls `Release` after generating, to unload what is left (small parts), and goes on to the next model (depth)
- A third mechanism: a chat model on AMD gives its pinned memory back after a while without a request and loads again in a second or two at the next one (`hwnpu-chat --idle`). It cannot be used from `ai` yet
- Making one thing of the three (a budget of memory, what may be evicted) is [D9](#10-design-questions)
- The time to load again (measured, AMD): speech to text 0.5 to 1.2 s, the translator 5 to 9 s (8B) or about 3 s (2B), the speaker 2.3 to 3.8 s. On Apple it is seconds once compiled, and tens of seconds to minutes the first time

### 9.6 Processes

- **On a Mac, one hwnpu process uses the Neural Engine at a time.** Two at once froze an M1's OS, so the process that first opens the device holds a lock until it ends (design.md 16.132). While an app that stays running runs, commands such as `hwnpu-chat` wait. There is no API yet for an app to let the NPU go (`Suspend`)
- AMD and Intel have no such lock. On AMD, processes can use the same NPU, sharing its time (commands were up to 50 times slower with another process on the NPU: new-model-checklist.md). Intel was not tried
- On a Mac, macOS's own services (photo analysis, Apple Intelligence) use the Neural Engine too. While they ran, chat fell to 0.7 tokens a second (design.md 16.112). An app cannot stop them
- **Take a process to have one `App`.** The store's settings are one per process ([D8](#10-design-questions))
- Several models in one process on AMD need the fix of the device heap's address (design.md 16.281)

### 9.7 The shapes of inputs and outputs

| | Shape |
|---|---|
| Input of speech to text and of the VAD | Mono, 16 kHz, float32 (−1 to 1). Windows of 30 s. The VAD ends an utterance at 25 s (`VADOptions.MaxSpeech`) |
| Output of the speaker | Mono, 24 kHz. A sentence is given whole (it cannot be played from its middle on) |
| Translation, chat | One text at a time. A chat's context length is the model's definition's |
| Pictures (planned) | Only the sizes of the export (fixed sizes by aspect ratio) |

- Sound of another rate or channel count is matched before it goes into a stream (`speech.ReadWAV(data, rate)` reads a file so; hwcamera's microphone has the OS do it)

## 10. Design questions

For each: what it is now, why, and the other options. Those **not decided** are marked.

### Registering and getting

**D1. The typed handle `Use[T]`** — now: `stt := speech.Transcribe("transcribe")`, and `stt.Get(ctx, rt)` returns a `*speech.Transcriber`.

- Why: a wrong type is a compile error. `ai` does not import the packages of models, so an app links only the code of the models it uses
- Option A: `rt.Transcriber(ctx, "transcribe")` (the branch before). Shorter, but `ai` imports every package of models and the name is written in two places
- Option B: a function `ai.Get(ctx, rt, stt)`. Go's methods cannot have type parameters, so `rt.Get(stt)` cannot be written
- **Not decided**: whether the word order of `stt.Get(ctx, rt)` (the handle as the subject) is right; whether keeping `Use`s in package variables is the way to write an app

**D2. The functions' names** — now: `speech.Transcribe` (a verb) returns a `Use`, and the type is `Transcriber`. `text.Chat`'s type is `Chatter` (the function takes the name).

- **Not decided**: whether the functions that make handles should be `speech.TranscribeUse("...")` or `speech.Transcriber.Use("...")`; the name `Chatter`

**D3. The name is always given** — now: `speech.Transcribe("transcribe")`. There is no default name.

- Option: the role's default use without an argument (`speech.Transcribe()` is `transcribe`). Shorter, but two uses of the same role are then harder to tell apart

**D4. Two models of one role** — now: different names are different uses (`text.Chat("chat")` and `text.Chat("code")`). The store's key is `ROLE:NAME`.

- One name in two line-ups (`translate` as a large one and as a small one) cannot be written. A definitions file of the app's with two uses' names can
- **Not decided**: whether a `Use` needs a name of its own (`text.Translate("translate").Named("subtitle")`)

### Picking models

**D5. How a line-up by memory is written** — now: `At(gb, name)` and `As(gb)` on each use.

- Option A: a table on the app, `app.LineUp(16, tr.Is("plamo3translate:2b"), ...)`. The whole line-up of a machine's memory reads in one place, at the cost of more API
- Option B: uses in a definitions file of the app's (section 3.3, possible now). Can be replaced after shipping without changing code, but the typed handles are tied to it by name only
- Option C: pick smaller models automatically until the total fits the machine's memory. The app knows which model is the main one, so it would have to give priorities (`use.Priority(n)`)
- **Not decided**: whether to add A or C; the tiers of `At` are cut by the machine's memory, and whether some should be cut by the memory that is free

**D6. The store when the line-up changes** — now: a record of another model than the one picked now is `ErrNotPrepared`. There is one record a use, so going back and forth between memory settings prepares again each time (the files are still there, so only what is missing is made).

- Option: several records a use. Going back and forth is fast, with a larger `store.json`

### Preparing

**D7. The two stages' API** — now: `plan.NoCompile = true`, then `plan.Run`.

- Option: `plan.Download(ctx, fn)` and `plan.Compile(ctx, fn)`. No negative flag, and it reads better
- **Not decided**: which; whether `AcceptLicenses` stays a flag or becomes `plan.Accept(licenses...)`

**D8. The store's settings are one per process** — now: `App.Plan` and `App.Open` call `modelstore.Open`, which replaces the process's settings (where the cache is, the sources, the definitions).

- It shows what is underneath (`internal/cache`, the table of definitions in `internal/chat`), which is one per process
- **Not decided**: whether to say that a process has one `App`; or to carry the store as a value (a large change underneath)

### Running

**D9. Memory is left to the app** — now: `Get` loads and `Release` unloads. There is no budget and nothing is evicted.

- Option: a budget in the `Runtime`, which unloads models that are not in use when it runs short (design.md 16.94). A `Use` says whether it may be unloaded (`use.Resident()`, `use.Evictable()`). The value an app holds must then be an indirection, so that it survives its model being unloaded
- **Not decided**: when. It changes the rule of how long a value from `Get` lives

**D10. Concurrency** — now: one request at a time for each model. Nothing orders the models among themselves (the NPU's driver does). Each model opens its own device.

- **Not decided**: an API of priorities (audio first, background work waits). On a Mac the process holds the Neural Engine's lock while it lives, so an app that stays running makes the other hwnpu commands wait

### Streams

**D11. A stream is an `iter.Seq2`, audio is `[]float32`** — now: a pull iterator. The sampling rate is not in the chunk; the stage decides it.

- Why: `range`, `break` and ctx are all it takes, and a stage is a function. Goroutines are only where they are needed (`Async`, `Latest`, `Hold`)
- Option A: channels. `select` waits on several streams, but closing and passing errors must be decided each time
- Option B: audio as `flow.Audio{Samples, Rate, Channels}`, with the rates matched when stages are joined (16.94's "ports declare their type and constraints"). A wrong join is found when it runs
- **Not decided**: whether to add B; the type of pictures (`image.Image` as it is, or a frame with its time); stereo

**D12. `Final` means two things** — now: for audio "the end of an utterance", for text "a settled sentence" (those before it are provisional and replaced by the next).

- **Not decided**: whether provisional and the end of a segment should be two fields (`Chunk.Partial` and `Chunk.End`); whether provisional text should be a difference instead of the whole text again

**D13. Control** — now: cancelling by ctx only.

- **Not decided**: an API to interrupt (stop speaking and go on to the next), to pause, and to change settings in the middle of a stream (the language to translate into). Whether they come as another stream (`tts.SpeakStream(ctx, in, opts, flow.Interrupt(events))`)

**D14. Clocks** — now: each stream has its own times. `Metronome` runs on the wall clock, and no clock is shared between streams.

- **Not decided**: what audio and video are aligned to (the player's clock?); a `Metronome` whose beats are counted in samples of audio

### The rest

**D15. Progress** — now: the stage and a line of the log (`Progress.Message`). No bytes or percentages, because the `Prepare` underneath downloads and converts in one function and only writes a log.

**D16. How settings are given** — now: a struct for the settings of a call (`TranscribeOptions{...}`), functions for those of `Open` (`ai.WithLog(w)`). The settings of a conversion (the weights' format, the context's length) are models.yaml's and cannot be changed from code.

- **Not decided**: whether the settings of a conversion can be given when registering (`text.Chat("lfm3").Context(8192)`). Moving the commands onto the library needs it

**D17. kong alone** — now: `aicli` is kong structs.

- Option: the same as functions too (`aicli.WriteModels(w, plan)`, `aicli.RunPrepare(ctx, app, opts)`), to be called from other command line libraries and from GUIs
- **Not decided**: whether to add the functions

**D18. The packages' names and split** — now: `ai`, `ai/speech`, `ai/text`, `ai/flow`, `ai/aicli`. Planned: `ai/vision`, `ai/imagegen`, `ai/sound`.

- **Not decided**: the name `ai` (hard to change once it is public); `speech` holding both speech to text and text to speech; the name `text` (easily confused beside the standard library's `text/...`)

**D19. Input and output devices** — now: the library has none. It takes streams and gives streams. The samples make streams of the microphone (and the camera) with hwcamera (`samples/capture`).

- Why: the library's dependencies do not grow. An input from a file or a network has the same shape
- Option: the code that makes streams of hwcamera as another module of hwnpu (not a dependency of the library). Apps then do not each write it, and the rules of "keep" and "the newest only" are in one place
- **Not decided**: whether to make that module; where an API for playing sound goes

**D20. An API to measure whether a pipeline keeps up, and one to order the calls** — now: neither. An app measures each stage alone and adds them up (section 9.2).

- Option A: the `Runtime` counts each model's calls and their time and gives them in `rt.Loaded()`. An app sees what the sum is on its machine, and changes its line-up or tells the user
- Option B: a command such as `APP models bench` that measures the registered models on fixed inputs and gives the machine's figures
- Option C: a priority on a use (`use.Priority(ai.Interactive)`): audio goes first and background work waits at its breaks. The calls to the NPU must then go through one place (D10)
- **Not decided**: which to add. A is small; C belongs with managing what stays loaded (D9)

**D21. Declaring the network (`Connect`)** — now (v0 added on 2026-11-21, section 6.6): nodes are joined with `Connect` and given to the app with `app.Connect(sources...)`. Nodes of models, nodes of the program (`flow.Each`, `flow.Tap`, `flow.Func`), inputs and outputs. The app's models are taken from the network, and `app.Run` loads them and runs it.

- Why: once the library knows the joins, it can estimate whether the pipeline keeps up (section 9.2), measure each edge (D20), have each edge say how it passes chunks on, estimate the memory and do the `Release`s (D9), and show the network. A fan-out is written without `Tee`
- Decided:
  - **A model and its place in the network are apart**: a `Use` is the model, `speech.TranscribeNode(use, opts)` its place. One model can be in two places
  - **Joined node to node** (`a.Connect(b)`): a node has one input and one output. An output can go to several nodes
  - **A node has its settings**: `node.Update(func(o *Options))` changes them while it runs too (one answer to D13's "changing settings in the middle")
  - **Listing stays**: `ai.NewApp(name, uses...)` and `Get`, for calling a model once a request
  - **The stream functions stay as the layer below**: a node's inside and an edge's way of passing are made of them
- **Not decided**:
  - **Nodes with several inputs** (a fan-in): a renderer that takes depth and the head's position, a speaker that takes text and an interruption. Perhaps nodes with ports: `depth.Connect(&render.Depth)`, `faces.Connect(&render.Head, flow.Newest())`. The pseudo-3D sample needs it
  - **Nodes with several outputs**: provisional captions and settled sentences, depth and confidence
  - **A "when finished" edge** (`flow.Whole()`): when the node before has ended, its model is unloaded before the next is loaded. It leaves `Release` to the library (D9)
  - **Measuring each edge**: the chunks that passed, those waiting, the delay (D20). With the network there is one place to add it
  - **Nodes' names**: a model's node is named after its use, so two nodes of one model have the same name
  - **Changing the joins while it runs**: not possible now

## 11. Not there yet

- Pictures: depth, faces, objects, poses, cutouts, OCR, labels (`ai/vision`); generation (`ai/imagegen`)
- Sound: music, sound effects, pitch, notes (`ai/sound`)
- Moving the commands: making `hwnpu-audio`, `hwnpu-chat` and `hwnpu-image` only call this library
- Managing what stays loaded (D9, D10), a local server (OpenAI compatible and others), bundles to ship, build tags
- Their order and details are in [app-api-tasks.md](app-api-tasks.md)

## 12. For hwnpu's developers: adding a package of models

A package of models has a `Role` of `internal/airole` and makes its `Use` with `ai.Define`. `ai` knows nothing else of it.

```go
// ai/vision/depth.go (a sketch)
package vision

func Depth(name string) ai.Use[*DepthEstimator] {
	return ai.Define[*DepthEstimator](depthRole, name)
}

var depthRole = &airole.Role{
	Name:  "depth",
	Kinds: []chat.Kind{chat.KindDepth},
	// prepare: what the command does on its first run. It returns the files Load reads
	Prepare: func(e airole.PrepareEnv) (f airole.Files, err error) {
		dir, err := ...(e.Ctx, e.Model, e.NPU, e.Log)
		f.Set("dir", dir)
		return f, err
	},
	// load: from the files alone. No network, no conversion
	Load: func(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
		...
		return &DepthEstimator{...}, nil
	},
}
```

- **`Prepare` and `Load` are apart**: `Prepare` downloads and converts, and `Load` reads only the paths written in `Files`. Those paths go into the store's `store.json` relative to the cache's root, so that what was prepared can be copied elsewhere and shipped
- **Call the same functions as the command, with the same settings**: the programs then have the command's names and places, and each uses what the other prepared. Steps that are inside a command (`cmd/...`) move into `internal` first
- **What differs by NPU** is read from `e.NPU`: an export of IRON when `e.NPU.AMD()`, else programs in `e.NPU.Format` (`coreai`, `intel`)
- **Give both a batch form and a stream form.** The stream form may start as a batch call at each segment (`Final`)
- **A model's value is an `io.Closer`** and runs one request at a time (it locks inside)
- **Tests** can go from the plan to the load without an NPU, with a fake `Role` (`ai/ai_test.go`). What depends on time uses `testing/synctest`
- A module outside cannot import `internal/airole`, so only packages inside hwnpu can make a `Use`
