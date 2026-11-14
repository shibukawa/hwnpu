// Package ai puts hwnpu's models into apps: speech to text, translation and speech, with the
// models made ready before they run.
//
// It is v0: its shape may change until two samples (an interpreter, a pseudo-3D photo) run on it
// (docs/app-api.md).
//
// Preparing and running are apart. An app says what it uses, sees what that takes, prepares it,
// and then runs it without touching the network or converting anything:
//
//	reg := ai.NewRegistry().Require("transcribe", "translate", "speech", "vad")
//	plan, err := reg.Plan(ctx, store) // downloads nothing
//	fmt.Println(plan.DownloadBytes(), plan.MemoryGB(), plan.Licenses())
//	err = plan.Run(ctx, func(p ai.Progress) { ... }) // download, convert, load once
//
//	rt, err := ai.Open(ctx, store) // ErrNotPrepared for a use Run has not prepared
//	defer rt.Close()
//	stt, err := rt.Transcriber(ctx, "transcribe")
//	text, err := stt.Transcribe(ctx, pcm16k, ai.TranscribeOptions{Language: "ja"})
//
// Each model has a batch form (one value in, one out) and a stream form: a Stream of chunks in,
// a Stream out, read with range.
//
//	speech := ai.NewVAD(ai.VADOptions{}).Detect(ctx, mic)     // the utterances
//	texts := stt.TranscribeStream(ctx, speech, sopts)         // each utterance's text
//	en := tr.TranslateStream(ctx, texts, topts)               // its translation
//	audio := tts.SpeakStream(ctx, ai.Async(ctx, en, 4), popts) // its speech, translating ahead
//	for c, err := range audio { play(c.Data) }
//
// Uses, families and models are those of models.yaml (hwnpu-chat list); a use picks its model
// for the machine's memory when the plan is made, and the store remembers the pick, so that the
// model prepared is the model run.
//
// The models' downloads and programs are hwnpu-chat's and hwnpu-audio's: DefaultStore is their
// directory, and what they prepared is used as it is (a Run then only checks it and writes it
// into the store's StoreFile).
package ai
