// Package ai puts hwnpu's models into apps: an app says which models it uses, makes them ready
// ahead of time, and runs them on the NPU, several together.
//
// It is v0: its shape may change until a second sample (a pseudo-3D photo) runs on it
// (docs/app-api.md).
//
// The models come from the packages of models, each a typed Use an app registers:
//
//	var (
//		stt = speech.Transcribe("transcribe")
//		tr  = text.Translate("translate").At(16, "plamo3translate:2b").At(32, "plamo3translate:8b")
//		tts = speech.Speak("speech")
//
//		app = ai.NewApp("interpreter", stt, tr, tts)
//	)
//
// The names are models, families and uses of the definitions (models.yaml; hwnpu-chat list
// shows them). A use picks its model for the machine's memory; At and As set the app's own
// line-up by memory, for models that run beside each other.
//
// Preparing and running are apart. An app sees what its models take, prepares them, and then
// runs them without touching the network or converting anything:
//
//	plan, err := app.Plan(ctx) // downloads nothing
//	fmt.Println(plan.DownloadBytes(), plan.MemoryGB(), plan.Licenses())
//	err = plan.Run(ctx, func(p ai.Progress) { ... }) // download, convert, compile for the NPU
//
//	rt, err := app.Open(ctx)
//	defer rt.Close()
//	t, err := stt.Get(ctx, rt) // a *speech.Transcriber; ErrNotPrepared when Run has not prepared it
//	res, err := t.Transcribe(ctx, pcm16k, speech.TranscribeOptions{Language: "ja"})
//
// The cli package gives an app's command line the commands that do the preparing (models,
// models download, models prepare) and show the NPU (status). The flow package joins the models:
// each has a batch form and a stream form, and flow passes one's chunks to the next when it has
// finished, as they come, or on the beat.
//
// The models' downloads and programs are those of hwnpu's commands (hwnpu-chat, hwnpu-audio),
// for the same NPU, through the same definitions and layers (Store.Models, modelstore): what
// the commands prepared is used as it is, and what an app prepares they use.
//
// The planning, the preparing, the compiling and the models run in hwnpud (cmd/hwnpud,
// docs/design/hwnpud.md) unless HWNPU_DAEMON is off: the app's code is the same, its models'
// methods are called over hwnpud's socket, and Apple's compiler keeps one cache for every
// program. A test, and HWNPU_DAEMON=off, run them in the app's own process.
package ai
