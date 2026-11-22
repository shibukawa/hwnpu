// Package sound has the models of music and sound for an app (ai.App): music in a style
// (Music: Magenta RealTime 2), sound effects from a description (Effect: MOSS-SoundEffect), the
// pitch of a voice or an instrument (Pitch: SwiftF0) and the notes of music (Notes: Basic
// Pitch).
//
//	var (
//		mus = sound.Music("music")
//		sfx = sound.Effect("sfx")
//	)
//	app := ai.NewApp("jukebox", mus, sfx)
//	...
//	g, err := mus.Get(ctx, rt)
//	style, err := g.Style(ctx, sound.Style{Text: "disco funk"}, sound.Style{Text: "jazz piano", Weight: 2})
//	music, stats := g.Generate(ctx, sound.MusicOptions{Codes: style.Codes, Duration: time.Minute})
//	for c, err := range flow.Pace(ctx, music, time.Second) { play(c.Data) } // as it is made
//	fmt.Println("seed", stats.Seed)
//
// Music comes as it is made, a second at a time, 48 kHz stereo with its samples interleaved:
// faster than real time on the NPU, so that a player paced by flow.Pace (or the wall clock of
// its own) has the next second before it needs it. Music made on the beat of a Metronome, and
// a style that changes while it plays, are to come.
//
// Sound effects, the pitch and the notes are made whole, one value in and one out. The pitch
// and the notes are small models that run where the definitions say (the CPU beside AMD's
// NPU, with ONNX Runtime when it is here).
//
// Each result carries what making it took (the seed used, the times of its stages), for an app
// that shows or logs it. The roles that check where their programs run refuse programs the
// NPU's compiler placed off the NPU unless the Runtime allows them (ai.AllowOffNPU).
package sound
