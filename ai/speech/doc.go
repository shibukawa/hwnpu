// Package speech has the models of speech for an app (ai.App): speech to text (Transcribe),
// text to speech (Speak), and a detector of where the speech is in audio (VAD).
//
// Each model has a batch form (one value in, one out) and a stream form that reads and gives
// flow.Streams, to be joined with the other packages' models:
//
//	var (
//		stt = speech.Transcribe("transcribe")
//		tts = speech.Speak("speech")
//	)
//	app := ai.NewApp("echo", stt, tts)
//	...
//	t, err := stt.Get(ctx, rt)
//	s, err := tts.Get(ctx, rt)
//	utterances := speech.NewVAD(speech.VADOptions{}).Detect(ctx, mic)
//	texts := t.TranscribeStream(ctx, utterances, speech.TranscribeOptions{})
//	for c, err := range s.SpeakStream(ctx, texts, speech.SpeakOptions{}) { play(c.Data) }
//
// Audio is mono float32 in [-1, 1]: at TranscribeRate (16 kHz) into the transcriber and the VAD,
// at Speaker.Rate (24 kHz) out of the speaker.
package speech
