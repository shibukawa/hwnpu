// Package flow passes data between models: what one makes goes to the next as a stream of
// chunks with their times.
//
// A Stream is an iterator (iter.Seq2) of chunks, read with range. Every stage of a pipeline is
// a function from streams to a stream, so stages are joined by passing one's result to the
// next, and the whole stops at a break, an error, or the cancel of its context.
//
// Stages are joined in three ways, by when the next one may start.
//
// When the one before has finished. The next stage needs the whole result (a picture, then its
// depth). Last reads a stream to its end; a stage's batch call (one value in, one out) is the
// same thing, and Of gives a whole value to a stage that reads a stream.
//
//	img, err := gen.Generate(ctx, prompt)   // all the steps
//	depth, err := est.Estimate(ctx, img)    // then its depth
//
// As the chunks come. The next stage starts on each chunk while the one before makes the next.
// Async lets stages overlap without losing a chunk (sentences to translate, then to speak);
// Latest drops what went stale while the reader was busy (camera frames into a detector); Hold
// keeps the newest for a stage that asks when it wants (the head's position for a renderer);
// Tee gives one stream to several readers.
//
//	speech := vad.Detect(ctx, mic)
//	texts := stt.TranscribeStream(ctx, speech, sopts)
//	en := tr.TranslateStream(ctx, texts, topts)
//	audio := tts.SpeakStream(ctx, flow.Async(ctx, en, 4), popts)
//
// On the beat. The stage owes its output at fixed times (music for a player, frames for a
// screen). A Metronome gives a tick a beat, a Lead before it is due; Pace hands chunks to a
// sink when their times come; Live makes a recording arrive as a microphone would give it.
//
//	head := flow.Hold(ctx, faces.DetectStream(ctx, flow.Latest(ctx, camera)))
//	for tick := range (flow.Metronome{Period: time.Second / 60, Skip: true}).Ticks(ctx) {
//		pos, _ := head.Latest()
//		draw(mesh, pos.Data, tick.Data.At)
//	}
//
// The joins can be written as a structure too, a network: sources, nodes and sinks joined with
// Connect, each edge saying how it passes chunks on (Keep, Newest), run by Run. A node is a
// function over streams (Func), a callback that sees each chunk and emits what goes on (Each,
// Tap), or a model's (the packages of models make them). What runs the network then knows its
// models and how they are joined.
//
//	mic.Connect(vad)
//	vad.Connect(hear)
//	hear.Connect(log)                 // flow.Tap: a node of the program
//	log.Connect(speak, flow.Keep(4))
//	speak.Connect(out)
//	err := flow.Run(ctx, mic)
//
// Times are the stream's own: a chunk's Start and Len count from the start of its stream, in
// samples for audio, so that they do not drift with the wall clock. Only Metronome, Pace and
// Live look at the wall clock.
package flow
