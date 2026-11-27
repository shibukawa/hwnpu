// Package soundwire is what the sound models' engine (ai/sound/engine, internal/music,
// internal/notes, internal/pitch, internal/sfx) and their client (ai/sound) share without the
// engine: the models' rates and defaults, and the reading and writing of MIDI and WAV files
// (docs/design/hwnpud.md, "Clients without the engine").
//
// The constants are the engine packages' (ai/sound/engine's tests check that they agree), and
// ReadMIDI, WriteMIDI and DecodeWAV are copies of internal/notes's and internal/llm/lfm3audio's,
// whose code the definitions hash (model-recipes.yaml): those are to point here when their code
// changes anyway.
package soundwire

// The models' constants.
const (
	// PitchRate and PitchHop are SwiftF0's (internal/pitch): 16 kHz mono in, a frame every 256
	// samples.
	PitchRate = 16000
	PitchHop  = 256
	// NotesRate is Basic Pitch's (internal/notes).
	NotesRate = 22050
	// EffectRate, EffectMaxSeconds and EffectPromptTokens are MOSS-SoundEffect's (internal/sfx).
	EffectRate         = 48000
	EffectMaxSeconds   = 30
	EffectPromptTokens = 256
	// MusicRate, MusicFrameRate and StyleCodes are Magenta RealTime's (internal/music).
	MusicRate      = 48000
	MusicFrameRate = 25
	StyleCodes     = 12
)

// Basic Pitch's defaults of making notes (internal/notes.DefaultOptions).
const (
	NotesOnset     = 0.5
	NotesFrame     = 0.3
	NotesMinNoteMS = 127.7
)

// Magenta RealTime's guidance scales (internal/music.DefaultGuidance).
const (
	GuidanceStyle = 3
	GuidanceNotes = 5
	GuidanceDrums = 1
)
