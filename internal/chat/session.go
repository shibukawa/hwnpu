package chat

import (
	"github.com/shibukawa/hwnpu/internal/llm/decode"
)

// session is the sequence in a model's cache: decode.Session for the models converted for Core
// ML (and Intel), ironSession for the ones IRON compiled for AMD's NPU.
type session interface {
	Q() int     // the most tokens one call reads
	Qs() []int  // the tokens per call of each program, the one that generates first
	Cache() int // the positions the cache holds
	Pos() int   // the tokens in the cache
	Reset()     // empty the sequence
	Mark() *decode.Mark
	// Save copies the whole cache to come back to with Rewind, or returns nil when the session
	// cannot (then the engine keeps no other conversation aside).
	Save() *decode.Mark
	Rewind(m *decode.Mark)
	Resume(rows []byte, m *decode.Mark) error
	Step(emb [][]float32) ([]float32, error)
	Speculate(emb [][]float32, accept func(row func(i int) []float32) int) ([]float32, int, error)
	Migrate(p decode.Program) error
	Add(p decode.Program) error
}

// tokenSession is a session that reads token ids itself (the model embeds them on the device).
type tokenSession interface {
	StepTokens(toks []int32) ([]float32, error)
	// SpeculateTokens is Speculate on token ids.
	SpeculateTokens(toks []int32, accept func(row func(i int) []float32) int) ([]float32, int, error)
}

// rewindPointer is a session that goes back inside its sequence without a mark taken there:
// RewindPoint is the mark of the furthest position at or before n it can go back to (nil for
// none, as the start needs no mark). Stateless reports whether that is n itself (the state is
// the caches' rows by position alone), so that marks are never needed.
type rewindPointer interface {
	RewindPoint(n int) *decode.Mark
	Stateless() bool
}

var _ session = (*decode.Session)(nil)
