package chat

import (
	"hash/maphash"
	"strings"
	"unicode"

	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

// repeats tells a generation that goes round in circles with a period too long for looping, or
// with a little changed each time round: a small model asked what it does not know may list the
// same items again and again, "- Gifu? Already." "- Mie? Already." ... until the context is
// full. A loop comes back to what it wrote twice before, where an answer that copies a draft
// (or its thinking, the tests of a class with the same lines) comes back to it once: what
// counts is how much of the latest text is the third time or more, in lines and in runs of
// tokens. The thinking and the answer are counted apart (reset), so that an answer may write out
// what was drafted twice in the thinking. Numbers are left as they are: rows that differ in
// their numbers alone are a table as often as a loop.
type repeats struct {
	kinds []tokenKind // by token; nil: no line to tell (no tokenizer)
	seed  maphash.Seed

	// lines: those with at least lineWords words, as their tokens' hash
	line         maphash.Hash
	words        int
	lineSeen     map[uint64]int
	lineRecent   [lineWindow]bool // the third time or more, for the latest lines
	lineN, lineH int              // how many lines in lineRecent, and of them the third time or more

	// runs of gramN tokens
	toks         [gramN]int32 // the latest tokens, toks[n%gramN] the oldest when n >= gramN
	n            int
	gramSeen     map[uint64]int
	gramRecent   [gramWindow]bool
	gramN, gramH int
}

type tokenKind uint8

const (
	kindWord    tokenKind = 1 << iota // a token with a letter in it
	kindLineEnd                       // a token with a line break in it
)

const (
	lineWords  = 2  // a line with fewer words ("}", "---", "| 1 | 2 |") does not count
	lineWindow = 16 // latest lines looked at
	lineLoop   = 12 // of them, the third time or more for a loop
	gramN      = 8
	gramWindow = 256
	gramLoop   = 230
)

// tokenKinds classifies a tokenizer's tokens for repeats.
func tokenKinds(tok *tokenizer.Tokenizer) []tokenKind {
	kinds := make([]tokenKind, tok.VocabSize())
	for id := range kinds {
		s := tok.Decode([]int32{int32(id)}, true)
		if strings.ContainsAny(s, "\r\n") {
			kinds[id] |= kindLineEnd
		}
		if strings.IndexFunc(s, unicode.IsLetter) >= 0 {
			kinds[id] |= kindWord
		}
	}
	return kinds
}

func newRepeats(kinds []tokenKind) *repeats {
	r := &repeats{kinds: kinds, seed: maphash.MakeSeed()}
	r.reset()
	return r
}

// reset starts counting again (the thinking or the answer begins).
func (r *repeats) reset() {
	r.line.SetSeed(r.seed)
	r.words = 0
	r.lineSeen = map[uint64]int{}
	r.lineN, r.lineH = 0, 0
	r.n = 0
	r.gramSeen = map[uint64]int{}
	r.gramN, r.gramH = 0, 0
}

// add takes the next token and says whether the generation has fallen into a loop.
func (r *repeats) add(tok int32) bool {
	var b [4]byte
	put := func(h *maphash.Hash, t int32) {
		b[0], b[1], b[2], b[3] = byte(t), byte(t>>8), byte(t>>16), byte(t>>24)
		h.Write(b[:])
	}
	looped := false
	r.toks[r.n%gramN] = tok
	r.n++
	if r.n >= gramN {
		var h maphash.Hash
		h.SetSeed(r.seed)
		for i := range gramN {
			put(&h, r.toks[(r.n+i)%gramN])
		}
		k := h.Sum64()
		looped = slide(r.gramRecent[:], &r.gramN, &r.gramH, r.gramSeen[k] >= 2) && r.gramH >= gramLoop
		r.gramSeen[k]++
	}
	if int(tok) < 0 || int(tok) >= len(r.kinds) {
		return looped
	}
	kind := r.kinds[tok]
	if kind&kindLineEnd == 0 {
		put(&r.line, tok)
		if kind&kindWord != 0 {
			r.words++
		}
		return looped
	}
	if r.words >= lineWords {
		k := r.line.Sum64()
		if slide(r.lineRecent[:], &r.lineN, &r.lineH, r.lineSeen[k] >= 2) && r.lineH >= lineLoop {
			looped = true
		}
		r.lineSeen[k]++
	}
	r.line.Reset()
	r.words = 0
	return looped
}

// slide puts hit into a window of the latest ones, with n of them in it and h of those hits; it
// says whether the window is full.
func slide(s []bool, n, h *int, hit bool) bool {
	i := *n % len(s)
	if *n >= len(s) && s[i] {
		*h--
	}
	s[i] = hit
	if hit {
		*h++
	}
	*n++
	return *n >= len(s)
}
