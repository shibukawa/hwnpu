package voice

import "strings"

// Splitter cuts the text of an answer, as it comes, into the pieces to speak: sentences, and the
// first one sooner, at a comma once it is First letters long, so that the first sound comes
// before the model has written the whole sentence and before a long one has been synthesized
// whole; a later piece is cut at a comma past Long letters. When speech is synthesized more
// slowly than it plays, a short first piece is what makes the answer start soon.
type Splitter struct {
	// First and Long are the letters after which a piece ends at a comma: the first piece, and
	// any other (defaults 8 and 40).
	First, Long int
	// Out gets each piece.
	Out func(piece string)

	buf  strings.Builder
	n    int // the letters in buf
	more bool
}

// Add takes the next text of the answer.
func (s *Splitter) Add(text string) {
	first, long := s.First, s.Long
	if first <= 0 {
		first = 8
	}
	if long <= 0 {
		long = 40
	}
	for _, r := range text {
		s.buf.WriteRune(r)
		s.n++
		switch {
		case strings.ContainsRune("。！？!?\n", r):
			s.Flush()
		case strings.ContainsRune("、，,", r) && (!s.more && s.n >= first || s.n >= long):
			s.Flush()
		}
	}
}

// Flush passes on what is held: the end of the answer.
func (s *Splitter) Flush() {
	t := strings.TrimSpace(s.buf.String())
	s.buf.Reset()
	s.n = 0
	if t == "" || strings.Trim(t, "。、！？!?,. ") == "" {
		return
	}
	s.more = true
	s.Out(t)
}
