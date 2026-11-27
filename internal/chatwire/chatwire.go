// Package chatwire is what the chat engine (internal/chat) and its clients (ai/text) share
// without the engine: the errors a model's call returns and the small rules of its answers
// (docs/design/hwnpud.md, "Clients without the engine").
package chatwire

import (
	"errors"
	"math"
	"unicode"
)

var (
	// ErrContext is the error of a prompt that does not fit in the context.
	ErrContext = errors.New("chat: the prompt does not fit in the context")
	// ErrImage is the error of an image or a clip the model cannot read.
	ErrImage = errors.New("chat: media")
	// ErrTooLong is returned for a text over MaxTokens when it is not to be truncated.
	ErrTooLong = errors.New("chat: the text is longer than the embedding model reads")
	// ErrEmpty is returned for a text with no tokens of its own.
	ErrEmpty = errors.New("chat: empty text")
)

// Cosine is the cosine similarity of two embeddings.
func Cosine(a, b []float32) float64 {
	var ab, aa, bb float64
	for i := range a {
		ab += float64(a[i]) * float64(b[i])
		aa += float64(a[i]) * float64(a[i])
		bb += float64(b[i]) * float64(b[i])
	}
	if aa == 0 || bb == 0 {
		return 0
	}
	return ab / math.Sqrt(aa*bb)
}

// AutoTarget is the language a text is translated into when not told: English for a text with
// hiragana, katakana or kanji, Japanese for any other.
func AutoTarget(text string) string {
	for _, r := range text {
		if unicode.In(r, unicode.Hiragana, unicode.Katakana, unicode.Han) {
			return "en"
		}
	}
	return "ja"
}
