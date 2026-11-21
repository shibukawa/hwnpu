package flow

import (
	"strings"
	"unicode/utf8"
)

// Text is a chunk of text: a transcript, a translation, a sentence to speak.
type Text struct {
	Text string
	// Lang is the language's code ("ja", "en"), "" when not known.
	Lang string
}

// sentenceEnds are the characters that end a sentence: Japanese and Chinese ones end it where
// they are, Western ones before a space or the end of the text.
const (
	sentenceEndsCJK  = "。！？．!?"
	sentenceEndsWest = ".!?"
)

// SplitSentences cuts a text into its sentences, each with what follows it up to the next one
// (the spaces after a period stay with the sentence before).
func SplitSentences(text string) []string {
	var out []string
	start := 0
	for i := 0; i < len(text); {
		r, size := utf8.DecodeRuneInString(text[i:])
		i += size
		end := false
		switch {
		case r < utf8.RuneSelf && strings.ContainsRune(sentenceEndsWest, r):
			end = i == len(text) || text[i] == ' ' || text[i] == '\n' || text[i] == '"' || text[i] == ')'
		case strings.ContainsRune(sentenceEndsCJK, r) || r == '\n':
			end = true
		}
		if !end {
			continue
		}
		// closing quotes and brackets, and the spaces after them, stay with the sentence
		for i < len(text) {
			r, size := utf8.DecodeRuneInString(text[i:])
			if !strings.ContainsRune("」』）)\"' \n", r) {
				break
			}
			i += size
		}
		if s := text[start:i]; strings.TrimSpace(s) != "" {
			out = append(out, s)
		}
		start = i
	}
	if s := text[start:]; strings.TrimSpace(s) != "" {
		out = append(out, s)
	}
	return out
}

// Sentences cuts each final chunk of a stream of text into its sentences, each a final chunk of
// its own, so that a reader of sentences (a speaker) starts on the first one sooner. The chunks
// before a final one, provisional, are left out. The sentences share their chunk's times.
func Sentences(s Stream[Text]) Stream[Text] {
	return func(yield func(Chunk[Text], error) bool) {
		for c, err := range s {
			if err != nil {
				yield(c, err)
				return
			}
			if !c.Final {
				continue
			}
			for _, t := range SplitSentences(c.Data.Text) {
				out := c
				out.Data.Text = strings.TrimSpace(t)
				if !yield(out, nil) {
					return
				}
			}
		}
	}
}
