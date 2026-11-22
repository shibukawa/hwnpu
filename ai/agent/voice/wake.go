package voice

import (
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Wake finds the assistant's name at the start of a transcript: with the name コダマ,
// "コダマ、今何時？" is the name and a request, "ねえ、こだま" the name alone. A transcriber writes
// a name in several ways (コダマ, こだま, 木霊, 児玉: give them all), and puts words before it
// (ねえ, ヘイ, OK), so a transcript is matched by its letters folded (Fold: hiragana as katakana,
// without punctuation and spaces), with only words of calling (Leads) before the name:
// "今日は児玉さんと会った" does not call it.
type Wake struct {
	// Names are the ways the name may be written, folded.
	Names []string
	// Leads are the words that may come before the name, folded: one of them, or two.
	Leads []string
}

// DefaultLeads are the words said before a name to call it, in Japanese and English.
var DefaultLeads = []string{"ネエ", "ネ", "ネー", "ヘイ", "ヘー", "オーケー", "オッケー", "ok", "okay", "hey", "オイ", "アノ", "アノー", "エット", "ハイ", "ヤア", "ヤー"}

// NewWake is a Wake of the names given (as they may be written; folded here), with
// DefaultLeads.
func NewWake(names ...string) *Wake {
	w := &Wake{}
	for _, l := range DefaultLeads {
		w.Leads = append(w.Leads, Fold(l))
	}
	for _, n := range names {
		if f := Fold(n); f != "" {
			w.Names = append(w.Names, f)
		}
	}
	return w
}

// Find reports whether the transcript starts with the name, and what is said after it (the
// request; "" when the name was said alone).
func (w *Wake) Find(transcript string) (rest string, ok bool) {
	// the transcript's letters folded, each with where it ends in the transcript
	var letters []rune
	var ends []int
	for i, r := range transcript {
		f := foldRune(r)
		if f < 0 {
			continue
		}
		letters = append(letters, f)
		ends = append(ends, i+utf8.RuneLen(r))
	}
	s := string(letters)
	best := -1
	for _, n := range w.Names {
		at := strings.Index(s, n)
		if at < 0 || !w.lead(s[:at]) {
			continue
		}
		end := utf8.RuneCountInString(s[:at]) + utf8.RuneCountInString(n) // in letters
		if best < 0 || end < best {
			best = end
		}
	}
	if best < 0 {
		return "", false
	}
	rest = transcript[ends[best-1]:]
	rest = strings.TrimLeftFunc(rest, func(r rune) bool { return unicode.IsPunct(r) || unicode.IsSpace(r) || r == 'ー' })
	// "コダマさん", "コダマちゃん": a suffix of the name, not the request
	for _, suffix := range []string{"さん", "ちゃん", "くん"} {
		if r, ok := strings.CutPrefix(rest, suffix); ok {
			rest = strings.TrimLeftFunc(r, func(r rune) bool { return unicode.IsPunct(r) || unicode.IsSpace(r) })
		}
	}
	return strings.TrimSpace(rest), true
}

// lead reports whether a text is nothing, or words of calling (Leads), one or two.
func (w *Wake) lead(s string) bool {
	if s == "" {
		return true
	}
	for _, a := range w.Leads {
		if s == a {
			return true
		}
		if rest, ok := strings.CutPrefix(s, a); ok && slices.Contains(w.Leads, rest) {
			return true
		}
	}
	return false
}

// Fold is a text's letters folded: hiragana as katakana, Latin letters in lower case, without
// punctuation, symbols and spaces.
func Fold(s string) string {
	var b strings.Builder
	for _, r := range s {
		if f := foldRune(r); f >= 0 {
			b.WriteRune(f)
		}
	}
	return b.String()
}

// foldRune is a letter folded, or -1 for one that is left out.
func foldRune(r rune) rune {
	switch {
	case r >= 'ぁ' && r <= 'ゖ':
		return r + ('ァ' - 'ぁ')
	case unicode.IsPunct(r) || unicode.IsSpace(r) || unicode.IsSymbol(r):
		return -1
	}
	return unicode.ToLower(r)
}
