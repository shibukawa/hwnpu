package supertonic

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"slices"
	"strings"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

// Languages are the languages Supertonic 3 speaks, by the codes of its language tokens; "na"
// is any other.
var Languages = []string{"en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "hi", "hr", "hu",
	"id", "it", "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi", "na"}

// The text's clean-up, as the reference's UnicodeProcessor does it (supertonic 1.3.1, core.py).
var (
	emoji = regexp.MustCompile("[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff\U0001f700-\U0001f77f" +
		"\U0001f780-\U0001f7ff\U0001f800-\U0001f8ff\U0001f900-\U0001f9ff\U0001fa00-\U0001fa6f\U0001fa70-\U0001faff" +
		"☀-⛿✀-➿\U0001f1e6-\U0001f1ff]+")
	symbols = strings.NewReplacer(
		"–", "-", "‑", "-", "—", "-", "¯", " ", "_", " ",
		"“", `"`, "”", `"`, "‘", "'", "’", "'", "´", "'", "`", "'",
		"[", " ", "]", " ", "|", " ", "/", " ", "#", " ", "→", " ", "←", " ")
	special       = regexp.MustCompile(`[♥☆♡©\\]`)
	abbreviations = strings.NewReplacer("@", " at ", "e.g.,", "for example, ", "i.e.,", "that is, ")
	spacing       = strings.NewReplacer(" ,", ",", " .", ".", " !", "!", " ?", "?", " ;", ";", " :", ":", " '", "'")
	whitespace    = regexp.MustCompile(`\s+`)
	ending        = regexp.MustCompile(`[.!?;:,'"')\]}…。」』】〉》›»]$`)
)

// Preprocess is a text as the model reads it: NFKD, without emoji and decorations, symbols and
// spacing made plain, a period at the end when it has no ending punctuation, between the
// language's tokens (<ja>...</ja>).
func Preprocess(text, lang string) (string, error) {
	if !slices.Contains(Languages, lang) {
		return "", fmt.Errorf("supertonic: no language %q (%s)", lang, strings.Join(Languages, ", "))
	}
	t := norm.NFKD.String(text)
	t = emoji.ReplaceAllString(t, "")
	t = symbols.Replace(t)
	t = special.ReplaceAllString(t, "")
	t = abbreviations.Replace(t)
	t = spacing.Replace(t)
	t = dupQuotes(t)
	t = strings.TrimSpace(whitespace.ReplaceAllString(t, " "))
	if !ending.MatchString(t) {
		t += "."
	}
	return "<" + lang + ">" + t + "</" + lang + ">", nil
}

// dupQuotes leaves one of each run of the same quote (" ' `), as the reference's (["'`])\1+.
func dupQuotes(s string) string {
	var b strings.Builder
	var prev rune
	for _, r := range s {
		if r == prev && (r == '"' || r == '\'' || r == '`') {
			continue
		}
		b.WriteRune(r)
		prev = r
	}
	return b.String()
}

// Indexer maps the characters (code points below 65536) to the model's ids, -1 for those it
// does not know (unicode_indexer.json).
type Indexer []int

// LoadIndexer reads unicode_indexer.json.
func LoadIndexer(path string) (Indexer, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var ix Indexer
	if err := json.Unmarshal(raw, &ix); err != nil {
		return nil, fmt.Errorf("supertonic: %s: %w", path, err)
	}
	return ix, nil
}

// IDs are the ids of a preprocessed text, without the characters the model does not know,
// which are returned too.
func (ix Indexer) IDs(text string) (ids []int64, unknown []string) {
	ids = make([]int64, 0, utf8.RuneCountInString(text))
	for _, r := range text {
		id := -1
		if int(r) < len(ix) {
			id = ix[r]
		}
		if id < 0 {
			unknown = append(unknown, string(r))
			continue
		}
		ids = append(ids, int64(id))
	}
	return ids, unknown
}
