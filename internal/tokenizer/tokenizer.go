// Package tokenizer turns text into token ids and back.
//
// Load reads a Hugging Face tokenizer.json with the byte-level BPE of GPT-2 and its descendants
// (LFM, Llama 3, Qwen): added tokens are matched in the raw text first, the rest is split by the
// pre-tokenizer pattern, each piece is mapped byte by byte to printable characters and merged by
// rank. Other models (WordPiece, Unigram, SentencePiece byte fallback) and other pre-tokenizer
// patterns are rejected by Load. LoadSentencePiece reads a SentencePiece BPE tokenizer.model
// instead (PLaMo).
package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Tokenizer is a loaded byte-level BPE tokenizer.
type Tokenizer struct {
	vocab        map[string]int32
	tokens       []string // by id; added tokens hold their content
	special      []bool   // by id
	ranks        map[[2]string]int
	ignoreMerges bool
	added        []string // contents, longest first
	addedID      map[string]int32
	control      []string // contents of the added tokens marked special, in the file's order
	cache        map[string][]int32
}

// The pre-tokenizer patterns Load accepts. They are applied by split, not by a regular expression:
// Go's regexp has no lookahead for \s+(?!\S).
var patterns = map[string]bool{
	// GPT-4 style, as LFM2/LFM3 and many others ship it.
	`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`: true,
}

type file struct {
	AddedTokens []struct {
		ID      int32  `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
	} `json:"added_tokens"`
	Normalizer   json.RawMessage `json:"normalizer"`
	PreTokenizer struct {
		Type          string `json:"type"`
		Pretokenizers []struct {
			Type    string `json:"type"`
			Pattern struct {
				Regex string `json:"Regex"`
			} `json:"pattern"`
			Behavior  string `json:"behavior"`
			Invert    bool   `json:"invert"`
			AddPrefix bool   `json:"add_prefix_space"`
			UseRegex  bool   `json:"use_regex"`
		} `json:"pretokenizers"`
	} `json:"pre_tokenizer"`
	Model struct {
		Type         string            `json:"type"`
		Vocab        map[string]int32  `json:"vocab"`
		Merges       []json.RawMessage `json:"merges"`
		IgnoreMerges bool              `json:"ignore_merges"`
		ByteFallback bool              `json:"byte_fallback"`
	} `json:"model"`
}

// Load reads a tokenizer.json.
func Load(path string) (*Tokenizer, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	t, err := parse(raw)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: %s: %w", path, err)
	}
	return t, nil
}

func parse(raw []byte) (*Tokenizer, error) {
	var f file
	if err := json.Unmarshal(raw, &f); err != nil {
		return nil, err
	}
	if f.Model.Type != "BPE" || f.Model.ByteFallback {
		return nil, fmt.Errorf("model %s (byte fallback %v) is not supported; only byte-level BPE", f.Model.Type, f.Model.ByteFallback)
	}
	if n := strings.TrimSpace(string(f.Normalizer)); n != "" && n != "null" {
		return nil, fmt.Errorf("normalizer %s is not supported", n)
	}
	pt := f.PreTokenizer
	if pt.Type != "Sequence" || len(pt.Pretokenizers) != 2 ||
		pt.Pretokenizers[0].Type != "Split" || pt.Pretokenizers[0].Behavior != "Isolated" || pt.Pretokenizers[0].Invert || !patterns[pt.Pretokenizers[0].Pattern.Regex] ||
		pt.Pretokenizers[1].Type != "ByteLevel" || pt.Pretokenizers[1].AddPrefix || pt.Pretokenizers[1].UseRegex {
		return nil, fmt.Errorf("pre-tokenizer is not supported; only Split (the GPT-4 pattern, isolated) then ByteLevel without prefix space")
	}
	t := &Tokenizer{vocab: f.Model.Vocab, ranks: map[[2]string]int{}, ignoreMerges: f.Model.IgnoreMerges, addedID: map[string]int32{}, cache: map[string][]int32{}}
	n := int32(0)
	for _, id := range f.Model.Vocab {
		n = max(n, id+1)
	}
	for _, a := range f.AddedTokens {
		n = max(n, a.ID+1)
	}
	t.tokens = make([]string, n)
	t.special = make([]bool, n)
	for s, id := range f.Model.Vocab {
		t.tokens[id] = s
	}
	for _, a := range f.AddedTokens {
		t.tokens[a.ID] = a.Content
		t.special[a.ID] = true
		t.addedID[a.Content] = a.ID
		t.added = append(t.added, a.Content)
		if a.Special {
			t.control = append(t.control, a.Content)
		}
	}
	sort.Slice(t.added, func(i, j int) bool { return len(t.added[i]) > len(t.added[j]) })
	for i, m := range f.Model.Merges {
		var pair [2]string
		var s string
		if json.Unmarshal(m, &s) == nil {
			a, b, ok := strings.Cut(s, " ")
			if !ok {
				return nil, fmt.Errorf("merge %d %q has no space", i, s)
			}
			pair = [2]string{a, b}
		} else if err := json.Unmarshal(m, &pair); err != nil {
			return nil, fmt.Errorf("merge %d: %w", i, err)
		}
		if _, dup := t.ranks[pair]; !dup {
			t.ranks[pair] = i
		}
	}
	return t, nil
}

// VocabSize is one more than the largest id.
func (t *Tokenizer) VocabSize() int { return len(t.tokens) }

// Control returns the contents of the added tokens marked special (llama.cpp's control tokens,
// such as "<|im_start|>"), in the order of tokenizer.json.
func (t *Tokenizer) Control() []string { return t.control }

// ID returns the id of a token by its content (an added token such as "<|im_end|>", or a
// vocabulary entry in its byte-level form).
func (t *Tokenizer) ID(token string) (int32, bool) {
	if id, ok := t.addedID[token]; ok {
		return id, true
	}
	id, ok := t.vocab[token]
	return id, ok
}

// Encode turns text into token ids. Added tokens written in the text (such as "<|im_start|>")
// become their ids.
func (t *Tokenizer) Encode(text string) []int32 {
	var ids []int32
	start := 0
	for i := 0; i < len(text); {
		if a := t.addedAt(text[i:]); a != "" {
			ids = t.encodeOrdinary(ids, text[start:i])
			ids = append(ids, t.addedID[a])
			i += len(a)
			start = i
			continue
		}
		_, n := utf8.DecodeRuneInString(text[i:])
		i += n
	}
	return t.encodeOrdinary(ids, text[start:])
}

func (t *Tokenizer) addedAt(s string) string {
	for _, a := range t.added {
		if strings.HasPrefix(s, a) {
			return a
		}
	}
	return ""
}

func (t *Tokenizer) encodeOrdinary(ids []int32, text string) []int32 {
	for _, piece := range split(text) {
		ids = append(ids, t.bpe(piece)...)
	}
	return ids
}

// bpe merges the byte-level form of one piece of text.
func (t *Tokenizer) bpe(piece string) []int32 {
	if ids, ok := t.cache[piece]; ok {
		return ids
	}
	var word strings.Builder
	var syms []string
	for _, b := range []byte(piece) {
		c := string(byteToRune[b])
		word.WriteString(c)
		syms = append(syms, c)
	}
	var ids []int32
	if id, ok := t.vocab[word.String()]; ok && t.ignoreMerges {
		ids = []int32{id}
	} else {
		for len(syms) > 1 {
			best, at := -1, -1
			for i := 0; i+1 < len(syms); i++ {
				if r, ok := t.ranks[[2]string{syms[i], syms[i+1]}]; ok && (best < 0 || r < best) {
					best, at = r, i
				}
			}
			if at < 0 {
				break
			}
			// Merge every occurrence of the best pair, left to right.
			first, second := syms[at], syms[at+1]
			out := syms[:0:0]
			for i := 0; i < len(syms); {
				if i+1 < len(syms) && syms[i] == first && syms[i+1] == second {
					out = append(out, first+second)
					i += 2
				} else {
					out = append(out, syms[i])
					i++
				}
			}
			syms = out
		}
		for _, s := range syms {
			if id, ok := t.vocab[s]; ok {
				ids = append(ids, id)
			}
			// A symbol outside the vocabulary cannot happen with a complete byte-level vocabulary.
		}
	}
	t.cache[piece] = ids
	return ids
}

// Decode turns token ids back into text. Added tokens are written as their content unless
// skipSpecial is set.
func (t *Tokenizer) Decode(ids []int32, skipSpecial bool) string {
	var out []byte
	for _, id := range ids {
		if id < 0 || int(id) >= len(t.tokens) {
			continue
		}
		if t.special[id] {
			if !skipSpecial {
				out = append(out, t.tokens[id]...)
			}
			continue
		}
		for _, r := range t.tokens[id] {
			if b, ok := runeToByte[r]; ok {
				out = append(out, b)
			}
		}
	}
	return string(out)
}

// byteToRune is GPT-2's map from bytes to printable characters (bytes_to_unicode).
var byteToRune, runeToByte = func() ([256]rune, map[rune]byte) {
	var m [256]rune
	inv := map[rune]byte{}
	n := 0
	for b := 0; b < 256; b++ {
		if b >= '!' && b <= '~' || b >= 0xA1 && b <= 0xAC || b >= 0xAE && b <= 0xFF {
			m[b] = rune(b)
		} else {
			m[b] = rune(256 + n)
			n++
		}
		inv[m[b]] = byte(b)
	}
	return m, inv
}()

// split applies the GPT-4 pre-tokenizer pattern
//
//	(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//
// trying the alternatives in order at each position, as a backtracking regular expression does.
func split(text string) []string {
	rs := []rune(text)
	var out []string
	for i := 0; i < len(rs); {
		n := match(rs, i)
		out = append(out, string(rs[i:i+n]))
		i += n
	}
	return out
}

func isL(r rune) bool  { return unicode.IsLetter(r) }
func isN(r rune) bool  { return unicode.IsNumber(r) }
func isS(r rune) bool  { return unicode.IsSpace(r) }
func isNL(r rune) bool { return r == '\r' || r == '\n' }

// match returns the length of the match at i (at least 1).
func match(rs []rune, i int) int {
	rest := len(rs) - i
	// (?i:'s|'t|'re|'ve|'m|'ll|'d)
	if rs[i] == '\'' && rest >= 2 {
		for _, c := range []string{"s", "t", "re", "ve", "m", "ll", "d"} {
			if rest > len(c) && strings.EqualFold(string(rs[i+1:i+1+len(c)]), c) {
				return 1 + len(c)
			}
		}
	}
	// [^\r\n\p{L}\p{N}]?\p{L}+
	letters := func(j int) int {
		k := j
		for k < len(rs) && isL(rs[k]) {
			k++
		}
		return k - j
	}
	if !isNL(rs[i]) && !isL(rs[i]) && !isN(rs[i]) && i+1 < len(rs) && isL(rs[i+1]) {
		return 1 + letters(i+1)
	}
	if isL(rs[i]) {
		return letters(i)
	}
	// \p{N}{1,3}
	if isN(rs[i]) {
		k := i
		for k < len(rs) && k-i < 3 && isN(rs[k]) {
			k++
		}
		return k - i
	}
	// ' ?[^\s\p{L}\p{N}]+[\r\n]*'
	other := func(r rune) bool { return !isS(r) && !isL(r) && !isN(r) }
	j := i
	if rs[j] == ' ' && j+1 < len(rs) && other(rs[j+1]) {
		j++
	}
	if other(rs[j]) {
		for j < len(rs) && other(rs[j]) {
			j++
		}
		for j < len(rs) && isNL(rs[j]) {
			j++
		}
		return j - i
	}
	// The remaining alternatives start with whitespace.
	end := i
	for end < len(rs) && isS(rs[end]) {
		end++
	}
	if end == i {
		return 1 // not reached: every character is a letter, a number, whitespace or other
	}
	// \s*[\r\n]+: up to just after the last line break of the run.
	for k := end - 1; k >= i; k-- {
		if isNL(rs[k]) {
			return k + 1 - i
		}
	}
	// \s+(?!\S): the whole run at the end of the text, otherwise all but its last character.
	if end == len(rs) {
		return end - i
	}
	if end-1 > i {
		return end - 1 - i
	}
	// \s+
	return end - i
}
