// Package tokenizer turns text into token ids and back.
//
// Load reads a Hugging Face tokenizer.json with the byte-level BPE of GPT-2 and its descendants
// (LFM, Llama 3, Qwen): added tokens are matched in the raw text first, the rest is put in
// Unicode normalization form C when the tokenizer says so (Qwen3.5) and split by the
// pre-tokenizer pattern (GPT-4's, or Qwen3.5's), each piece is mapped byte by byte to printable characters and merged by
// rank. It also reads the BPE with byte fallback that SentencePiece models ship as tokenizer.json
// (Gemma): spaces become ▁, the text between added tokens is merged by rank as one piece, and a
// character outside the vocabulary becomes its UTF-8 bytes (<0xE3>). Other models (WordPiece,
// Unigram) and other normalizers and pre-tokenizer patterns are rejected by Load.
// LoadSentencePiece reads a SentencePiece BPE tokenizer.model instead (PLaMo's embedding model),
// and LoadPlamo PLaMo 2/3's tokenizer.jsonl, which Load also reads when given its path.
package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"

	"github.com/shibukawa/hwnpu/internal/mapped"
)

// Tokenizer is a loaded BPE tokenizer, byte-level or with byte fallback.
type Tokenizer struct {
	vocab        *strTable  // the model's vocabulary: token → id
	tokens       *strTable  // by id; added tokens hold their content
	special      []byte     // by id: 1 for an added token (PLaMo: a control or unknown piece)
	merges       *pairTable // byte-level: the merges of pairs of symbols, by their ids (mergeSymbols)
	byteSym      [256]int32 // byte-level: each byte's symbol (byteToRune's character)
	ignoreMerges bool
	added        []string // contents, longest first
	addedID      map[string]int32
	control      []string // contents of the added tokens marked special, in the file's order
	cache        map[string][]int32
	style        style // the pre-tokenizer pattern
	nfc          bool  // the text is put in Unicode normalization form C first

	fallback *fallback    // the BPE with byte fallback, or nil for byte-level
	plamo    *plamo       // PLaMo's scored pieces (LoadPlamo), or nil
	file     *mapped.File // the binary file the tables are in (nil: built from the text)
}

// style is a pre-tokenizer pattern that split implements.
type style int

const (
	// gpt4 is the GPT-4 pattern, as LFM2/LFM3 and many others ship it.
	gpt4 style = iota
	// qwen35 is Qwen3.5's: combining marks (\p{M}) count as letters, and numbers are split into
	// single digits.
	qwen35
	// qwen2 is Qwen2's and Qwen3's: GPT-4's with numbers split into single digits.
	qwen2
	// gpt2 is GPT-2's, of a ByteLevel pre-tokenizer with use_regex (Whisper's):
	//
	//	's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
	gpt2
	// gpt2Digits is a Digits pre-tokenizer with individual_digits, then GPT-2's pattern on each
	// piece (SmolLM's and Ouro's): every number character a piece of its own, the pattern applied
	// to the runs between them.
	gpt2Digits
	// o200k is o200k_harmony's (gpt-oss): letter runs split at case, a contraction joined to the
	// run, numbers in threes, and '/' after a run of other characters:
	//
	//	[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|
	//	[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|
	//	\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
	o200k
)

// The pre-tokenizer patterns Load accepts. They are applied by split, not by a regular expression:
// Go's regexp has no lookahead for \s+(?!\S).
var patterns = map[string]style{
	`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`:                                                                                                                                                                gpt4,
	`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`:                                                                                                                                                         qwen35,
	`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`:                                                                                                                                                                     qwen2,
	`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`: o200k,
}

type file struct {
	AddedTokens []struct {
		ID      int32  `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
	} `json:"added_tokens"`
	Normalizer   json.RawMessage `json:"normalizer"`
	PreTokenizer struct {
		Type string `json:"type"`
		// of a ByteLevel pre-tokenizer alone (GPT-2's, Whisper's)
		AddPrefix     bool `json:"add_prefix_space"`
		UseRegex      bool `json:"use_regex"`
		Pretokenizers []struct {
			Type    string `json:"type"`
			Pattern struct {
				Regex string `json:"Regex"`
			} `json:"pattern"`
			Behavior  string `json:"behavior"`
			Invert    bool   `json:"invert"`
			AddPrefix bool   `json:"add_prefix_space"`
			UseRegex  bool   `json:"use_regex"`
			// of a Digits pre-tokenizer
			IndividualDigits bool `json:"individual_digits"`
		} `json:"pretokenizers"`
	} `json:"pre_tokenizer"`
	Model struct {
		Type         string            `json:"type"`
		Vocab        map[string]int32  `json:"vocab"`
		Merges       []json.RawMessage `json:"merges"`
		IgnoreMerges bool              `json:"ignore_merges"`
		ByteFallback bool              `json:"byte_fallback"`
	} `json:"model"`
	// pairs are the merges as splitModel read them (Model.Merges is then empty); nil: not read
	pairs [][2]string
}

// Load reads a tokenizer.json, or PLaMo's tokenizer.jsonl (LoadPlamo): the binary file that
// stands for it (BinaryPath) when there is one, which needs neither of them; else the file,
// kept as a binary file in the cache for the next time (cachedLoad).
func Load(path string) (*Tokenizer, error) {
	if t, err, ok := openBeside(path, loadBinary); ok {
		return t, err
	}
	return cachedLoad(path, loadText, (*Tokenizer).WriteBinary, loadBinary)
}

// loadText reads a tokenizer.json, or PLaMo's tokenizer.jsonl.
func loadText(path string) (*Tokenizer, error) {
	if strings.HasSuffix(path, ".jsonl") {
		return loadPlamoText(path)
	}
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
	// the vocabulary and the merges, nearly all of the file, read by themselves (splitModel)
	small, vocab, pairs, fast := splitModel(raw)
	if fast {
		raw = small
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		return nil, err
	}
	if fast {
		f.Model.Vocab, f.pairs = vocab, pairs
	}
	if f.Model.Type == "BPE" && f.Model.ByteFallback {
		return parseFallback(raw, &f)
	}
	if f.Model.Type != "BPE" {
		return nil, fmt.Errorf("model %s is not supported; only BPE", f.Model.Type)
	}
	nfc := false
	if n := strings.TrimSpace(string(f.Normalizer)); n != "" && n != "null" {
		var norm struct {
			Type string `json:"type"`
		}
		if json.Unmarshal(f.Normalizer, &norm) != nil || norm.Type != "NFC" {
			return nil, fmt.Errorf("normalizer %s is not supported; only NFC", n)
		}
		nfc = true
	}
	pt := f.PreTokenizer
	if pt.Type == "ByteLevel" && pt.UseRegex && !pt.AddPrefix {
		t := newTokenizer(&f)
		t.style, t.nfc = gpt2, nfc
		if err := t.readMerges(&f); err != nil {
			return nil, err
		}
		return t, nil
	}
	if pt.Type == "Sequence" && len(pt.Pretokenizers) == 2 &&
		pt.Pretokenizers[0].Type == "Digits" && pt.Pretokenizers[0].IndividualDigits &&
		pt.Pretokenizers[1].Type == "ByteLevel" && pt.Pretokenizers[1].UseRegex && !pt.Pretokenizers[1].AddPrefix {
		t := newTokenizer(&f)
		t.style, t.nfc = gpt2Digits, nfc
		if err := t.readMerges(&f); err != nil {
			return nil, err
		}
		return t, nil
	}
	if pt.Type != "Sequence" || len(pt.Pretokenizers) != 2 ||
		pt.Pretokenizers[0].Type != "Split" || pt.Pretokenizers[0].Behavior != "Isolated" || pt.Pretokenizers[0].Invert || !knownPattern(pt.Pretokenizers[0].Pattern.Regex) ||
		pt.Pretokenizers[1].Type != "ByteLevel" || pt.Pretokenizers[1].AddPrefix || pt.Pretokenizers[1].UseRegex {
		return nil, fmt.Errorf("pre-tokenizer is not supported; only Split (the GPT-4, Qwen3, Qwen3.5 or o200k pattern, isolated) or individual Digits, then ByteLevel without prefix space")
	}
	t := newTokenizer(&f)
	t.style, t.nfc = patterns[pt.Pretokenizers[0].Pattern.Regex], nfc
	if err := t.readMerges(&f); err != nil {
		return nil, err
	}
	return t, nil
}

// newTokenizer holds the vocabulary and the added tokens of f.
func newTokenizer(f *file) *Tokenizer {
	t := &Tokenizer{vocab: vocabTable(f.Model.Vocab), ignoreMerges: f.Model.IgnoreMerges, addedID: map[string]int32{}, cache: map[string][]int32{}}
	n := int32(0)
	for _, id := range f.Model.Vocab {
		n = max(n, id+1)
	}
	for _, a := range f.AddedTokens {
		n = max(n, a.ID+1)
	}
	tokens := make([]string, n)
	t.special = make([]byte, n)
	for s, id := range f.Model.Vocab {
		tokens[id] = s
	}
	for _, a := range f.AddedTokens {
		tokens[a.ID] = a.Content
		t.special[a.ID] = 1
		t.addedID[a.Content] = a.ID
		t.added = append(t.added, a.Content)
		if a.Special {
			t.control = append(t.control, a.Content)
		}
	}
	t.tokens = newStrTable(tokens, nil, false)
	t.vocab = vocabOver(t.tokens, f.Model.Vocab, t.vocab)
	sortAdded(t.added)
	return t
}

// vocabOver is the vocabulary as an index of the tokens (indexOf) when each of its tokens is the
// token of its id, one token an id (no added token in the place of one of them): the strings once
// in a binary file rather than twice. Else it is table, the vocabulary's own.
func vocabOver(tokens *strTable, vocab map[string]int32, table *strTable) *strTable {
	ids := make([]int32, 0, len(vocab))
	seen := make([]bool, tokens.n())
	for s, id := range vocab {
		if id < 0 || int(id) >= len(seen) || seen[id] || tokens.str(int(id)) != s {
			return table
		}
		seen[id] = true
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	return indexOf(tokens, ids)
}

// sortAdded orders added tokens longest first (the first that matches in a text is taken).
func sortAdded(added []string) {
	sort.SliceStable(added, func(i, j int) bool { return len(added[i]) > len(added[j]) })
}

// vocabTable is the table of a vocabulary, its tokens in the order of their ids (and of the
// tokens, for ids given twice), so that the same vocabulary makes the same bytes.
func vocabTable(vocab map[string]int32) *strTable {
	strs := make([]string, 0, len(vocab))
	for s := range vocab {
		strs = append(strs, s)
	}
	sort.Slice(strs, func(i, j int) bool {
		if a, b := vocab[strs[i]], vocab[strs[j]]; a != b {
			return a < b
		}
		return strs[i] < strs[j]
	})
	val := make([]int32, len(strs))
	for i, s := range strs {
		val[i] = vocab[s]
	}
	return newStrTable(strs, val, true)
}

// pairsOf are the merges of f by rank, as "a b" or ["a", "b"] in the file.
func pairsOf(f *file) ([][2]string, error) {
	if f.pairs != nil {
		return f.pairs, nil
	}
	pairs := make([][2]string, 0, len(f.Model.Merges))
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
		pairs = append(pairs, pair)
	}
	return pairs, nil
}

// readMerges reads the merge ranks of the byte-level BPE (mergeSymbols).
func (t *Tokenizer) readMerges(f *file) error {
	pairs, err := pairsOf(f)
	if err != nil {
		return err
	}
	t.mergeSymbols(f.Model.Vocab, pairs)
	return nil
}

// mergeSymbols makes the byte-level BPE's tables: each byte's symbol and the merges by the ids of
// their symbols, a pair written twice at its first rank. A symbol that is no token (a merge's
// part or what it makes, which the vocabulary need not hold) takes an id past the tokens': it
// merges as the string did and is dropped from what Encode gives, as the string was.
func (t *Tokenizer) mergeSymbols(vocab map[string]int32, pairs [][2]string) {
	next := int32(t.tokens.n())
	extra := map[string]int32{}
	idOf := func(s string) int32 {
		if id, ok := vocab[s]; ok {
			return id
		}
		if id, ok := extra[s]; ok {
			return id
		}
		extra[s] = next
		next++
		return next - 1
	}
	for b := range t.byteSym {
		t.byteSym[b] = idOf(string(byteToRune[b]))
	}
	seen := make(map[[2]int32]bool, len(pairs))
	entries := make([]pairEntry, 0, len(pairs))
	for rank, pair := range pairs {
		a, b := idOf(pair[0]), idOf(pair[1])
		if seen[[2]int32{a, b}] {
			continue
		}
		seen[[2]int32{a, b}] = true
		entries = append(entries, pairEntry{a, b, int32(rank), idOf(pair[0] + pair[1])})
	}
	t.merges = newPairTable(entries)
}

// VocabSize is one more than the largest id.
func (t *Tokenizer) VocabSize() int { return t.tokens.n() }

// Control returns the contents of the added tokens marked special (llama.cpp's control tokens,
// such as "<|im_start|>"), in the order of tokenizer.json.
func (t *Tokenizer) Control() []string { return t.control }

// ID returns the id of a token by its content (an added token such as "<|im_end|>", or a
// vocabulary entry in its byte-level form).
func (t *Tokenizer) ID(token string) (int32, bool) {
	defer runtime.KeepAlive(t) // the tables may be in t's mapping (loadBinary)
	if id, ok := t.addedID[token]; ok {
		return id, true
	}
	return t.vocab.find(token)
}

// Encode turns text into token ids. Added tokens written in the text (such as "<|im_start|>")
// become their ids.
func (t *Tokenizer) Encode(text string) []int32 {
	defer runtime.KeepAlive(t)
	if t.plamo != nil {
		return t.plamo.encode(nil, strings.ToValidUTF8(text, "\uFFFD"))
	}
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
	if t.fallback != nil {
		return t.fallback.encode(t, ids, text)
	}
	if t.nfc {
		text = norm.NFC.String(text)
	}
	for _, piece := range splitStyle(text, t.style) {
		ids = append(ids, t.bpe(piece)...)
	}
	return ids
}

// bpe merges the byte-level form of one piece of text: the lowest ranked pair of neighbours, every
// time it is there from the left, until no pair merges.
func (t *Tokenizer) bpe(piece string) []int32 {
	if ids, ok := t.cache[piece]; ok {
		return ids
	}
	var ids []int32
	if t.ignoreMerges {
		var word strings.Builder
		for _, b := range []byte(piece) {
			word.WriteRune(byteToRune[b])
		}
		if id, ok := t.vocab.find(word.String()); ok {
			ids = []int32{id}
		}
	}
	if ids == nil {
		syms := make([]int32, len(piece))
		for i := 0; i < len(piece); i++ {
			syms[i] = t.byteSym[piece[i]]
		}
		for len(syms) > 1 {
			best, at, made := int32(-1), -1, int32(0)
			for i := 0; i+1 < len(syms); i++ {
				if r, id, ok := t.merges.find(syms[i], syms[i+1]); ok && (best < 0 || r < best) {
					best, at, made = r, i, id
				}
			}
			if at < 0 {
				break
			}
			// Merge every occurrence of the best pair, left to right.
			first, second := syms[at], syms[at+1]
			out := syms[:0]
			for i := 0; i < len(syms); {
				if i+1 < len(syms) && syms[i] == first && syms[i+1] == second {
					out = append(out, made)
					i += 2
				} else {
					out = append(out, syms[i])
					i++
				}
			}
			syms = out
		}
		n := int32(t.tokens.n())
		ids = make([]int32, 0, len(syms))
		for _, s := range syms {
			// a symbol that is no token cannot happen with a complete byte-level vocabulary
			if s < n {
				ids = append(ids, s)
			}
		}
	}
	t.cache[piece] = ids
	return ids
}

// Decode turns token ids back into text. Added tokens are written as their content unless
// skipSpecial is set.
func (t *Tokenizer) Decode(ids []int32, skipSpecial bool) string {
	defer runtime.KeepAlive(t)
	if t.fallback != nil {
		return t.fallback.decode(t, ids, skipSpecial)
	}
	if t.plamo != nil {
		return t.plamo.decode(t, ids, skipSpecial)
	}
	var out []byte
	for _, id := range ids {
		if id < 0 || int(id) >= t.tokens.n() {
			continue
		}
		if t.special[id] != 0 {
			if !skipSpecial {
				out = append(out, t.tokens.str(int(id))...)
			}
			continue
		}
		for _, r := range t.tokens.str(int(id)) {
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
func split(text string) []string { return splitStyle(text, gpt4) }

// splitStyle applies the pattern of a style: split's for gpt4, and for qwen35
//
//	(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
func splitStyle(text string, st style) []string {
	rs := []rune(text)
	var out []string
	if st == gpt2Digits {
		// each number character alone, GPT-2's pattern on the runs between them (the end of a
		// run is the end of the text for its \s+(?!\S))
		for i := 0; i < len(rs); {
			if isN(rs[i]) {
				out = append(out, string(rs[i]))
				i++
				continue
			}
			j := i
			for j < len(rs) && !isN(rs[j]) {
				j++
			}
			out = append(out, splitStyle(string(rs[i:j]), gpt2)...)
			i = j
		}
		return out
	}
	for i := 0; i < len(rs); {
		n := match(rs, i, st)
		out = append(out, string(rs[i:i+n]))
		i += n
	}
	return out
}

// matchGPT2 is match for GPT-2's pattern:
//
//	's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//
// the contractions in lower case only, and a letter, number or other run taking one space before
// it.
func matchGPT2(rs []rune, i int) int {
	rest := len(rs) - i
	if rs[i] == '\'' && rest >= 2 {
		for _, c := range []string{"s", "t", "re", "ve", "m", "ll", "d"} {
			if rest > len(c) && string(rs[i+1:i+1+len(c)]) == c {
				return 1 + len(c)
			}
		}
	}
	other := func(r rune) bool { return !isS(r) && !isL(r) && !isN(r) }
	j := i
	if rs[j] == ' ' && j+1 < len(rs) && !isS(rs[j+1]) {
		j++ // the space before a letter, number or other run
	}
	for _, class := range []func(rune) bool{isL, isN, other} {
		if class(rs[j]) {
			k := j
			for k < len(rs) && class(rs[k]) {
				k++
			}
			return k - i
		}
	}
	// \s+(?!\S): the whole run of whitespace at the end of the text, otherwise all but its last
	// character (which then goes with what follows, as the space of ' ?\p{L}+' or alone); \s+
	end := i
	for end < len(rs) && isS(rs[end]) {
		end++
	}
	if end == len(rs) || end-1 == i {
		return end - i
	}
	return end - 1 - i
}

func isL(r rune) bool  { return unicode.IsLetter(r) }
func isN(r rune) bool  { return unicode.IsNumber(r) }
func isS(r rune) bool  { return unicode.IsSpace(r) }
func isNL(r rune) bool { return r == '\r' || r == '\n' }
func isM(r rune) bool  { return unicode.IsMark(r) }

// o200k's two letter classes: [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}] and [\p{Ll}\p{Lm}\p{Lo}\p{M}].
func isUpperish(r rune) bool {
	return unicode.In(r, unicode.Lu, unicode.Lt, unicode.Lm, unicode.Lo, unicode.M)
}
func isLowerish(r rune) bool { return unicode.In(r, unicode.Ll, unicode.Lm, unicode.Lo, unicode.M) }

// contraction returns the length of (?i:'s|'t|'re|'ve|'m|'ll|'d) at j, or 0.
func contraction(rs []rune, j int) int {
	if j >= len(rs) || rs[j] != '\'' {
		return 0
	}
	for _, c := range []string{"s", "t", "re", "ve", "m", "ll", "d"} {
		if len(rs)-j > len(c) && strings.EqualFold(string(rs[j+1:j+1+len(c)]), c) {
			return 1 + len(c)
		}
	}
	return 0
}

// matchO200k is match for o200k's pattern (the style's comment), each alternative tried in
// order with a backtracking regular expression's choices: the optional first character taken
// first, the upper run as long as the lower run after it still has a character.
func matchO200k(rs []rune, i int) int {
	prefix := func(r rune) bool { return !isNL(r) && !isL(r) && !isN(r) }
	run := func(j int, f func(rune) bool) int {
		k := j
		for k < len(rs) && f(rs[k]) {
			k++
		}
		return k
	}
	// first: [U]*[Lw]+ from j, or -1. U* gives back characters until one starts the Lw run.
	first := func(j int) int {
		for u := run(j, isUpperish); u >= j; u-- {
			if u < len(rs) && isLowerish(rs[u]) {
				e := run(u, isLowerish)
				return e + contraction(rs, e)
			}
		}
		return -1
	}
	// second: [U]+[Lw]* from j, or -1.
	second := func(j int) int {
		u := run(j, isUpperish)
		if u == j {
			return -1
		}
		e := run(u, isLowerish)
		return e + contraction(rs, e)
	}
	for _, alt := range []func(int) int{first, second} {
		if prefix(rs[i]) && i+1 < len(rs) {
			if e := alt(i + 1); e >= 0 {
				return e - i
			}
		}
		if e := alt(i); e >= 0 {
			return e - i
		}
	}
	// \p{N}{1,3}
	if isN(rs[i]) {
		k := i
		for k < len(rs) && k-i < 3 && isN(rs[k]) {
			k++
		}
		return k - i
	}
	// ' ?[^\s\p{L}\p{N}]+[\r\n/]*'
	other := func(r rune) bool { return !isS(r) && !isL(r) && !isN(r) }
	j := i
	if rs[j] == ' ' && j+1 < len(rs) && other(rs[j+1]) {
		j++
	}
	if other(rs[j]) {
		j = run(j, other)
		j = run(j, func(r rune) bool { return isNL(r) || r == '/' })
		return j - i
	}
	// The remaining alternatives start with whitespace, as the GPT-4 pattern's.
	end := run(i, isS)
	if end == i {
		return 1
	}
	for k := end - 1; k >= i; k-- {
		if isNL(rs[k]) {
			return k + 1 - i
		}
	}
	if end == len(rs) || end-1 == i {
		return end - i
	}
	return end - 1 - i
}

func knownPattern(p string) bool {
	_, ok := patterns[p]
	return ok
}

// match returns the length of the match at i (at least 1) of the style's pattern.
func match(rs []rune, i int, st style) int {
	if st == gpt2 {
		return matchGPT2(rs, i)
	}
	if st == o200k {
		return matchO200k(rs, i)
	}
	// the letters of the second alternative: \p{L}, or [\p{L}\p{M}] in Qwen3.5's
	letter := isL
	digits := 3
	if st == qwen35 {
		letter = func(r rune) bool { return isL(r) || isM(r) }
		digits = 1
	}
	if st == qwen2 {
		digits = 1
	}
	rest := len(rs) - i
	// (?i:'s|'t|'re|'ve|'m|'ll|'d)
	if rs[i] == '\'' && rest >= 2 {
		for _, c := range []string{"s", "t", "re", "ve", "m", "ll", "d"} {
			if rest > len(c) && strings.EqualFold(string(rs[i+1:i+1+len(c)]), c) {
				return 1 + len(c)
			}
		}
	}
	// [^\r\n\p{L}\p{N}]?\p{L}+ (a mark may be the optional first character: it is not \p{L})
	letters := func(j int) int {
		k := j
		for k < len(rs) && letter(rs[k]) {
			k++
		}
		return k - j
	}
	if !isNL(rs[i]) && !isL(rs[i]) && !isN(rs[i]) && i+1 < len(rs) && letter(rs[i+1]) {
		return 1 + letters(i+1)
	}
	if letter(rs[i]) {
		return letters(i)
	}
	// \p{N}{1,3}, or \p{N}
	if isN(rs[i]) {
		k := i
		for k < len(rs) && k-i < digits && isN(rs[k]) {
			k++
		}
		return k - i
	}
	// ' ?[^\s\p{L}\p{N}]+[\r\n]*', and \p{M} also excluded in Qwen3.5's
	other := func(r rune) bool { return !isS(r) && !letter(r) && !isN(r) }
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
