package tokenizer

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"unicode/utf8"
)

// plamo is PLaMo 2/3's tokenizer (tokenization_plamo.py's Plamo3Tokenizer): a vocabulary of
// scored pieces (tokenizer.jsonl, one [piece, score, type] a line) and a Viterbi search for the
// cover of the text with the least cost, the cost of a piece its score negated, a character no
// piece covers its UTF-8 bytes (the BYTE pieces <0xE3>) at a fixed cost. tokenization_plamo.py
// runs the search over an Aho-Corasick table; plamo finds the same pieces with a map of every
// piece's prefixes, and breaks ties the same way: of equal costs, the longest piece at a
// position, a character's bytes only when strictly cheaper than every piece.
//
// Before the search, the text is cut around special tokens (<|plamo:...|>), runs of a character
// repeated four times or more and runs of two spaces or more (tokenizer_config.json's
// break_around_repeated_chars_threshold and break_around_consecutive_spaces_threshold): no piece
// crosses a cut. tokenization_plamo.py marks the cuts with U+EE00, a piece of its own that is
// dropped, which is the same as searching the parts one by one.
type plamo struct {
	scores   map[string]int64 // piece → score × 10⁴, rounded as Python's round
	ids      map[string]int32 // piece → id, the last of equal pieces
	prefixes map[string]bool  // every prefix of a piece (the pieces included)
	bytes    [256]int32       // the BYTE pieces
	repeats  int              // break_around_repeated_chars_threshold (0: none)
	spaces   int              // break_around_consecutive_spaces_threshold (0: none)
}

const (
	plamoUnknown = -10000000 // tokenization_plamo.py's UNKNOWN_SCORE: a character's bytes
)

// LoadPlamo reads PLaMo's tokenizer.jsonl and the thresholds of the tokenizer_config.json beside
// it (none when it is missing).
func LoadPlamo(path string) (*Tokenizer, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cfg struct {
		Spaces  *int `json:"break_around_consecutive_spaces_threshold"`
		Repeats *int `json:"break_around_repeated_chars_threshold"`
	}
	if b, err := os.ReadFile(strings.TrimSuffix(path, "tokenizer.jsonl") + "tokenizer_config.json"); err == nil {
		if err := json.Unmarshal(b, &cfg); err != nil {
			return nil, fmt.Errorf("tokenizer: %s: tokenizer_config.json: %w", path, err)
		}
	}
	spaces, repeats := 0, 0
	if cfg.Spaces != nil {
		spaces = *cfg.Spaces
	}
	if cfg.Repeats != nil {
		repeats = *cfg.Repeats
	}
	t, err := parsePlamo(raw, spaces, repeats)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: %s: %w", path, err)
	}
	return t, nil
}

func parsePlamo(raw []byte, spaces, repeats int) (*Tokenizer, error) {
	p := &plamo{scores: map[string]int64{}, ids: map[string]int32{}, prefixes: map[string]bool{}, spaces: spaces, repeats: repeats}
	t := &Tokenizer{vocab: map[string]int32{}, addedID: map[string]int32{}, plamo: p}
	for i := range p.bytes {
		p.bytes[i] = -1
	}
	sc := bufio.NewScanner(bytes.NewReader(raw))
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var row []any
		if err := json.Unmarshal(line, &row); err != nil {
			return nil, fmt.Errorf("line %d: %w", len(t.tokens)+1, err)
		}
		piece, ok1 := row[0].(string)
		score, ok2 := row[1].(float64)
		if len(row) < 2 || !ok1 || !ok2 {
			return nil, fmt.Errorf("line %d is not [piece, score, ...]", len(t.tokens)+1)
		}
		kind := ""
		if len(row) > 2 {
			kind, _ = row[2].(string)
		}
		id := int32(len(t.tokens))
		t.tokens = append(t.tokens, piece)
		t.special = append(t.special, kind == "CONTROL" || kind == "UNKNOWN")
		t.vocab[piece] = id
		if kind == "BYTE" {
			var b int
			if len(piece) != 6 || !strings.HasPrefix(piece, "<0x") || !strings.HasSuffix(piece, ">") {
				return nil, fmt.Errorf("byte piece %q is not <0xXX>", piece)
			}
			if _, err := fmt.Sscanf(piece[3:5], "%02X", &b); err != nil {
				return nil, fmt.Errorf("byte piece %q: %w", piece, err)
			}
			p.bytes[b] = id
			continue
		}
		if kind == "CONTROL" || kind == "UNKNOWN" {
			t.control = append(t.control, piece)
		}
		p.ids[piece] = id
		p.scores[piece] = int64(math.RoundToEven(score * 1e4))
		for j := range piece {
			if j > 0 {
				p.prefixes[piece[:j]] = true
			}
		}
		p.prefixes[piece] = true
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	for b, id := range p.bytes {
		if id < 0 {
			return nil, fmt.Errorf("no byte piece <0x%02X>", b)
		}
	}
	return t, nil
}

// encode appends the ids of text.
func (p *plamo) encode(ids []int32, text string) []int32 {
	for _, part := range p.cut(text) {
		ids = p.search(ids, part)
	}
	return ids
}

// cut splits text where tokenization_plamo.py puts U+EE00: around each special token
// <|plamo:...|> (up to 64 characters, no '|' or space within), then around each run of a
// character (other than a newline) repeated `repeats` times or more and each run of `spaces`
// spaces or more, the first that matches at a position, as Python's re.sub finds them.
func (p *plamo) cut(text string) []string {
	var parts []string
	start := 0
	add := func(from, to int) {
		if to > from {
			parts = append(parts, text[from:to])
		}
	}
	for i := 0; i < len(text); {
		if n := specialAt(text[i:]); n > 0 {
			p.cutRuns(&parts, text[start:i])
			add(i, i+n)
			i += n
			start = i
			continue
		}
		_, n := utf8.DecodeRuneInString(text[i:])
		i += n
	}
	p.cutRuns(&parts, text[start:])
	return parts
}

// specialAt is the length of the special token text starts with (0: none).
func specialAt(s string) int {
	const open = "<|plamo:"
	if !strings.HasPrefix(s, open) {
		return 0
	}
	n := 0
	for i, r := range s[len(open):] {
		if r == '|' {
			if strings.HasPrefix(s[len(open)+i:], "|>") {
				return len(open) + i + 2
			}
			return 0
		}
		if isPySpace(r) || n == 64 {
			return 0
		}
		n++
	}
	return 0
}

// isPySpace is Python's \s on a str pattern (Unicode whitespace, as str.isspace).
func isPySpace(r rune) bool {
	switch r {
	case ' ', '\t', '\n', '\v', '\f', '\r', 0x1c, 0x1d, 0x1e, 0x1f, 0x85, 0xa0, 0x1680, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000:
		return true
	}
	return r >= 0x2000 && r <= 0x200a
}

// cutRuns appends the parts of text cut around the runs of repeated characters and of spaces.
func (p *plamo) cutRuns(parts *[]string, text string) {
	start := 0
	for i := 0; i < len(text); {
		r, n := utf8.DecodeRuneInString(text[i:])
		// the length of the run of r from i, in bytes and characters
		end, count := i, 0
		for end < len(text) {
			q, m := utf8.DecodeRuneInString(text[end:])
			if q != r {
				break
			}
			end += m
			count++
		}
		if (p.repeats > 0 && r != '\n' && count >= p.repeats) || (p.spaces > 0 && r == ' ' && count >= p.spaces) {
			if i > start {
				*parts = append(*parts, text[start:i])
			}
			*parts = append(*parts, text[i:end])
			i, start = end, end
			continue
		}
		i += n
	}
	if len(text) > start {
		*parts = append(*parts, text[start:])
	}
}

// search appends the ids of the cheapest cover of text (tokenization_plamo.py's _encode).
func (p *plamo) search(ids []int32, text string) []int32 {
	n := len(text)
	cost := make([]int64, n+1) // the least cost from each byte offset to the end
	next := make([]int, n+1)   // the length in bytes of the piece there, its bytes if negative
	id := make([]int32, n+1)
	var lens []int
	for i := n - 1; i >= 0; i-- {
		if !utf8.RuneStart(text[i]) {
			continue
		}
		cost[i] = math.MaxInt64
		// every piece from i, the longest first
		lens = lens[:0]
		for j := i; j < n; {
			_, m := utf8.DecodeRuneInString(text[j:])
			j += m
			if !p.prefixes[text[i:j]] {
				break
			}
			if _, ok := p.scores[text[i:j]]; ok {
				lens = append(lens, j-i)
			}
		}
		for k := len(lens) - 1; k >= 0; k-- {
			l := lens[k]
			if c := cost[i+l] - p.scores[text[i:i+l]]; c < cost[i] {
				cost[i], next[i], id[i] = c, l, p.ids[text[i:i+l]]
			}
		}
		_, m := utf8.DecodeRuneInString(text[i:])
		if c := cost[i+m] - plamoUnknown; c < cost[i] {
			cost[i], next[i], id[i] = c, m, -1
		}
	}
	for i := 0; i < n; i += next[i] {
		if id[i] >= 0 {
			ids = append(ids, id[i])
			continue
		}
		for _, b := range []byte(text[i : i+next[i]]) {
			ids = append(ids, p.bytes[b])
		}
	}
	return ids
}

// decode writes the pieces of ids, a BYTE piece as its byte; special pieces (CONTROL, UNKNOWN)
// only unless skipSpecial.
func (p *plamo) decode(t *Tokenizer, ids []int32, skipSpecial bool) string {
	var out []byte
	for _, id := range ids {
		if id < 0 || int(id) >= len(t.tokens) {
			continue
		}
		if t.special[id] && skipSpecial {
			continue
		}
		s := t.tokens[id]
		if len(s) == 6 && strings.HasPrefix(s, "<0x") && p.bytes[hexByte(s[3:5])] == id {
			out = append(out, byte(hexByte(s[3:5])))
			continue
		}
		out = append(out, s...)
	}
	return strings.ToValidUTF8(string(out), "�")
}

func hexByte(s string) int {
	v := 0
	for _, c := range s {
		v <<= 4
		switch {
		case c >= '0' && c <= '9':
			v |= int(c - '0')
		case c >= 'a' && c <= 'f':
			v |= int(c-'a') + 10
		case c >= 'A' && c <= 'F':
			v |= int(c-'A') + 10
		default:
			return 0
		}
	}
	return v
}
