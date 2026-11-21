package tokenizer

import (
	"encoding/json"
	"strings"
	"unicode/utf8"
)

// fastModel turns the fast reading of a tokenizer.json's vocabulary and merges off (tests compare
// the two readings).
var fastModel = true

// splitModel reads model.vocab and model.merges of a tokenizer.json by themselves and gives the
// document without them (an empty object and an empty array in their places), for encoding/json to
// read the rest of. They are nearly all of the file (Gemma 2's: 32 MB, 256000 tokens and 580000
// merges), and reading them as a map and as raw messages decoded one by one took 0.9 s of every
// start of a Sana picture on an M1; read here in one pass over their bytes, 0.2 s (docs/design/image-apple.md). ok is false for anything this does not read exactly as encoding/json would (a
// vocabulary that is not an object of strings to integers, a merge that is neither "a b" nor
// ["a", "b"], a key written twice, a string that is not valid UTF-8): the caller then reads the
// whole document with encoding/json, as before.
func splitModel(raw []byte) (small []byte, vocab map[string]int32, pairs [][2]string, ok bool) {
	if !fastModel {
		return nil, nil, nil, false
	}
	// the document's members, model's own scanned for the two rather than skipped
	i := skipSpace(raw, 0)
	if i >= len(raw) || raw[i] != '{' {
		return nil, nil, nil, false
	}
	var in []span
	for i = skipSpace(raw, i+1); ; {
		lo, hi, val, ok := memberKey(raw, i)
		if !ok {
			return nil, nil, nil, false
		}
		end := 0
		if string(raw[lo:hi]) == "model" {
			if in != nil { // twice: encoding/json would take the last
				return nil, nil, nil, false
			}
			if in, end, ok = members(raw, val, "vocab", "merges"); !ok {
				return nil, nil, nil, false
			}
		} else if end = valueEnd(raw, val); end < 0 {
			return nil, nil, nil, false
		}
		if i, ok = nextMember(raw, end); !ok {
			return nil, nil, nil, false
		}
		if i == 0 {
			break
		}
	}
	if in == nil || raw[in[0].lo] != '{' || raw[in[1].lo] != '[' {
		return nil, nil, nil, false
	}
	v, m := in[0], in[1]
	// each as a string of its own, which its tokens are substrings of: the vocabulary's stays
	// with the tokenizer, the merges' as long as their pairs do
	if vocab, ok = readVocab(string(raw[v.lo:v.hi])); !ok {
		return nil, nil, nil, false
	}
	if pairs, ok = readPairs(string(raw[m.lo:m.hi])); !ok {
		return nil, nil, nil, false
	}
	a, b, fa, fb := v, m, "{}", "[]"
	if b.lo < a.lo {
		a, b, fa, fb = b, a, fb, fa
	}
	small = make([]byte, 0, len(raw)-(a.hi-a.lo)-(b.hi-b.lo)+4)
	small = append(small, raw[:a.lo]...)
	small = append(small, fa...)
	small = append(small, raw[a.hi:b.lo]...)
	small = append(small, fb...)
	small = append(small, raw[b.hi:]...)
	return small, vocab, pairs, true
}

// span is a value's bytes s[lo:hi].
type span struct{ lo, hi int }

// chars is a JSON text to scan: the file's bytes, or a part of it as a string.
type chars interface{ ~string | ~[]byte }

func skipSpace[S chars](s S, i int) int {
	for i < len(s) && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r') {
		i++
	}
	return i
}

// stringEnd is the index of the closing quote of the string that opens at s[i]; plain says it has
// no escape. -1: it does not close, or holds a control character.
func stringEnd[S chars](s S, i int) (end int, plain bool) {
	plain = true
	for j := i + 1; j < len(s); j++ {
		switch c := s[j]; {
		case c == '"':
			return j, plain
		case c == '\\':
			plain = false
			j++
		case c < 0x20:
			return -1, false
		}
	}
	return -1, false
}

// valueEnd is the index past the value that starts at s[i] (-1: malformed).
func valueEnd[S chars](s S, i int) int {
	if i >= len(s) {
		return -1
	}
	switch s[i] {
	case '"':
		end, _ := stringEnd(s, i)
		if end < 0 {
			return -1
		}
		return end + 1
	case '{', '[':
		depth := 0
		for j := i; j < len(s); j++ {
			switch s[j] {
			case '"':
				end, _ := stringEnd(s, j)
				if end < 0 {
					return -1
				}
				j = end
			case '{', '[':
				depth++
			case '}', ']':
				if depth--; depth == 0 {
					return j + 1
				}
			}
		}
		return -1
	}
	j := i
	for j < len(s) {
		if c := s[j]; c == ',' || c == '}' || c == ']' || c == ' ' || c == '\n' || c == '\t' || c == '\r' {
			break
		}
		j++
	}
	if j == i {
		return -1
	}
	return j
}

// memberKey reads the member that starts at s[i] as far as its value: its key s[lo:hi] and where
// the value starts. ok is false for what is not a member, or a key with an escape (which could
// spell a key looked for another way).
func memberKey[S chars](s S, i int) (lo, hi, val int, ok bool) {
	if i >= len(s) || s[i] != '"' {
		return 0, 0, 0, false
	}
	end, plain := stringEnd(s, i)
	if end < 0 || !plain {
		return 0, 0, 0, false
	}
	val = skipSpace(s, end+1)
	if val >= len(s) || s[val] != ':' {
		return 0, 0, 0, false
	}
	return i + 1, end, skipSpace(s, val+1), true
}

// nextMember is where the member after a value that ended at s[i] starts, 0 when the object
// closes there.
func nextMember[S chars](s S, i int) (next int, ok bool) {
	i = skipSpace(s, i)
	if i >= len(s) {
		return 0, false
	}
	switch s[i] {
	case '}':
		return 0, true
	case ',':
		return skipSpace(s, i+1), true
	}
	return 0, false
}

// members are the values of keys in the object that opens at s[i], and the index past the object.
// ok is false when the object is malformed, misses one of the keys or has it twice (encoding/json
// would take the last).
func members[S chars](s S, i int, keys ...string) (vs []span, end int, ok bool) {
	if i >= len(s) || s[i] != '{' {
		return nil, 0, false
	}
	vs = make([]span, len(keys))
	found := 0
	for i = skipSpace(s, i+1); ; {
		lo, hi, val, ok := memberKey(s, i)
		if !ok {
			return nil, 0, false
		}
		end := valueEnd(s, val)
		if end < 0 {
			return nil, 0, false
		}
		for k, key := range keys {
			if string(s[lo:hi]) == key {
				if vs[k].hi != 0 {
					return nil, 0, false
				}
				vs[k] = span{val, end}
				found++
			}
		}
		if i, ok = nextMember(s, end); !ok {
			return nil, 0, false
		}
		if i == 0 {
			// the closing brace is the first byte that is not a space after the last value
			return vs, skipSpace(s, end) + 1, found == len(keys)
		}
	}
}

// text is the string that opens at s[i], and the index past its closing quote (-1: not a string
// encoding/json reads the same).
func text(s string, i int) (string, int) {
	if i >= len(s) || s[i] != '"' {
		return "", -1
	}
	end, plain := stringEnd(s, i)
	if end < 0 {
		return "", -1
	}
	if plain {
		if t := s[i+1 : end]; utf8.ValidString(t) {
			return t, end + 1
		}
		return "", -1 // encoding/json replaces what is not UTF-8
	}
	if !utf8.ValidString(s[i : end+1]) {
		return "", -1
	}
	var t string
	if json.Unmarshal([]byte(s[i:end+1]), &t) != nil {
		return "", -1
	}
	return t, end + 1
}

// readVocab reads an object of tokens to their ids.
func readVocab(s string) (map[string]int32, bool) {
	vocab := make(map[string]int32, len(s)/20)
	i := skipSpace(s, 1)
	if i < len(s) && s[i] == '}' {
		return vocab, true
	}
	for {
		k, j := text(s, i)
		if j < 0 {
			return nil, false
		}
		i = skipSpace(s, j)
		if i >= len(s) || s[i] != ':' {
			return nil, false
		}
		i = skipSpace(s, i+1)
		neg := false
		if i < len(s) && s[i] == '-' {
			neg = true
			i++
		}
		n, digits := int64(0), 0
		for ; i < len(s) && s[i] >= '0' && s[i] <= '9'; i++ {
			if n = n*10 + int64(s[i]-'0'); n > 1<<31 {
				return nil, false
			}
			digits++
		}
		if neg {
			n = -n
		}
		if digits == 0 || n > 1<<31-1 || (digits > 1 && s[i-digits] == '0') {
			return nil, false
		}
		vocab[k] = int32(n)
		i = skipSpace(s, i)
		if i >= len(s) {
			return nil, false
		}
		if s[i] == '}' {
			return vocab, i == len(s)-1
		}
		if s[i] != ',' { // a fraction or an exponent too
			return nil, false
		}
		i = skipSpace(s, i+1)
	}
}

// readPairs reads an array of merges, each "a b" or ["a", "b"].
func readPairs(s string) ([][2]string, bool) {
	pairs := make([][2]string, 0, len(s)/16)
	i := skipSpace(s, 1)
	if i < len(s) && s[i] == ']' {
		return pairs, true
	}
	for {
		if i >= len(s) {
			return nil, false
		}
		var pair [2]string
		switch s[i] {
		case '"':
			t, j := text(s, i)
			if j < 0 {
				return nil, false
			}
			a, b, ok := strings.Cut(t, " ")
			if !ok {
				return nil, false
			}
			pair, i = [2]string{a, b}, j
		case '[':
			a, j := text(s, skipSpace(s, i+1))
			if j < 0 {
				return nil, false
			}
			if j = skipSpace(s, j); j >= len(s) || s[j] != ',' {
				return nil, false
			}
			b, j := text(s, skipSpace(s, j+1))
			if j < 0 {
				return nil, false
			}
			if j = skipSpace(s, j); j >= len(s) || s[j] != ']' {
				return nil, false
			}
			pair, i = [2]string{a, b}, j+1
		default:
			return nil, false
		}
		pairs = append(pairs, pair)
		i = skipSpace(s, i)
		if i >= len(s) {
			return nil, false
		}
		if s[i] == ']' {
			return pairs, i == len(s)-1
		}
		if s[i] != ',' {
			return nil, false
		}
		i = skipSpace(s, i+1)
	}
}
