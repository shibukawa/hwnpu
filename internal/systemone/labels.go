package systemone

import (
	"fmt"
	"slices"
)

// Labels returns the codes the prompt gives the options of q and, for each option, the tokens
// whose logits stand for it at the prompt's last token (the best of them counts), as llama.cpp's
// d1_labels: a noul option is yes or no in three cases, a score level its digit, and a choice
// option its code alone or after a space. Choice codes are the options' keys when they are all
// single letters, else A, B, ... (AA, AB, ... after Z); a code that is not one token of its own,
// or whose token another option has, is replaced by the first free one of a pool. encode
// tokenizes text without special tokens.
func Labels(q Question, encode func(string) []int32) (codes []string, groups [][]int32, err error) {
	single := func(forms ...string) []int32 {
		var out []int32
		for _, f := range forms {
			if t := encode(f); len(t) == 1 && !slices.Contains(out, t[0]) {
				out = append(out, t[0])
			}
		}
		return out
	}
	if q.Type != "choice" {
		for _, o := range q.Options {
			var g []int32
			switch {
			case q.Type == "score":
				g = single(o.Key)
			case o.Key == "true":
				g = single("yes", "Yes", "YES")
			default:
				g = single("no", "No", "NO")
			}
			if len(g) == 0 {
				return nil, nil, fmt.Errorf("systemone: the label of %s is not a single token", o.Key)
			}
			codes = append(codes, o.Key)
			groups = append(groups, g)
		}
		return codes, groups, nil
	}

	n := len(q.Options)
	letters := true
	for _, o := range q.Options {
		letters = letters && len(o.Key) == 1 && (o.Key[0] >= 'A' && o.Key[0] <= 'Z' || o.Key[0] >= 'a' && o.Key[0] <= 'z')
	}
	want := make([]string, n)
	for i := range want {
		switch {
		case letters:
			want[i] = q.Options[i].Key
		case i < 26:
			want[i] = string(rune('A' + i))
		default:
			want[i] = string([]byte{byte('A' + (i-26)/26), byte('A' + (i-26)%26)})
		}
	}
	var pool []string
	for c := 'A'; c <= 'Z'; c++ {
		pool = append(pool, string(c))
	}
	for i := 0; i < 100; i++ {
		pool = append(pool, fmt.Sprintf("%02d", i))
	}
	for c := 'a'; c <= 'z'; c++ {
		pool = append(pool, string(c))
	}
	for i := 0; i < 200; i++ {
		pool = append(pool, fmt.Sprintf("#%d", i))
	}
	for a := 'A'; a <= 'Z'; a++ {
		for b := 'A'; b <= 'Z'; b++ {
			pool = append(pool, string([]rune{a, b}))
		}
	}
	var used []int32
	take := func(code string) bool {
		t := encode(code)
		if len(t) != 1 || slices.Contains(used, t[0]) {
			return false
		}
		used = append(used, t[0])
		g := []int32{t[0]}
		for _, s := range single(" " + code) {
			if s != t[0] {
				g = append(g, s)
			}
		}
		codes = append(codes, code)
		groups = append(groups, g)
		return true
	}
	for _, code := range want {
		taken := take(code)
		for i := 0; !taken && i < len(pool); i++ {
			taken = take(pool[i])
		}
		if !taken {
			return nil, nil, bad("no single-token label left for %d options", n)
		}
	}
	return codes, groups, nil
}
