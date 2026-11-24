package tokenizer

import (
	"bytes"
	"container/heap"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"unicode/utf8"
)

// fallback is the BPE with byte fallback of a SentencePiece model converted to tokenizer.json
// (Gemma): the normalizer replaces spaces with ▁ and nothing splits the text further, so the
// text between added tokens is one piece. Its characters are the first symbols, a character
// outside the vocabulary becoming the byte tokens of its UTF-8 encoding, and pairs merge by
// rank, the lowest first and the leftmost on a tie, as the tokenizers library's Word does.
type fallback struct {
	merges  *pairTable // pairs of ids → rank and the id they make
	byteID  [256]int32 // the id each byte falls back to
	byteTok [256]int32 // the id of each byte's token <0xNN>, -1 where a character stands for it
	byteOf  map[int32]byte
}

// newFallback is the fallback of merges and the byte tokens (byteTok).
func newFallback(merges *pairTable, byteID, byteTok [256]int32) *fallback {
	fb := &fallback{merges: merges, byteID: byteID, byteTok: byteTok, byteOf: map[int32]byte{}}
	for b, id := range byteTok {
		if id >= 0 {
			fb.byteOf[id] = byte(b)
		}
	}
	return fb
}

type merge struct {
	rank int
	id   int32
}

// parseFallback reads a tokenizer.json with byte fallback. It accepts the normalizer, pre-tokenizer
// and decoder that Gemma's has and rejects others.
func parseFallback(raw []byte, f *file) (*Tokenizer, error) {
	var g struct {
		Normalizer *struct {
			Type    string `json:"type"`
			Pattern struct {
				String *string `json:"String"`
			} `json:"pattern"`
			Content string `json:"content"`
		} `json:"normalizer"`
		PreTokenizer *struct {
			Type    string `json:"type"`
			Pattern struct {
				String *string `json:"String"`
			} `json:"pattern"`
			Behavior string `json:"behavior"`
			Invert   bool   `json:"invert"`
		} `json:"pre_tokenizer"`
		Decoder json.RawMessage `json:"decoder"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		return nil, err
	}
	n := g.Normalizer
	if n == nil || n.Type != "Replace" || n.Pattern.String == nil || *n.Pattern.String != " " || n.Content != "▁" {
		return nil, fmt.Errorf("normalizer %s is not supported with byte fallback; only replacing spaces with ▁", f.Normalizer)
	}
	// Splitting at spaces finds none once they are ▁.
	if pt := g.PreTokenizer; pt != nil && (pt.Type != "Split" || pt.Pattern.String == nil || *pt.Pattern.String != " " || pt.Invert) {
		return nil, fmt.Errorf("pre-tokenizer %s is not supported with byte fallback", pt.Type)
	}
	const decoder = `{"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"}]}`
	var compact bytes.Buffer
	if err := json.Compact(&compact, g.Decoder); err != nil || compact.String() != decoder {
		return nil, fmt.Errorf("decoder %s is not supported with byte fallback", g.Decoder)
	}

	t := newTokenizer(f)
	vocab := f.Model.Vocab
	var byteID, byteTok [256]int32
	for b := 0; b < 256; b++ {
		byteTok[b] = -1
		id, ok := vocab[fmt.Sprintf("<0x%02X>", b)]
		if !ok {
			// Gemma 2's vocabulary has "\t" and no <0x09>: a byte that is a character of
			// the vocabulary never falls back, as encode looks the character up first
			if c, single := vocab[string(rune(b))]; b < 0x80 && single {
				byteID[b] = c
				continue
			}
			return nil, fmt.Errorf("byte fallback without the byte token <0x%02X>", b)
		}
		byteID[b], byteTok[b] = id, id
	}
	merges := map[[2]int32]merge{}
	// add notes the merge of pair at rank, unless first finds the pair already there.
	var joined []byte
	add := func(pair [2]string, rank int, first bool) error {
		a, okA := vocab[pair[0]]
		b, okB := vocab[pair[1]]
		joined = append(append(joined[:0], pair[0]...), pair[1]...)
		id, okM := vocab[string(joined)]
		if !okA || !okB || !okM {
			return fmt.Errorf("merge %q %q is not in the vocabulary", pair[0], pair[1])
		}
		key := [2]int32{a, b}
		if _, dup := merges[key]; !dup || !first {
			merges[key] = merge{rank, id}
		}
		return nil
	}
	// The pairs as splitModel read them go in without their ranks by their strings first: a pair
	// written twice keeps its first rank as in readMerges, since two pairs of strings are one
	// pair of ids only when they are the same strings. That needs every id to be of one token.
	direct := f.pairs != nil
	if direct {
		seen := make([]bool, t.tokens.n())
		for _, id := range vocab {
			if id < 0 || seen[id] {
				direct = false
				break
			}
			seen[id] = true
		}
	}
	if direct {
		merges = make(map[[2]int32]merge, len(f.pairs))
		for rank, pair := range f.pairs {
			if err := add(pair, rank, true); err != nil {
				return nil, err
			}
		}
	} else {
		pairs, err := pairsOf(f)
		if err != nil {
			return nil, err
		}
		ranks := map[[2]string]int{}
		for i, pair := range pairs {
			if _, dup := ranks[pair]; !dup {
				ranks[pair] = i
			}
		}
		for pair, rank := range ranks {
			if err := add(pair, rank, false); err != nil {
				return nil, err
			}
		}
	}
	entries := make([]pairEntry, 0, len(merges))
	for k, m := range merges {
		entries = append(entries, pairEntry{k[0], k[1], int32(m.rank), m.id})
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].rank < entries[j].rank })
	t.fallback = newFallback(newPairTable(entries), byteID, byteTok)
	return t, nil
}

type fbSymbol struct {
	id         int32
	prev, next int
	gone       bool // merged into its left neighbor
}

type fbPair struct {
	pos, rank int
	id        int32
}

// fbQueue pops the lowest rank first and, on a tie, the leftmost pair.
type fbQueue []fbPair

func (q fbQueue) Len() int { return len(q) }
func (q fbQueue) Less(i, j int) bool {
	return q[i].rank < q[j].rank || q[i].rank == q[j].rank && q[i].pos < q[j].pos
}
func (q fbQueue) Swap(i, j int) { q[i], q[j] = q[j], q[i] }
func (q *fbQueue) Push(x any)   { *q = append(*q, x.(fbPair)) }
func (q *fbQueue) Pop() any {
	old := *q
	x := old[len(old)-1]
	*q = old[:len(old)-1]
	return x
}

// encode normalizes and merges one stretch of text with no added token in it.
func (fb *fallback) encode(t *Tokenizer, ids []int32, text string) []int32 {
	if text == "" {
		return ids
	}
	var syms []fbSymbol
	add := func(id int32) {
		syms = append(syms, fbSymbol{id: id, prev: len(syms) - 1, next: len(syms) + 1})
	}
	var buf [utf8.UTFMax]byte
	for _, r := range text {
		if r == ' ' {
			r = '▁'
		}
		if id, ok := t.vocab.find(string(r)); ok {
			add(id)
			continue
		}
		for _, b := range buf[:utf8.EncodeRune(buf[:], r)] {
			add(fb.byteID[b])
		}
	}
	syms[len(syms)-1].next = -1

	var q fbQueue
	push := func(l int) {
		if l < 0 || syms[l].next < 0 {
			return
		}
		if rank, id, ok := fb.merges.find(syms[l].id, syms[syms[l].next].id); ok {
			heap.Push(&q, fbPair{pos: l, rank: int(rank), id: id})
		}
	}
	for i := range syms {
		push(i)
	}
	for q.Len() > 0 {
		p := heap.Pop(&q).(fbPair)
		l := &syms[p.pos]
		if l.gone || l.next < 0 {
			continue
		}
		r := &syms[l.next]
		if _, id, ok := fb.merges.find(l.id, r.id); !ok || id != p.id {
			continue // stale: one side merged since
		}
		l.id, l.next, r.gone = p.id, r.next, true
		if l.next >= 0 {
			syms[l.next].prev = p.pos
		}
		push(l.prev)
		push(p.pos)
	}
	for i := 0; i >= 0; i = syms[i].next {
		ids = append(ids, syms[i].id)
	}
	return ids
}

// decode turns ids back into text: ▁ becomes a space, and runs of byte tokens their UTF-8
// characters (a byte that does not form one becomes U+FFFD).
func (fb *fallback) decode(t *Tokenizer, ids []int32, skipSpecial bool) string {
	var out strings.Builder
	var pending []byte
	flush := func() {
		for b := pending; len(b) > 0; {
			r, n := utf8.DecodeRune(b)
			out.WriteRune(r)
			b = b[n:]
		}
		pending = pending[:0]
	}
	for _, id := range ids {
		if id < 0 || int(id) >= t.tokens.n() {
			continue
		}
		if b, ok := fb.byteOf[id]; ok {
			pending = append(pending, b)
			continue
		}
		flush()
		if t.special[id] != 0 {
			if !skipSpecial {
				out.WriteString(t.tokens.str(int(id)))
			}
			continue
		}
		out.WriteString(strings.ReplaceAll(t.tokens.str(int(id)), "▁", " "))
	}
	flush()
	return out.String()
}
