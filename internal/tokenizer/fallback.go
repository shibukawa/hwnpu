package tokenizer

import (
	"bytes"
	"container/heap"
	"encoding/json"
	"fmt"
	"strings"
	"unicode/utf8"
)

// fallback is the BPE with byte fallback of a SentencePiece model converted to tokenizer.json
// (Gemma): the normalizer replaces spaces with ▁ and nothing splits the text further, so the
// text between added tokens is one piece. Its characters are the first symbols, a character
// outside the vocabulary becoming the byte tokens of its UTF-8 encoding, and pairs merge by
// rank, the lowest first and the leftmost on a tie, as the tokenizers library's Word does.
type fallback struct {
	merges map[[2]int32]merge
	byteID [256]int32
	byteOf map[int32]byte
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
	fb := &fallback{merges: map[[2]int32]merge{}, byteOf: map[int32]byte{}}
	for b := 0; b < 256; b++ {
		id, ok := t.vocab[fmt.Sprintf("<0x%02X>", b)]
		if !ok {
			return nil, fmt.Errorf("byte fallback without the byte token <0x%02X>", b)
		}
		fb.byteID[b], fb.byteOf[id] = id, byte(b)
	}
	if err := t.readMerges(f); err != nil {
		return nil, err
	}
	for pair, rank := range t.ranks {
		a, okA := t.vocab[pair[0]]
		b, okB := t.vocab[pair[1]]
		id, okM := t.vocab[pair[0]+pair[1]]
		if !okA || !okB || !okM {
			return nil, fmt.Errorf("merge %q %q is not in the vocabulary", pair[0], pair[1])
		}
		fb.merges[[2]int32{a, b}] = merge{rank, id}
	}
	t.ranks = nil
	t.fallback = fb
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
		if id, ok := t.vocab[string(r)]; ok {
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
		if m, ok := fb.merges[[2]int32{syms[l].id, syms[syms[l].next].id}]; ok {
			heap.Push(&q, fbPair{pos: l, rank: m.rank, id: m.id})
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
		if m, ok := fb.merges[[2]int32{l.id, r.id}]; !ok || m.id != p.id {
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
		if id < 0 || int(id) >= len(t.tokens) {
			continue
		}
		if b, ok := fb.byteOf[id]; ok {
			pending = append(pending, b)
			continue
		}
		flush()
		if t.special[id] {
			if !skipSpecial {
				out.WriteString(t.tokens[id])
			}
			continue
		}
		out.WriteString(strings.ReplaceAll(t.tokens[id], "▁", " "))
	}
	flush()
	return out.String()
}
