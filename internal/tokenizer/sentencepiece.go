package tokenizer

import (
	"container/heap"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"unicode/utf8"
)

// SentencePiece is a loaded SentencePiece BPE tokenizer (a tokenizer.model), as PLaMo ships it.
//
// It encodes as the sentencepiece library does, wrapped the way a Hugging Face slow tokenizer
// wraps it: the special pieces (control symbols and the unknown piece) are matched in the raw
// text first, and each stretch between them is normalized (spaces become ▁ and a ▁ is
// prepended), cut into characters and user-defined symbols, and merged pair by pair, highest
// piece score first and the leftmost pair on a tie; a symbol left outside the vocabulary
// becomes its UTF-8 bytes. Load rejects what it does not implement: Unigram, a normalization
// rule table, removing extra whitespace, whitespace as a suffix, unused pieces, and BPE without
// byte fallback.
type SentencePiece struct {
	pieces      []string // by id
	special     []bool   // by id: control and unknown pieces
	vocab       map[string]int32
	scores      []float32 // by id
	bytes       [256]int32
	byteOf      []int16  // by id: the byte of a <0xNN> piece, or -1
	user        []string // user-defined pieces, longest first
	specials    []string // longest first
	specialID   map[string]int32
	pieceID     map[string]int32 // every piece, for ID
	dummyPrefix bool
}

// The piece types of sentencepiece_model.proto.
const (
	spNormal      = 1
	spUnknown     = 2
	spControl     = 3
	spUserDefined = 4
	spUnused      = 5
	spByte        = 6
)

// LoadSentencePiece reads a tokenizer.model.
func LoadSentencePiece(path string) (*SentencePiece, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	t, err := parseSentencePiece(raw)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: %s: %w", path, err)
	}
	return t, nil
}

func parseSentencePiece(raw []byte) (*SentencePiece, error) {
	t := &SentencePiece{vocab: map[string]int32{}, specialID: map[string]int32{}, pieceID: map[string]int32{}}
	modelType, byteFallback, suffix := 1, false, false // UNIGRAM is the proto's default
	dummyPrefix, removeSpaces, escapeSpaces := true, true, true
	var charsmap, denormCharsmap []byte
	err := protoFields(raw, func(f int, v uint64, b []byte) error {
		switch f {
		case 1:
			piece, score, typ := "", float32(0), uint64(spNormal)
			if err := protoFields(b, func(f int, v uint64, b []byte) error {
				switch f {
				case 1:
					piece = string(b)
				case 2:
					score = math.Float32frombits(uint32(v))
				case 3:
					typ = v
				}
				return nil
			}); err != nil {
				return err
			}
			id := int32(len(t.pieces))
			t.pieces = append(t.pieces, piece)
			t.scores = append(t.scores, score)
			t.special = append(t.special, typ == spControl || typ == spUnknown)
			t.byteOf = append(t.byteOf, -1)
			if _, dup := t.pieceID[piece]; dup {
				return fmt.Errorf("piece %q is defined twice", piece)
			}
			t.pieceID[piece] = id
			switch typ {
			case spNormal:
				t.vocab[piece] = id
			case spUserDefined:
				t.vocab[piece] = id
				t.user = append(t.user, piece)
			case spControl, spUnknown:
				t.specials = append(t.specials, piece)
				t.specialID[piece] = id
			case spByte:
				var c byte
				if _, err := fmt.Sscanf(piece, "<0x%02X>", &c); err != nil || piece != fmt.Sprintf("<0x%02X>", c) {
					return fmt.Errorf("byte piece %q", piece)
				}
				t.bytes[c] = id + 1 // 0 marks a missing byte until the check below
				t.byteOf[id] = int16(c)
			case spUnused:
				return fmt.Errorf("unused piece %q is not supported", piece)
			default:
				return fmt.Errorf("piece %q has unknown type %d", piece, typ)
			}
		case 2:
			return protoFields(b, func(f int, v uint64, b []byte) error {
				switch f {
				case 3:
					modelType = int(v)
				case 24:
					suffix = v != 0
				case 35:
					byteFallback = v != 0
				}
				return nil
			})
		case 3:
			return protoFields(b, func(f int, v uint64, b []byte) error {
				switch f {
				case 2:
					charsmap = b
				case 3:
					dummyPrefix = v != 0
				case 4:
					removeSpaces = v != 0
				case 5:
					escapeSpaces = v != 0
				}
				return nil
			})
		case 5:
			return protoFields(b, func(f int, v uint64, b []byte) error {
				if f == 2 {
					denormCharsmap = b
				}
				return nil
			})
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	switch {
	case modelType != 2:
		return nil, fmt.Errorf("model type %d is not supported; only BPE (2)", modelType)
	case !byteFallback:
		return nil, errors.New("BPE without byte fallback is not supported")
	case len(charsmap) > 0 || len(denormCharsmap) > 0:
		return nil, errors.New("a normalization rule table is not supported; only the identity normalizer")
	case removeSpaces:
		return nil, errors.New("remove_extra_whitespaces is not supported")
	case !escapeSpaces || suffix:
		return nil, errors.New("only whitespace escaped as a ▁ prefix is supported")
	}
	for c, id := range t.bytes {
		if id == 0 {
			return nil, fmt.Errorf("byte piece <0x%02X> is missing", c)
		}
		t.bytes[c] = id - 1
	}
	t.dummyPrefix = dummyPrefix
	sort.SliceStable(t.user, func(i, j int) bool { return len(t.user[i]) > len(t.user[j]) })
	sort.SliceStable(t.specials, func(i, j int) bool { return len(t.specials[i]) > len(t.specials[j]) })
	return t, nil
}

// protoFields calls f for each field of a protobuf message: v holds a varint or a fixed-width
// value, b a length-delimited one.
func protoFields(b []byte, f func(field int, v uint64, b []byte) error) error {
	for len(b) > 0 {
		key, n := binary.Uvarint(b)
		if n <= 0 {
			return errors.New("malformed protobuf")
		}
		b = b[n:]
		var v uint64
		var data []byte
		switch key & 7 {
		case 0:
			if v, n = binary.Uvarint(b); n <= 0 {
				return errors.New("malformed protobuf varint")
			}
			b = b[n:]
		case 1:
			if len(b) < 8 {
				return errors.New("malformed protobuf")
			}
			v, b = binary.LittleEndian.Uint64(b), b[8:]
		case 2:
			l, n := binary.Uvarint(b)
			if n <= 0 || l > uint64(len(b)-n) {
				return errors.New("malformed protobuf length")
			}
			data, b = b[n:n+int(l)], b[n+int(l):]
		case 5:
			if len(b) < 4 {
				return errors.New("malformed protobuf")
			}
			v, b = uint64(binary.LittleEndian.Uint32(b)), b[4:]
		default:
			return fmt.Errorf("protobuf wire type %d is not supported", key&7)
		}
		if err := f(int(key>>3), v, data); err != nil {
			return err
		}
	}
	return nil
}

// VocabSize is the number of pieces.
func (t *SentencePiece) VocabSize() int { return len(t.pieces) }

// ID returns the id of a piece ("<|startoftext|>", "▁the", "<0xE3>").
func (t *SentencePiece) ID(piece string) (int32, bool) {
	id, ok := t.pieceID[piece]
	return id, ok
}

// Encode turns text into token ids, without a BOS. Special pieces written in the text (such as
// "<|endoftext|>") become their ids.
func (t *SentencePiece) Encode(text string) []int32 {
	var ids []int32
	start := 0
	for i := 0; i < len(text); {
		if s := longestPrefix(t.specials, text[i:]); s != "" {
			ids = t.encodeOrdinary(ids, text[start:i])
			ids = append(ids, t.specialID[s])
			i += len(s)
			start = i
			continue
		}
		_, n := utf8.DecodeRuneInString(text[i:])
		i += n
	}
	return t.encodeOrdinary(ids, text[start:])
}

func longestPrefix(list []string, s string) string {
	for _, p := range list {
		if strings.HasPrefix(s, p) {
			return p
		}
	}
	return ""
}

// encodeOrdinary normalizes and merges one stretch of text with no special piece in it.
func (t *SentencePiece) encodeOrdinary(ids []int32, text string) []int32 {
	if text == "" {
		return ids
	}
	// The identity normalizer: spaces become ▁, a malformed byte becomes U+FFFD.
	var nb strings.Builder
	if t.dummyPrefix {
		nb.WriteString("▁")
	}
	for _, r := range text {
		if r == ' ' {
			r = '▁'
		}
		nb.WriteRune(r)
	}
	s := nb.String()

	// Characters, and user-defined pieces as single symbols that never merge.
	var syms []spSymbol
	for i := 0; i < len(s); {
		n, frozen := len(longestPrefix(t.user, s[i:])), true
		if n == 0 {
			_, n = utf8.DecodeRuneInString(s[i:])
			frozen = false
		}
		syms = append(syms, spSymbol{start: i, end: i + n, prev: len(syms) - 1, next: len(syms) + 1, frozen: frozen})
		i += n
	}
	syms[len(syms)-1].next = -1

	var q spQueue
	push := func(l, r int) {
		if l < 0 || r < 0 || syms[l].frozen || syms[r].frozen {
			return
		}
		if id, ok := t.vocab[s[syms[l].start:syms[r].end]]; ok {
			heap.Push(&q, spPair{left: l, right: r, score: t.scores[id], size: syms[r].end - syms[l].start})
		}
	}
	for i := 1; i < len(syms); i++ {
		push(i-1, i)
	}
	for q.Len() > 0 {
		p := heap.Pop(&q).(spPair)
		l, r := &syms[p.left], &syms[p.right]
		if l.end == l.start || r.end == r.start || r.end-l.start != p.size || l.next != p.right {
			continue // stale: one side merged since
		}
		l.end, l.next = r.end, r.next
		if r.next >= 0 {
			syms[r.next].prev = p.left
		}
		r.start, r.end = 0, 0
		push(l.prev, p.left)
		push(p.left, l.next)
	}
	for i := 0; i >= 0; i = syms[i].next {
		piece := s[syms[i].start:syms[i].end]
		if id, ok := t.vocab[piece]; ok {
			ids = append(ids, id)
			continue
		}
		for _, c := range []byte(piece) {
			ids = append(ids, t.bytes[c])
		}
	}
	return ids
}

type spSymbol struct {
	start, end int // in the normalized text; empty once merged into its left neighbor
	prev, next int
	frozen     bool
}

type spPair struct {
	left, right int
	score       float32
	size        int
}

// spQueue pops the highest score first and, on a tie, the leftmost pair, as sentencepiece's
// agenda does.
type spQueue []spPair

func (q spQueue) Len() int { return len(q) }
func (q spQueue) Less(i, j int) bool {
	return q[i].score > q[j].score || q[i].score == q[j].score && q[i].left < q[j].left
}
func (q spQueue) Swap(i, j int) { q[i], q[j] = q[j], q[i] }
func (q *spQueue) Push(x any)   { *q = append(*q, x.(spPair)) }
func (q *spQueue) Pop() any {
	old := *q
	x := old[len(old)-1]
	*q = old[:len(old)-1]
	return x
}

// Decode turns token ids back into text, so that Decode(Encode(s), false) == s (except that a ▁
// written in s comes back as a space). Special pieces are written as their content unless
// skipSpecial is set; each stretch between them loses the ▁ prepended in Encode. With
// skipSpecial the stretches join first, as Hugging Face's decode(skip_special_tokens=True) does.
func (t *SentencePiece) Decode(ids []int32, skipSpecial bool) string {
	var out []byte
	runEmpty := true // nothing written since the start or the last special piece
	var pending []byte
	flush := func() {
		for b := pending; len(b) > 0; {
			r, n := utf8.DecodeRune(b)
			out = utf8.AppendRune(out, r) // a malformed byte becomes U+FFFD
			b = b[n:]
			runEmpty = false
		}
		pending = pending[:0]
	}
	for _, id := range ids {
		if id < 0 || int(id) >= len(t.pieces) {
			continue
		}
		piece := t.pieces[id]
		if t.special[id] {
			if !skipSpecial {
				flush()
				out = append(out, piece...)
				runEmpty = true
			}
			continue
		}
		if c := t.byteOf[id]; c >= 0 {
			pending = append(pending, byte(c))
			continue
		}
		flush()
		if runEmpty && t.dummyPrefix {
			piece = strings.TrimPrefix(piece, "▁")
		}
		piece = strings.ReplaceAll(piece, "▁", " ")
		out = append(out, piece...)
		runEmpty = runEmpty && piece == ""
	}
	flush()
	return string(out)
}
