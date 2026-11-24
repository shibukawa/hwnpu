package tokenizer

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"unsafe"

	"github.com/shibukawa/hwnpu/internal/mapped"
)

// A binary tokenizer file (BinaryExt) holds a loaded tokenizer's tables (flat.go) as they are in
// memory, so that loading one is mapping it: no parsing, no map to build, its pages read by the
// system as they are looked up. It is made from a tokenizer.json, PLaMo's tokenizer.jsonl or a
// Unigram (WriteBinary) when an export is built, and it stands in for the file beside it of the
// same name: Load, LoadPlamo and LoadUnigram take it first.
//
//	"HWNPUTOK"  u32 version  u32 n  n bytes of JSON: the kind, the small settings (meta) and
//	            where each section is
//	the sections, each from a multiple of 8: little-endian arrays of int32, uint32, int64,
//	float64, or bytes
//
// A file of another version is not read (the loader goes to the text file beside it, if any).

// BinaryExt ends the name of a binary tokenizer file: tokenizer.json's is tokenizer.tok.
const BinaryExt = ".tok"

const (
	tokbinMagic = "HWNPUTOK"
	// tokbinVersion changes with the layout, the hashes of the tables (strHash, pairHash) or
	// what the tables mean
	tokbinVersion = 1
)

// BinaryPath is the binary tokenizer file that stands for the tokenizer file at path:
// tokenizer.json → tokenizer.tok, tokenizer.jsonl → tokenizer.tok, spm.model → spm.tok.
func BinaryPath(path string) string {
	if strings.HasSuffix(path, BinaryExt) {
		return path
	}
	return strings.TrimSuffix(path, filepath.Ext(path)) + BinaryExt
}

type tokbinHeader struct {
	Kind     string            `json:"kind"`
	Meta     json.RawMessage   `json:"meta"`
	Sections map[string][2]int `json:"sections"` // name → offset, length in bytes
}

const (
	kindBPE     = "bpe"     // Tokenizer: byte-level BPE, or BPE with byte fallback
	kindPlamo   = "plamo"   // Tokenizer of PLaMo's pieces
	kindUnigram = "unigram" // Unigram
)

// tokbinWriter collects the sections of a file.
type tokbinWriter struct {
	names []string
	data  map[string][]byte
}

func (w *tokbinWriter) add(name string, b []byte) {
	if w.data == nil {
		w.data = map[string][]byte{}
	}
	w.names = append(w.names, name)
	w.data[name] = b
}

func (w *tokbinWriter) i32(name string, v []int32) {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], uint32(x))
	}
	w.add(name, b)
}

func (w *tokbinWriter) u32(name string, v []uint32) {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], x)
	}
	w.add(name, b)
}

func (w *tokbinWriter) i64(name string, v []int64) {
	b := make([]byte, 8*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint64(b[8*i:], uint64(x))
	}
	w.add(name, b)
}

func (w *tokbinWriter) f64(name string, v []float64) {
	b := make([]byte, 8*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint64(b[8*i:], math.Float64bits(x))
	}
	w.add(name, b)
}

func (w *tokbinWriter) strs(name string, t *strTable) {
	w.u32(name+".off", t.off)
	w.add(name+".data", t.data)
	if t.val != nil {
		w.i32(name+".val", t.val)
	}
	if t.slots != nil {
		w.i32(name+".slots", t.slots)
	}
}

// write writes the file to path, by way of a temporary file beside it renamed into place.
func (w *tokbinWriter) write(path, kind string, meta any) error {
	m, err := json.Marshal(meta)
	if err != nil {
		return err
	}
	h := tokbinHeader{Kind: kind, Meta: m, Sections: map[string][2]int{}}
	// the header's length depends on the offsets it holds: lay out until it stays the same
	var head []byte
	for size := 0; ; {
		off := align8(16 + size)
		for _, n := range w.names {
			h.Sections[n] = [2]int{off, len(w.data[n])}
			off = align8(off + len(w.data[n]))
		}
		if head, err = json.Marshal(h); err != nil {
			return err
		}
		if len(head) == size {
			break
		}
		size = len(head)
	}
	var buf bytes.Buffer
	buf.WriteString(tokbinMagic)
	binary.Write(&buf, binary.LittleEndian, uint32(tokbinVersion))
	binary.Write(&buf, binary.LittleEndian, uint32(len(head)))
	buf.Write(head)
	for _, n := range w.names {
		buf.Write(make([]byte, h.Sections[n][0]-buf.Len()))
		buf.Write(w.data[n])
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), "."+filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.Write(buf.Bytes()); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	os.Chmod(tmp.Name(), 0o644)
	return os.Rename(tmp.Name(), path)
}

func align8(n int) int { return (n + 7) &^ 7 }

// tokbinReader is an opened file's sections, in its mapping.
type tokbinReader struct {
	f    *mapped.File
	head tokbinHeader
	err  error
}

// errNotBinary is a file that is no binary tokenizer file of this version.
var errNotBinary = errors.New("not a binary tokenizer file of this version")

func openTokbin(path string) (*tokbinReader, error) {
	f, err := mapped.Open(path)
	if err != nil {
		return nil, err
	}
	b := f.Bytes()
	if len(b) < 16 || string(b[:8]) != tokbinMagic || binary.LittleEndian.Uint32(b[8:]) != tokbinVersion {
		f.Close()
		return nil, fmt.Errorf("tokenizer: %s: %w", path, errNotBinary)
	}
	n := int(binary.LittleEndian.Uint32(b[12:]))
	r := &tokbinReader{f: f}
	if 16+n > len(b) || json.Unmarshal(b[16:16+n], &r.head) != nil {
		f.Close()
		return nil, fmt.Errorf("tokenizer: %s: a bad header", path)
	}
	return r, nil
}

// section is the bytes of a section; a missing one or one past the file is an error kept in r.err
// (nil then).
func (r *tokbinReader) section(name string, optional bool) []byte {
	s, ok := r.head.Sections[name]
	b := r.f.Bytes()
	if !ok {
		if !optional && r.err == nil {
			r.err = fmt.Errorf("no section %s", name)
		}
		return nil
	}
	if s[0] < 0 || s[1] < 0 || s[0] > len(b) || s[1] > len(b)-s[0] {
		if r.err == nil {
			r.err = fmt.Errorf("section %s is past the end", name)
		}
		return nil
	}
	return b[s[0] : s[0]+s[1] : s[0]+s[1]]
}

// nativeLE: the arrays are the file's bytes as they are where the machine is little-endian.
var nativeLE = binary.NativeEndian.Uint16([]byte{1, 0}) == 1

// view is the section's values of type T (4 or 8 bytes) in place when the machine reads them as
// the file holds them, else decoded with get.
func view[T int32 | uint32 | int64 | float64](r *tokbinReader, name string, optional bool, get func([]byte) T) []T {
	b := r.section(name, optional)
	if b == nil {
		return nil
	}
	var z T
	size := int(unsafe.Sizeof(z))
	if len(b)%size != 0 {
		if r.err == nil {
			r.err = fmt.Errorf("section %s is not of %d-byte values", name, size)
		}
		return nil
	}
	if len(b) == 0 {
		return []T{}
	}
	if nativeLE && uintptr(unsafe.Pointer(&b[0]))%uintptr(size) == 0 {
		return unsafe.Slice((*T)(unsafe.Pointer(&b[0])), len(b)/size)
	}
	out := make([]T, len(b)/size)
	for i := range out {
		out[i] = get(b[size*i:])
	}
	return out
}

func (r *tokbinReader) i32(name string, optional bool) []int32 {
	return view(r, name, optional, func(b []byte) int32 { return int32(binary.LittleEndian.Uint32(b)) })
}

func (r *tokbinReader) u32(name string) []uint32 {
	return view(r, name, false, binary.LittleEndian.Uint32)
}

func (r *tokbinReader) i64(name string) []int64 {
	return view(r, name, false, func(b []byte) int64 { return int64(binary.LittleEndian.Uint64(b)) })
}

func (r *tokbinReader) f64(name string) []float64 {
	return view(r, name, false, func(b []byte) float64 { return math.Float64frombits(binary.LittleEndian.Uint64(b)) })
}

// strs is a string table, checked so that a bad file cannot make str or find read past it.
func (r *tokbinReader) strs(name string) *strTable {
	t := &strTable{off: r.u32(name + ".off"), data: r.section(name+".data", false), val: r.i32(name+".val", true), slots: r.i32(name+".slots", true)}
	if r.err != nil {
		return nil
	}
	n := len(t.off) - 1
	ok := n >= 0 && t.off[0] == 0 && int(t.off[n]) == len(t.data) && (t.val == nil || len(t.val) == n)
	for i := 0; ok && i < n; i++ {
		ok = t.off[i] <= t.off[i+1]
	}
	if s := len(t.slots); t.slots != nil && (s == 0 || s&(s-1) != 0) {
		ok = false
	}
	for _, k := range t.slots {
		if k < 0 || int(k) > n {
			ok = false
			break
		}
	}
	if !ok {
		r.err = fmt.Errorf("the table %s is not well formed", name)
		return nil
	}
	return t
}

func (r *tokbinReader) pairs(name string) *pairTable {
	p := &pairTable{slots: r.i32(name, false)}
	if s := len(p.slots) / 4; r.err == nil && (len(p.slots)%4 != 0 || s == 0 || s&(s-1) != 0) {
		r.err = fmt.Errorf("the pairs %s are not well formed", name)
	}
	return p
}

// bpeMeta is a Tokenizer's small settings.
type bpeMeta struct {
	Style        style            `json:"style"`
	NFC          bool             `json:"nfc,omitempty"`
	IgnoreMerges bool             `json:"ignore_merges,omitempty"`
	Added        []string         `json:"added"`
	AddedID      map[string]int32 `json:"added_id"`
	Control      []string         `json:"control"`
	ByteSym      *[256]int32      `json:"byte_sym,omitempty"` // byte-level
	ByteID       *[256]int32      `json:"byte_id,omitempty"`  // byte fallback
	ByteTok      *[256]int32      `json:"byte_tok,omitempty"` // byte fallback
}

type plamoMeta struct {
	Control []string   `json:"control"`
	Bytes   [256]int32 `json:"bytes"`
	Repeats int        `json:"repeats"`
	Spaces  int        `json:"spaces"`
}

type unigramMeta struct {
	MaxLen  int     `json:"max_len"`
	Unk     int32   `json:"unk"`
	UnkCost float64 `json:"unk_cost"`
	After   []int32 `json:"after"`
	Space   string  `json:"space"`
	Bytes   []int32 `json:"bytes"` // nil: no byte fallback
	Norm    bool    `json:"norm"`
}

// WriteBinary writes the tokenizer to path (BinaryPath of its file).
func (t *Tokenizer) WriteBinary(path string) error {
	defer runtime.KeepAlive(t)
	var w tokbinWriter
	w.strs("tokens", t.tokens)
	if t.vocab.shares(t.tokens) {
		w.i32("vocab.slots", t.vocab.slots) // an index of the tokens
	} else {
		w.strs("vocab", t.vocab)
	}
	w.add("special", t.special)
	if t.plamo != nil {
		p := t.plamo
		w.strs("pieces", p.pieces)
		w.i32("ids", p.ids)
		w.i64("scores", p.scores)
		w.strs("prefixes", p.prefixes)
		return w.write(path, kindPlamo, plamoMeta{Control: t.control, Bytes: p.bytes, Repeats: p.repeats, Spaces: p.spaces})
	}
	m := bpeMeta{Style: t.style, NFC: t.nfc, IgnoreMerges: t.ignoreMerges, Added: t.added, AddedID: t.addedID, Control: t.control}
	if fb := t.fallback; fb != nil {
		m.ByteID, m.ByteTok = &fb.byteID, &fb.byteTok
		w.i32("merges", fb.merges.slots)
	} else {
		m.ByteSym = &t.byteSym
		w.i32("merges", t.merges.slots)
	}
	return w.write(path, kindBPE, m)
}

// WriteBinary writes the Unigram to path (BinaryPath of its file).
func (u *Unigram) WriteBinary(path string) error {
	defer runtime.KeepAlive(u)
	var w tokbinWriter
	w.strs("pieces", u.pieces)
	w.f64("scores", u.scores)
	m := unigramMeta{MaxLen: u.maxLen, Unk: u.unk, UnkCost: u.unkCost, After: u.after, Space: u.space, Bytes: u.bytes}
	if u.norm != nil {
		m.Norm = true
		w.u32("trie", u.norm.trie)
		w.add("repl", u.norm.repl)
	}
	return w.write(path, kindUnigram, m)
}

// loadBinary opens a binary file of a Tokenizer (bpe or plamo).
func loadBinary(path string) (*Tokenizer, error) {
	r, err := openTokbin(path)
	if err != nil {
		return nil, err
	}
	t := &Tokenizer{file: r.f, tokens: r.strs("tokens"), special: r.section("special", false)}
	if _, own := r.head.Sections["vocab.off"]; own {
		t.vocab = r.strs("vocab")
	} else if t.tokens != nil {
		t.vocab = &strTable{off: t.tokens.off, data: t.tokens.data, slots: r.i32("vocab.slots", false)}
		if s := len(t.vocab.slots); r.err == nil && (s == 0 || s&(s-1) != 0) {
			r.err = errors.New("the vocabulary's index is not well formed")
		}
		for _, k := range t.vocab.slots {
			if r.err == nil && (k < 0 || int(k) > t.tokens.n()) {
				r.err = errors.New("the vocabulary's index is past the tokens")
			}
		}
	}
	fail := func(err error) (*Tokenizer, error) {
		r.f.Close()
		return nil, fmt.Errorf("tokenizer: %s: %w", path, err)
	}
	if r.err == nil && len(t.special) != t.tokens.n() {
		r.err = errors.New("the special tokens are not one a token")
	}
	n := int32(0)
	if r.err == nil {
		n = int32(t.tokens.n())
	}
	inRange := func(ids []int32, limit int32) bool {
		for _, id := range ids {
			if id < 0 || id >= limit {
				return false
			}
		}
		return true
	}
	switch r.head.Kind {
	case kindBPE:
		var m bpeMeta
		if err := json.Unmarshal(r.head.Meta, &m); err != nil {
			return fail(err)
		}
		if int(m.Style) < 0 || m.Style > o200k {
			return fail(fmt.Errorf("pre-tokenizer style %d", m.Style))
		}
		t.style, t.nfc, t.ignoreMerges, t.added, t.addedID, t.control = m.Style, m.NFC, m.IgnoreMerges, m.Added, m.AddedID, m.Control
		if t.addedID == nil {
			t.addedID = map[string]int32{}
		}
		t.cache = map[string][]int32{}
		merges := r.pairs("merges")
		switch {
		case m.ByteID != nil && m.ByteTok != nil:
			if r.err == nil && !inRange(m.ByteID[:], n) {
				r.err = errors.New("a byte's id is past the tokens")
			}
			t.fallback = newFallback(merges, *m.ByteID, *m.ByteTok)
		case m.ByteSym != nil:
			t.merges, t.byteSym = merges, *m.ByteSym
		default:
			return fail(errors.New("neither byte-level nor byte fallback"))
		}
	case kindPlamo:
		var m plamoMeta
		if err := json.Unmarshal(r.head.Meta, &m); err != nil {
			return fail(err)
		}
		p := &plamo{pieces: r.strs("pieces"), ids: r.i32("ids", false), scores: r.i64("scores"), prefixes: r.strs("prefixes"), bytes: m.Bytes, repeats: m.Repeats, spaces: m.Spaces}
		if r.err == nil && (len(p.ids) != p.pieces.n() || len(p.scores) != p.pieces.n() || !inRange(p.ids, n) || !inRange(p.bytes[:], n) ||
			(p.pieces.val != nil && !inRange(p.pieces.val, int32(len(p.ids))))) {
			r.err = errors.New("the pieces' ids or scores are not one a piece")
		}
		t.plamo, t.control, t.addedID = p, m.Control, map[string]int32{}
	default:
		return fail(fmt.Errorf("a %s file is no Tokenizer", r.head.Kind))
	}
	if r.err != nil {
		return fail(r.err)
	}
	// unmapped once t is gone: the exported methods keep t alive while they read the tables, and
	// nothing they give is in the mapping (Decode's strings are copies)
	runtime.AddCleanup(t, func(f *mapped.File) { f.Close() }, r.f)
	return t, nil
}

// loadUnigramBinary opens a binary file of a Unigram.
func loadUnigramBinary(path string) (*Unigram, error) {
	r, err := openTokbin(path)
	if err != nil {
		return nil, err
	}
	fail := func(err error) (*Unigram, error) {
		r.f.Close()
		return nil, fmt.Errorf("tokenizer: %s: %w", path, err)
	}
	if r.head.Kind != kindUnigram {
		return fail(fmt.Errorf("a %s file is no Unigram", r.head.Kind))
	}
	var m unigramMeta
	if err := json.Unmarshal(r.head.Meta, &m); err != nil {
		return fail(err)
	}
	u := &Unigram{file: r.f, pieces: r.strs("pieces"), scores: r.f64("scores"), maxLen: m.MaxLen, unk: m.Unk, unkCost: m.UnkCost, after: m.After, space: m.Space, bytes: m.Bytes}
	if m.Norm {
		u.norm = &charsmap{trie: r.u32("trie"), repl: r.section("repl", false)}
	}
	if r.err != nil {
		return fail(r.err)
	}
	if u.pieces.val != nil {
		for _, id := range u.pieces.val {
			if id < 0 || int(id) >= len(u.scores) {
				return fail(errors.New("a piece's id is past the scores"))
			}
		}
	}
	runtime.AddCleanup(u, func(f *mapped.File) { f.Close() }, r.f)
	return u, nil
}

// WriteBinary reads the tokenizer file at src (a tokenizer.json of BPE or Unigram, PLaMo's
// tokenizer.jsonl, or a SentencePiece Unigram's spm.model) and writes it as a binary file to dst
// (BinaryPath(src) when dst is ""). It checks that the binary file encodes as the source does.
func WriteBinary(src, dst string) error {
	if dst == "" {
		dst = BinaryPath(src)
	}
	if dst == src {
		return fmt.Errorf("tokenizer: %s is a binary file already", src)
	}
	kind, err := kindOf(src)
	if err != nil {
		return err
	}
	switch kind {
	case kindUnigram:
		var u *Unigram
		if strings.HasSuffix(src, ".model") {
			u, err = parseSentencePieceUnigramFile(src)
		} else {
			u, err = loadUnigramText(src)
		}
		if err != nil {
			return err
		}
		if err := u.WriteBinary(dst); err != nil {
			return err
		}
		b, err := loadUnigramBinary(dst)
		if err != nil {
			return err
		}
		return sameEncoding(src, func(s string) []int32 { return u.Encode(s) }, func(s string) []int32 { return b.Encode(s) })
	default:
		t, err := loadText(src)
		if err != nil {
			return err
		}
		if err := t.WriteBinary(dst); err != nil {
			return err
		}
		b, err := loadBinary(dst)
		if err != nil {
			return err
		}
		return sameEncoding(src, t.Encode, b.Encode)
	}
}

// sameEncoding checks a binary file against its source on a few texts.
func sameEncoding(src string, a, b func(string) []int32) error {
	for _, s := range checkTexts {
		x, y := a(s), b(s)
		if len(x) != len(y) {
			return fmt.Errorf("tokenizer: %s: the binary file encodes %q otherwise", src, s)
		}
		for i := range x {
			if x[i] != y[i] {
				return fmt.Errorf("tokenizer: %s: the binary file encodes %q otherwise", src, s)
			}
		}
	}
	return nil
}

var checkTexts = []string{
	"Hello, world! The quick brown fox jumps over the lazy dog 1234567.",
	"日本語の文章と、English が混ざった text です。　全角スペースも。",
	"  spaces\tand\nnew lines\r\n  ",
	"emoji 🤖🚀, accents é é ü, and 😀 a rare one: 𠮷",
	"<|im_start|>user\nHi<|im_end|><start_of_turn>model<|plamo:op|>",
}

// kindOf tells what the tokenizer file at path is: PLaMo's tokenizer.jsonl, a SentencePiece
// model (Unigram), or a tokenizer.json of BPE or Unigram (its model's type).
func kindOf(path string) (string, error) {
	switch {
	case strings.HasSuffix(path, ".jsonl"):
		return kindPlamo, nil
	case strings.HasSuffix(path, ".model"):
		return kindUnigram, nil
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	var f struct {
		Model struct {
			Type string `json:"type"`
		} `json:"model"`
	}
	if small, _, _, ok := splitModel(raw); ok {
		raw = small
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		return "", fmt.Errorf("tokenizer: %s: %w", path, err)
	}
	if f.Model.Type == "Unigram" {
		return kindUnigram, nil
	}
	return kindBPE, nil
}
