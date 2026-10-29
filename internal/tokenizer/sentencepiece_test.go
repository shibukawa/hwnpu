package tokenizer

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func pbVarint(b []byte, field int, v uint64) []byte {
	return binary.AppendUvarint(binary.AppendUvarint(b, uint64(field<<3)), v)
}

func pbBytes(b []byte, field int, data string) []byte {
	b = binary.AppendUvarint(b, uint64(field<<3|2))
	return append(binary.AppendUvarint(b, uint64(len(data))), data...)
}

func pbFloat(b []byte, field int, x float32) []byte {
	return binary.LittleEndian.AppendUint32(binary.AppendUvarint(b, uint64(field<<3|5)), math.Float32bits(x))
}

type tinyPiece struct {
	piece string
	score float32
	typ   uint64
}

// tinyPieces: the special and user-defined pieces of PLaMo's kind, all 256 bytes, and a few
// normal pieces whose scores decide the merge order.
var tinyPieces = func() []tinyPiece {
	p := []tinyPiece{{"<unk>", 0, spUnknown}, {"<s>", 0, spControl}, {"</s>", 0, spControl}, {"▁▁", 0, spUserDefined}, {"\t", 0, spUserDefined}}
	for c := 0; c < 256; c++ {
		p = append(p, tinyPiece{fmt.Sprintf("<0x%02X>", c), 0, spByte})
	}
	for _, n := range []tinyPiece{{"▁", 0, 0}, {"a", 0, 0}, {"b", 0, 0}, {"c", 0, 0}, {"ab", -1, 0}, {"bc", -2, 0}, {"▁a", -3, 0}, {"▁ab", -4, 0}, {"aa", -5, 0}, {"ca", -0.7, 0}, {"\ta", -0.5, 0}} {
		n.typ = spNormal
		p = append(p, n)
	}
	return p
}()

var (
	tinyTrainer    = string(pbVarint(pbVarint(nil, 3, 2), 35, 1)) // BPE, byte fallback
	tinyNormalizer = string(pbVarint(pbVarint(pbBytes(nil, 1, "identity"), 3, 1), 4, 0))
)

func tinyModel(pieces []tinyPiece, trainer, normalizer string) []byte {
	var b []byte
	for _, p := range pieces {
		b = pbBytes(b, 1, string(pbVarint(pbFloat(pbBytes(nil, 1, p.piece), 2, p.score), 3, p.typ)))
	}
	return pbBytes(pbBytes(b, 2, trainer), 3, normalizer)
}

func TestSentencePiece(t *testing.T) {
	tk, err := parseSentencePiece(tinyModel(tinyPieces, tinyTrainer, tinyNormalizer))
	if err != nil {
		t.Fatal(err)
	}
	id := func(pieces ...string) []int32 {
		var ids []int32
		for _, p := range pieces {
			i, ok := tk.ID(p)
			if !ok {
				t.Fatalf("no piece %q", p)
			}
			ids = append(ids, i)
		}
		return ids
	}
	for _, c := range []struct {
		in   string
		want []int32
	}{
		{"", nil},
		{"abc", id("▁ab", "c")},             // ab (-1) before ▁a (-3), then ▁ab
		{"\tbca", id("▁", "\t", "b", "ca")}, // ca (-0.7) before the pair left of it, bc (-2)
		{"\taaa", id("▁", "\t", "aa", "a")}, // a tie goes to the leftmost pair; \t never merges, though \ta is a piece
		{"a b", id("▁a", "▁", "b")},
		{"  a", id("▁▁", "▁a")}, // the user-defined ▁▁ is taken whole
		{"é", id("▁", "<0xC3>", "<0xA9>")},
		{"a</s>b", id("▁a", "</s>", "▁", "b")}, // each stretch gets its own ▁
		{"<s>", id("<s>")},
	} {
		got := tk.Encode(c.in)
		if !slices.Equal(got, c.want) {
			t.Errorf("Encode(%q) = %v, want %v", c.in, got, c.want)
		}
		if back := tk.Decode(got, false); back != c.in {
			t.Errorf("Decode(%v) = %q, want %q", got, back, c.in)
		}
	}
	if got := tk.Decode(id("▁a", "</s>", "▁", "b"), true); got != "a b" {
		t.Errorf("Decode skipping specials = %q", got)
	}
	if got := tk.Decode(id("▁", "<0xC3>"), false); got != "�" {
		t.Errorf("Decode of a broken byte sequence = %q", got)
	}
}

func TestSentencePieceRejects(t *testing.T) {
	unused := append(slices.Clone(tinyPieces), tinyPiece{"zz", 0, spUnused})
	noByte := slices.DeleteFunc(slices.Clone(tinyPieces), func(p tinyPiece) bool { return p.piece == "<0x41>" })
	for name, raw := range map[string][]byte{
		"unigram":              tinyModel(tinyPieces, string(pbVarint(nil, 35, 1)), tinyNormalizer),
		"no byte fallback":     tinyModel(tinyPieces, string(pbVarint(nil, 3, 2)), tinyNormalizer),
		"nfkc rules":           tinyModel(tinyPieces, tinyTrainer, tinyNormalizer+string(pbBytes(nil, 2, "rules"))),
		"remove extra spaces":  tinyModel(tinyPieces, tinyTrainer, string(pbVarint(nil, 3, 1))),
		"no escaped spaces":    tinyModel(tinyPieces, tinyTrainer, tinyNormalizer+string(pbVarint(nil, 5, 0))),
		"whitespace as suffix": tinyModel(tinyPieces, tinyTrainer+string(pbVarint(nil, 24, 1)), tinyNormalizer),
		"unused piece":         tinyModel(unused, tinyTrainer, tinyNormalizer),
		"missing byte":         tinyModel(noByte, tinyTrainer, tinyNormalizer),
		"truncated":            tinyModel(tinyPieces, tinyTrainer, tinyNormalizer)[:100],
	} {
		if _, err := parseSentencePiece(raw); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

// TestPlamoEmbedding checks PLaMo-Embedding-1B's tokenizer against the ids Hugging Face gave
// (testdata/plamo-embedding-1b.json, from tools/ref/tokenizer_golden.py). It needs HWNPU_PLAMO_DIR
// with tokenizer.model.
func TestPlamoEmbedding(t *testing.T) {
	dir := os.Getenv("HWNPU_PLAMO_DIR")
	if dir == "" {
		t.Skip("HWNPU_PLAMO_DIR not set")
	}
	tk, err := LoadSentencePiece(filepath.Join(dir, "tokenizer.model"))
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile("testdata/plamo-embedding-1b.json")
	if err != nil {
		t.Fatal(err)
	}
	var golden struct {
		BOS   int32 `json:"bos"`
		Cases []struct {
			Text string  `json:"text"`
			IDs  []int32 `json:"ids"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &golden); err != nil {
		t.Fatal(err)
	}
	if n := tk.VocabSize(); n != 50112 {
		t.Errorf("VocabSize = %d", n)
	}
	if id, ok := tk.ID("<|startoftext|>"); !ok || id != golden.BOS {
		t.Errorf("<|startoftext|> = %d %v, want %d", id, ok, golden.BOS)
	}
	for _, c := range golden.Cases {
		got := tk.Encode(c.Text)
		if !slices.Equal(got, c.IDs) {
			t.Errorf("Encode(%q)\n = %v\nwant %v", c.Text, got, c.IDs)
		}
		if back := tk.Decode(got, false); back != c.Text && !strings.Contains(c.Text, "▁") {
			t.Errorf("round trip %q -> %q", c.Text, back)
		}
	}
}
