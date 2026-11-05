package tokenizer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// tinyPlamo is a vocabulary of PLaMo's tokenizer.jsonl form: the ids are the lines.
func tinyPlamo(t *testing.T) *Tokenizer {
	t.Helper()
	var lines []byte
	add := func(row ...any) {
		b, _ := json.Marshal(row)
		lines = append(append(lines, b...), '\n')
	}
	add("<|plamo:unk|>", 0.0, "UNKNOWN") // 0
	add("<|plamo:bos|>", 0.0, "CONTROL") // 1
	add("<|plamo:tag|>", 0.0, "CONTROL") // 2
	for b := 0; b < 256; b++ {           // 3..258
		add(string([]byte{'<', '0', 'x', "0123456789ABCDEF"[b>>4], "0123456789ABCDEF"[b&15], '>'}), 0.0, "BYTE")
	}
	add("a", -1.0, "NORMAL")   // 259
	add("b", -1.0, "NORMAL")   // 260
	add("ab", -1.5, "NORMAL")  // 261: cheaper than a + b (2.0)
	add("abc", -4.0, "NORMAL") // 262: dearer than ab + c? c is unknown: bytes
	add("aa", -2.0, "NORMAL")  // 263: ties a + a; the longer wins
	add("  ", -1.0, "NORMAL")  // 264
	add(" ", -1.0, "NORMAL")   // 265
	tk, err := parsePlamo(lines, 2, 4)
	if err != nil {
		t.Fatal(err)
	}
	return tk
}

func TestPlamoTiny(t *testing.T) {
	tk := tinyPlamo(t)
	for _, c := range []struct {
		in   string
		want []int32
	}{
		{"ab", []int32{261}},
		{"aa", []int32{263}},
		{"abc", []int32{262}},   // 4.0 against ab (1.5) + c's byte (1000.0)
		{"c", []int32{3 + 'c'}}, // a byte piece
		{"é", []int32{3 + 0xC3, 3 + 0xA9}},
		{"<|plamo:tag|>ab", []int32{2, 261}},
		{"a<|plamo:tag|><|plamo:bos|>b", []int32{259, 2, 1, 260}},
		{"aaaa", []int32{263, 263}},             // a run of four is cut out: aa aa within it
		{"baaaab", []int32{260, 263, 263, 260}}, // no ab across the cut
		{"a   b", []int32{259, 264, 265, 260}},  // three spaces: cut out, "  " then " "
	} {
		got := tk.Encode(c.in)
		if !slices.Equal(got, c.want) {
			t.Errorf("Encode(%q) = %v, want %v", c.in, got, c.want)
		}
		if back := tk.Decode(got, false); back != c.in {
			t.Errorf("Decode(%v) = %q, want %q", got, back, c.in)
		}
	}
	if got := tk.Decode([]int32{1, 259, 2}, true); got != "a" {
		t.Errorf("Decode skipping specials = %q", got)
	}
	if id, ok := tk.ID("<|plamo:tag|>"); !ok || id != 2 {
		t.Errorf("<|plamo:tag|> = %d %v", id, ok)
	}
}

// TestPlamo3 checks PLaMo 3's tokenizer against the ids Hugging Face gave
// (testdata/plamo-3-translate-2b.json, from tools/ref/tokenizer_golden.py). It needs
// HWNPU_PLAMO3_DIR with tokenizer.jsonl and tokenizer_config.json; HWNPU_PLAMO3_GOLDEN names
// another golden file (one with --random cases).
func TestPlamo3(t *testing.T) {
	dir := os.Getenv("HWNPU_PLAMO3_DIR")
	if dir == "" {
		t.Skip("HWNPU_PLAMO3_DIR not set")
	}
	tk, err := Load(filepath.Join(dir, "tokenizer.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	golden := "testdata/plamo-3-translate-2b.json"
	if g := os.Getenv("HWNPU_PLAMO3_GOLDEN"); g != "" {
		golden = g
	}
	raw, err := os.ReadFile(golden)
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		BOS   int32 `json:"bos"`
		Cases []struct {
			Text string  `json:"text"`
			IDs  []int32 `json:"ids"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	if n := tk.VocabSize(); n != 107520 {
		t.Errorf("VocabSize = %d", n)
	}
	if id, ok := tk.ID("<|plamo:bos|>"); !ok || id != g.BOS {
		t.Errorf("<|plamo:bos|> = %d %v, want %d", id, ok, g.BOS)
	}
	bad := 0
	for _, c := range g.Cases {
		got := tk.Encode(c.Text)
		if !slices.Equal(got, c.IDs) {
			if bad++; bad <= 20 {
				t.Errorf("Encode(%q)\n = %v\nwant %v", c.Text, got, c.IDs)
			}
		}
		if back := tk.Decode(got, false); back != c.Text {
			t.Errorf("round trip %q -> %q", c.Text, back)
		}
	}
	if bad > 0 {
		t.Errorf("%d of %d cases differ", bad, len(g.Cases))
	}
}
