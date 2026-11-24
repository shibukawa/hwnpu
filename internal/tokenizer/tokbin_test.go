package tokenizer

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

// tinyUnigram is a Unigram tokenizer.json in T5's form, without a normalizer.
const tinyUnigram = `{
  "normalizer": null,
  "pre_tokenizer": {"type": "Sequence", "pretokenizers": [{"type": "WhitespaceSplit"},
    {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true}]},
  "post_processor": {"type": "TemplateProcessing", "single": [{"Sequence": {"id": "A"}}, {"SpecialToken": {"id": "</s>"}}],
    "special_tokens": {"</s>": {"ids": [1]}}},
  "model": {"type": "Unigram", "unk_id": 2,
    "vocab": [["<pad>", 0], ["</s>", 0], ["<unk>", 0], ["▁", -2], ["▁a", -1.5], ["b", -2.5], ["▁ab", -2], ["c", -3], ["▁abc", -5]]}
}`

var binaryTexts = []string{"", "ab abab c", "  ab\tbc\n", "abc abcab ab c a b", "<|end|> ab<eos>", "日本語 é 🤖", "<|plamo:tag|>aaaa  b", "\x00\xff"}

// A tokenizer's binary file encodes, decodes and finds ids as the tokenizer it was written from.
func TestBinaryRoundTrip(t *testing.T) {
	bpe, err := parse([]byte(tiny))
	if err != nil {
		t.Fatal(err)
	}
	fb, err := parse([]byte(tinyFallback()))
	if err != nil {
		t.Fatal(err)
	}
	for name, tk := range map[string]*Tokenizer{"byte-level": bpe, "fallback": fb, "plamo": tinyPlamo(t)} {
		path := filepath.Join(t.TempDir(), "tokenizer"+BinaryExt)
		if err := tk.WriteBinary(path); err != nil {
			t.Fatal(name, err)
		}
		b, err := loadBinary(path)
		if err != nil {
			t.Fatal(name, err)
		}
		if b.VocabSize() != tk.VocabSize() || !slices.Equal(b.Control(), tk.Control()) {
			t.Errorf("%s: vocabulary %d control %v, want %d %v", name, b.VocabSize(), b.Control(), tk.VocabSize(), tk.Control())
		}
		for _, s := range binaryTexts {
			want := tk.Encode(s)
			if got := b.Encode(s); !slices.Equal(got, want) {
				t.Errorf("%s: Encode(%q) = %v, want %v", name, s, got, want)
			}
			for _, skip := range []bool{false, true} {
				if got, want := b.Decode(want, skip), tk.Decode(want, skip); got != want {
					t.Errorf("%s: Decode(%v, %v) = %q, want %q", name, want, skip, got, want)
				}
			}
		}
		for _, tok := range []string{"ab", "<|end|>", "<eos>", "<0x41>", "a", "nothing"} {
			gid, gok := b.ID(tok)
			wid, wok := tk.ID(tok)
			if gid != wid || gok != wok {
				t.Errorf("%s: ID(%q) = %d %v, want %d %v", name, tok, gid, gok, wid, wok)
			}
		}
		// the same tokenizer makes the same bytes
		again := filepath.Join(filepath.Dir(path), "again"+BinaryExt)
		tk.WriteBinary(again)
		x, _ := os.ReadFile(path)
		y, _ := os.ReadFile(again)
		if string(x) != string(y) {
			t.Errorf("%s: two writes differ", name)
		}
	}

	u, err := parseUnigram([]byte(tinyUnigram))
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "t5"+BinaryExt)
	if err := u.WriteBinary(path); err != nil {
		t.Fatal(err)
	}
	ub, err := loadUnigramBinary(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range binaryTexts {
		if got, want := ub.Encode(s), u.Encode(s); !slices.Equal(got, want) {
			t.Errorf("unigram: Encode(%q) = %v, want %v", s, got, want)
		}
	}
	if _, err := loadBinary(path); err == nil {
		t.Error("a Unigram's file loaded as a Tokenizer")
	}
}

// Load takes the binary file beside the tokenizer file, even when that file is gone; a binary
// file of another version is passed by for the file, and is an error without it.
func TestLoadBeside(t *testing.T) {
	dir := t.TempDir()
	js := filepath.Join(dir, "tokenizer.json")
	os.WriteFile(js, []byte(tiny), 0o644)
	if err := WriteBinary(js, ""); err != nil {
		t.Fatal(err)
	}
	tok := filepath.Join(dir, "tokenizer"+BinaryExt)
	want, _ := parse([]byte(tiny))
	os.Remove(js)
	tk, err := Load(js)
	if err != nil || tk.file == nil || !slices.Equal(tk.Encode("abab c"), want.Encode("abab c")) {
		t.Fatalf("the binary file in place of the gone tokenizer.json: %v (mapped %v)", err, tk != nil && tk.file != nil)
	}

	raw, _ := os.ReadFile(tok)
	binary.LittleEndian.PutUint32(raw[8:], tokbinVersion+1)
	os.WriteFile(tok, raw, 0o644)
	if _, err := Load(js); err == nil {
		t.Error("a binary file of another version loaded")
	}
	os.WriteFile(js, []byte(tiny), 0o644)
	if tk, err := Load(js); err != nil || tk.file != nil {
		t.Errorf("another version's file beside the text: %v (mapped %v), want the text", err, tk != nil && tk.file != nil)
	}

	// Unigram's and PLaMo's loaders too
	t5 := filepath.Join(dir, "t5_tokenizer.json")
	os.WriteFile(t5, []byte(tinyUnigram), 0o644)
	if err := WriteBinary(t5, ""); err != nil {
		t.Fatal(err)
	}
	os.Remove(t5)
	if u, err := LoadUnigram(t5); err != nil || u.file == nil {
		t.Errorf("Unigram's binary file: %v", err)
	}
}

// A broken binary file is an error, not a crash.
func TestBinaryBroken(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tokenizer"+BinaryExt)
	tk, _ := parse([]byte(tinyFallback()))
	if err := tk.WriteBinary(path); err != nil {
		t.Fatal(err)
	}
	raw, _ := os.ReadFile(path)
	for _, n := range []int{0, 7, 15, 20, len(raw) / 2, len(raw) - 1} {
		os.WriteFile(path, raw[:n], 0o644)
		if _, err := loadBinary(path); err == nil {
			t.Errorf("a file cut at %d of %d bytes loaded", n, len(raw))
		}
	}
	// a header that points past the file, or tables out of shape
	n := int(binary.LittleEndian.Uint32(raw[12:]))
	var h tokbinHeader
	json.Unmarshal(raw[16:16+n], &h)
	for _, name := range []string{"tokens.off", "special", "merges"} {
		bad := h
		bad.Sections = map[string][2]int{}
		for k, v := range h.Sections {
			bad.Sections[k] = v
		}
		bad.Sections[name] = [2]int{h.Sections[name][0], h.Sections[name][1] - 4}
		head, _ := json.Marshal(bad)
		if len(head) > n {
			continue
		}
		b := append([]byte(nil), raw...)
		copy(b[16:], head)
		for i := 16 + len(head); i < 16+n; i++ {
			b[i] = ' '
		}
		os.WriteFile(path, b, 0o644)
		if _, err := loadBinary(path); err == nil {
			t.Errorf("a file with %s cut short loaded", name)
		}
	}
}

// A tokenizer read from its text is kept in the cache as a binary file, which the next load maps;
// a changed file is read again.
func TestCachedLoad(t *testing.T) {
	cacheDir := t.TempDir()
	old := CacheDir
	CacheDir = func() string { return cacheDir }
	defer func() { CacheDir = old }()
	js := filepath.Join(t.TempDir(), "tokenizer.json")
	os.WriteFile(js, []byte(tiny), 0o644)
	first, err := Load(js)
	if err != nil || first.file != nil {
		t.Fatalf("first: %v (mapped %v)", err, first != nil && first.file != nil)
	}
	entries, _ := filepath.Glob(filepath.Join(cacheDir, "*"+BinaryExt))
	if len(entries) != 1 {
		t.Fatalf("cache entries %v", entries)
	}
	second, err := Load(js)
	if err != nil || second.file == nil || !slices.Equal(second.Encode("abab bc"), first.Encode("abab bc")) {
		t.Fatalf("second: %v (mapped %v)", err, second != nil && second.file != nil)
	}
	future := time.Now().Add(time.Hour)
	os.Chtimes(js, future, future)
	if third, err := Load(js); err != nil || third.file != nil {
		t.Errorf("after a change: %v (mapped %v), want the text read again", err, third != nil && third.file != nil)
	}
	if entries, _ := filepath.Glob(filepath.Join(cacheDir, "*"+BinaryExt)); len(entries) != 2 {
		t.Errorf("cache entries after a change %v", entries)
	}
}

func TestBinaryPath(t *testing.T) {
	for in, want := range map[string]string{
		"a/tokenizer.json": "a/tokenizer.tok", "a/tokenizer.jsonl": "a/tokenizer.tok", "text/qwen_tokenizer.json": "text/qwen_tokenizer.tok",
		"spm.model": "spm.tok", "x/tokenizer.tok": "x/tokenizer.tok",
	} {
		if got := BinaryPath(in); got != want {
			t.Errorf("BinaryPath(%q) = %q, want %q", in, got, want)
		}
	}
	if !strings.HasSuffix(BinaryPath("tokenizer.json"), BinaryExt) {
		t.Error("ext")
	}
}
