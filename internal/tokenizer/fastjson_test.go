package tokenizer

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// slowly runs f with the vocabulary and the merges read by encoding/json, as before splitModel.
func slowly(f func()) {
	fastModel = false
	defer func() { fastModel = true }()
	f()
}

// TestSplitModel checks that a tokenizer.json read with splitModel is the tokenizer encoding/json
// alone reads: of small documents with what a vocabulary and merges can hold (escapes, surrogate
// pairs, both forms of a merge, keys in another order, spaces), and that splitModel leaves to
// encoding/json what it does not read exactly (a key twice, an id with a fraction, bytes that are
// not UTF-8, a merge without a space).
func TestSplitModel(t *testing.T) {
	const head = `{"added_tokens":[{"id":7,"content":"<|im_end|>","special":true}],"normalizer":null,` +
		`"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"use_regex":true},"model":{"type":"BPE",`
	docs := map[string]string{
		"plain":       head + `"vocab":{"a":0,"b":1,"ab":2,"c":3,"abc":4,"Ġ":5,"Ġa":6},"merges":["a b","ab c","Ġ a"]}}`,
		"pairs":       head + `"merges":[["a","b"],["ab","c"],["Ġ","a"]],"vocab":{"a":0,"b":1,"ab":2,"c":3,"abc":4,"Ġ":5,"Ġa":6}}}`,
		"escapes":     head + `"vocab":{"\"":0,"\\":1,"\u00e9":2,"\ud83d\ude00":3,"\n":4,"a":5,"\"\\":6},"merges":["\" \\"]}}`,
		"spaces":      head + "\n \"vocab\" : { \"a\" : 0 ,\n\t\"b\":1, \"ab\" : 2 } ,\r\n \"merges\" : [ \"a b\" ,\n [ \"ab\" , \"a\" ] ] } }",
		"empty":       head + `"vocab":{},"merges":[]}}`,
		"more keys":   head + `"dropout":null,"vocab":{"a":0,"b":1,"ab":2},"unk_token":null,"merges":["a b"],"ignore_merges":true}}`,
		"twice":       head + `"vocab":{"a":0,"b":1,"ab":2},"merges":["a b"],"vocab":{"a":0,"b":1,"ab":3}}}`,
		"token twice": head + `"vocab":{"a":0,"b":1,"a":2,"ab":3},"merges":["a b"]}}`,
		"fraction":    head + `"vocab":{"a":0,"b":1.0,"ab":2},"merges":["a b"]}}`,
		"exponent":    head + `"vocab":{"a":0,"b":1e0,"ab":2},"merges":["a b"]}}`,
		"not utf-8":   head + "\"vocab\":{\"a\":0,\"\xff\":1,\"ab\":2},\"merges\":[\"a b\"]}}",
		"no space":    head + `"vocab":{"a":0,"b":1,"ab":2},"merges":["ab"]}}`,
		"long merge":  head + `"vocab":{"a":0,"b":1,"ab":2},"merges":[["a","b","c"]]}}`,
		"big id":      head + `"vocab":{"a":0,"b":4294967296,"ab":2},"merges":["a b"]}}`,
	}
	split := map[string]bool{"plain": true, "pairs": true, "escapes": true, "spaces": true, "empty": true, "more keys": true, "token twice": true}
	for name, doc := range docs {
		_, _, _, ok := splitModel([]byte(doc))
		if ok != split[name] {
			t.Errorf("%s: splitModel read it: %v, want %v", name, ok, split[name])
		}
		fast, fastErr := parse([]byte(doc))
		var slow *Tokenizer
		var slowErr error
		slowly(func() { slow, slowErr = parse([]byte(doc)) })
		if (fastErr == nil) != (slowErr == nil) || (fastErr != nil && fastErr.Error() != slowErr.Error()) {
			t.Errorf("%s: %v, without splitModel %v", name, fastErr, slowErr)
			continue
		}
		if fastErr == nil && !reflect.DeepEqual(fast, slow) {
			t.Errorf("%s: not the tokenizer encoding/json reads: %+v and %+v", name, fast, slow)
		}
	}
}

// TestSplitModelFiles compares the two readings of real tokenizers: HWNPU_TEST_TOKENIZERS is a list
// of tokenizer.json files (or directories searched for them), separated as PATH is.
func TestSplitModelFiles(t *testing.T) {
	list := os.Getenv("HWNPU_TEST_TOKENIZERS")
	if list == "" {
		t.Skip("HWNPU_TEST_TOKENIZERS")
	}
	var files []string
	for _, p := range filepath.SplitList(list) {
		filepath.WalkDir(p, func(path string, d os.DirEntry, err error) error {
			if err == nil && !d.IsDir() && d.Name() == "tokenizer.json" {
				files = append(files, path)
			}
			return nil
		})
	}
	texts := []string{"Hello, world! 1234 + 5678 = ?", "こんにちは、世界。東京タワーの高さは 333 m です。", "def f(x):\n\treturn x ** 2  # 🙂\n",
		strings.Repeat("a red fox sitting in a snowy forest, photograph ", 8)}
	n := 0
	for _, path := range files {
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		fast, fastErr := parse(raw)
		var slow *Tokenizer
		var slowErr error
		slowly(func() { slow, slowErr = parse(raw) })
		if (fastErr == nil) != (slowErr == nil) {
			t.Errorf("%s: %v, without splitModel %v", path, fastErr, slowErr)
			continue
		}
		if fastErr != nil {
			continue // not a tokenizer this package reads (Unigram, WordPiece)
		}
		_, _, _, ok := splitModel(raw)
		if !reflect.DeepEqual(fast, slow) {
			t.Errorf("%s: not the tokenizer encoding/json reads", path)
		}
		for _, s := range texts {
			if a, b := fast.Encode(s), slow.Encode(s); !reflect.DeepEqual(a, b) {
				t.Errorf("%s: %q encodes as %v, without splitModel %v", path, s, a, b)
			}
		}
		t.Logf("%s: %d tokens, read by splitModel: %v", path, fast.VocabSize(), ok)
		n++
	}
	if n == 0 {
		t.Fatalf("no tokenizer read of %d files", len(files))
	}
}
