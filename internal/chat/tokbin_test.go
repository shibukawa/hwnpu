package chat

import (
	"context"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/shibukawa/bonsaiai/internal/cache"
	shared "github.com/shibukawa/bonsaiai/internal/store"
	"github.com/shibukawa/bonsaiai/internal/tokenizer"
)

const tinyBPEJSON = `{
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
  "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": ["a b"]}
}`

// A checkpoint's tokenizer is made a binary file once, goes in the source written back to, and
// another machine fetches it there without the tokenizer's text.
func TestPrepareTokenizer(t *testing.T) {
	ctx := context.Background()
	share := t.TempDir()
	if _, err := withSources(t, []string{"file://" + filepath.ToSlash(share) + "?writeback=true"}); err != nil {
		t.Fatal(err)
	}
	fakeHub(t, map[string]string{"lab/tok/resolve/main/config.json": `{}`, "lab/tok/resolve/main/tokenizer.json": tinyBPEJSON})
	t.Setenv(cache.RootEnv, t.TempDir())
	p, err := PrepareTokenizer(ctx, "lab/tok", "tokenizer.json", nil)
	if err != nil {
		t.Fatal(err)
	}
	if !tokenizer.BinaryCurrent(tokenizer.BinaryPath(p)) {
		t.Fatalf("%s: no binary file beside it", p)
	}
	tk, err := tokenizer.Load(p)
	if err != nil || !slices.Equal(tk.Encode("ab"), []int32{2}) {
		t.Fatalf("load: %v", err)
	}
	shared.Flush(ctx)
	es, _ := shared.ListEntries(ctx, shared.Sources()[0])
	if len(es) != 1 || !strings.HasPrefix(es[0], "converted/tok/any/tokenizer/") {
		t.Fatalf("entries %v", es)
	}

	// another machine, whose hub has no tokenizer.json: the source's binary file
	fakeHub(t, map[string]string{"lab/tok/resolve/main/config.json": `{}`})
	t.Setenv(cache.RootEnv, t.TempDir())
	p2, err := PrepareTokenizer(ctx, "lab/tok", "tokenizer.json", nil)
	if err != nil {
		t.Fatal(err)
	}
	if tk, err := tokenizer.Load(p2); err != nil || !slices.Equal(tk.Encode("ab"), []int32{2}) {
		t.Fatalf("fetched: %v", err)
	}
	// and the next time, nothing is asked
	if _, err := PrepareTokenizer(ctx, "lab/tok", "tokenizer.json", nil); err != nil {
		t.Fatal(err)
	}
}
