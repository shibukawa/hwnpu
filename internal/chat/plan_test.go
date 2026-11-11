package chat

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
)

// TestExpect looks up the sizes of a repository's files as the Hub gives them (a redirect with
// X-Linked-Size for a large file, the file's length for a small one) and counts their downloads
// into the sum of all of them, one after another.
func TestExpect(t *testing.T) {
	files := map[string]string{"a.bin": strings.Repeat("a", 3000), "b.bin": strings.Repeat("b", 5000), "small.json": "{}", "here.json": "[]"}
	var heads atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		name, ok := strings.CutPrefix(r.URL.Path, "/owner/x/resolve/main/")
		body, known := files[name]
		if !ok || !known {
			http.NotFound(w, r)
			return
		}
		if r.Method == http.MethodHead {
			heads.Add(1)
			if len(body) > 100 { // as the Hub answers for a file kept outside git: a redirect to the CDN
				w.Header().Set("X-Linked-Size", strconv.Itoa(len(body)))
				w.Header().Set("Location", "https://cdn.example/"+name)
				w.WriteHeader(http.StatusFound)
				return
			}
			w.Header().Set("Content-Length", strconv.Itoa(len(body)))
			return
		}
		from := 0
		if rg := r.Header.Get("Range"); rg != "" {
			from, _ = strconv.Atoi(strings.TrimSuffix(strings.TrimPrefix(rg, "bytes="), "-"))
			w.Header().Set("Content-Length", strconv.Itoa(len(body)-from))
			w.WriteHeader(http.StatusPartialContent)
		}
		io.WriteString(w, body[from:])
	}))
	defer srv.Close()
	defer func(u string) { hubURL = u }(hubURL)
	hubURL = srv.URL + "/"
	t.Setenv("HF_TOKEN", "")
	t.Setenv("HF_TOKEN_PATH", filepath.Join(t.TempDir(), "none"))

	dir := t.TempDir()
	at := func(f string) string { return filepath.Join(dir, f) }
	os.WriteFile(at("here.json"), []byte("[]"), 0o644)                       // here: not looked up
	os.WriteFile(at("b.bin.part"), []byte(strings.Repeat("b", 1000)), 0o644) // cut off: goes on
	var want []File
	for _, f := range []string{"a.bin", "b.bin", "small.json", "here.json", "missing.bin"} {
		want = append(want, File{"owner/x", f, at(f)})
	}
	Expect(context.Background(), want)
	if heads.Load() != 3 {
		t.Errorf("%d HEAD requests of files the Hub has, want 3 (not the file that is here)", heads.Load())
	}
	if done, total := planNow(); done != 1000 || total != 8002 {
		t.Fatalf("before: %d of %d bytes, want 1000 of 8002", done, total)
	}
	steps := []struct {
		file        string
		done, total int64
	}{{"a.bin", 4000, 8002}, {"small.json", 4002, 8002}, {"b.bin", 0, 0}}
	for _, s := range steps {
		if err := fetch(context.Background(), "owner/x", s.file, at(s.file), nil); err != nil {
			t.Fatal(err)
		}
		if done, total := planNow(); done != s.done || total != s.total {
			t.Errorf("after %s: %d of %d bytes, want %d of %d", s.file, done, total, s.done, s.total)
		}
	}
	if b, _ := os.ReadFile(at("b.bin")); string(b) != files["b.bin"] {
		t.Errorf("b.bin went on wrong: %d bytes", len(b))
	}
}
