package chat

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// fetch downloads a missing file, goes on from a download cut off, and leaves a file that is
// there alone.
func TestFetch(t *testing.T) {
	body := strings.Repeat("0123456789", 1000)
	var ranges []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/org/m/resolve/main/f.bin" {
			http.NotFound(w, r)
			return
		}
		ranges = append(ranges, r.Header.Get("Range"))
		http.ServeContent(w, r, "f.bin", time.Time{}, strings.NewReader(body))
	}))
	defer srv.Close()
	defer func(u string) { hubURL = u }(hubURL)
	hubURL = srv.URL + "/"

	dir := t.TempDir()
	path := filepath.Join(dir, "f.bin")
	if err := os.WriteFile(path+".part", []byte(body[:4321]), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := fetch(context.Background(), "org/m", "f.bin", path, io.Discard); err != nil {
		t.Fatal(err)
	}
	got, _ := os.ReadFile(path)
	if string(got) != body {
		t.Errorf("got %d bytes, want the %d of the file", len(got), len(body))
	}
	if _, err := os.Stat(path + ".part"); !os.IsNotExist(err) {
		t.Error("the .part file is left")
	}
	if err := fetch(context.Background(), "org/m", "f.bin", path, io.Discard); err != nil {
		t.Fatal(err)
	}
	if len(ranges) != 1 || ranges[0] != "bytes=4321-" {
		t.Errorf("requests with ranges %q, want one for bytes=4321-", ranges)
	}
	if err := fetch(context.Background(), "org/m", "none.bin", filepath.Join(dir, "none.bin"), io.Discard); err == nil {
		t.Error("no error for a file the repository does not have")
	}
}
