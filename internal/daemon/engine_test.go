package daemon

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A load of a name no definition has fails with the error of resolving it, before any device
// is opened, and the engine keeps serving.
func TestEngineLoadUnknown(t *testing.T) {
	t.Setenv("HWNPU_CACHE", t.TempDir())
	e := &Engine{Log: io.Discard}
	e.models, e.aliases = map[string]*held{}, map[string]string{}
	e.relay.base = io.Discard
	dir := shortDir(t)
	ln, err := ListenUnix(filepath.Join(dir, "e.sock"))
	if err != nil {
		t.Fatal(err)
	}
	srv := &http.Server{Handler: e.Handler()}
	go srv.Serve(ln)
	defer srv.Close()
	c := NewClient(filepath.Join(dir, "e.sock"))
	_, err = c.Load(context.Background(), LoadRequest{Model: "no-such-model-of-hwnpud"}, &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "no-such-model-of-hwnpud") {
		t.Fatalf("load: %v", err)
	}
	st, err := c.Status(context.Background())
	if err != nil || st.PID != os.Getpid() || len(st.Models) != 0 {
		t.Fatalf("status: %+v, %v", st, err)
	}
	// a completion without a model is refused
	resp, err := c.HTTP.Post(c.URL("/v1/chat/completions"), "application/json", strings.NewReader(`{"messages":[]}`))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("completion without a model: %s", resp.Status)
	}
}
