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
	"time"

	"github.com/shibukawa/hwnpu/internal/chat"
)

// A load of a name no definition has fails with the error of resolving it, before any device
// is opened, and the engine keeps serving.
func TestEngineLoadUnknown(t *testing.T) {
	t.Setenv("HWNPU_CACHE", t.TempDir())
	e := &Engine{Log: io.Discard}
	e.init()
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

// On AMD's NPU a load of another model closes the IRON model held, once the requests using it
// have ended; a request that comes meanwhile does not get it.
func TestEvictIRONWaitsForRequests(t *testing.T) {
	e := &Engine{Log: io.Discard}
	e.init()
	a := &held{e: &chat.Engine{}, model: chat.Model{Name: "a"}, iron: true}
	mac := &held{e: &chat.Engine{}, model: chat.Model{Name: "mac"}}
	e.models["a"], e.models["mac"] = a, mac
	e.aliases["a@0"], e.aliases["chat@0"], e.aliases["mac@0"] = "a", "a", "mac"
	if e.lookup("a@0") != a {
		t.Fatal("lookup")
	}
	done := make(chan struct{})
	go func() {
		e.evictIRON()
		close(done)
	}()
	select {
	case <-done:
		t.Fatal("closed with a request going on")
	case <-time.After(100 * time.Millisecond):
	}
	if e.lookup("a@0") != nil || e.lookup("chat@0") != nil {
		t.Fatal("a new request got the model being closed")
	}
	e.release(a)
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("not closed after its request ended")
	}
	if e.lookup("mac@0") != mac || len(e.models) != 1 || len(e.aliases) != 1 {
		t.Fatalf("models %v, aliases %v", e.models, e.aliases)
	}
}

// closeAll waits for in-flight requests to release their held models before closing them,
// preventing premature model teardown or mutex deadlock during graceful shutdown.
func TestCloseAllWaitsForRequests(t *testing.T) {
	e := &Engine{Log: io.Discard}
	e.init()
	m := &held{e: &chat.Engine{}, model: chat.Model{Name: "m"}}
	e.models["m"] = m
	e.aliases["m@0"] = "m"
	if e.lookup("m@0") != m {
		t.Fatal("lookup")
	}
	done := make(chan struct{})
	go func() {
		e.closeAll()
		close(done)
	}()
	select {
	case <-done:
		t.Fatal("closeAll returned while request still held model")
	case <-time.After(100 * time.Millisecond):
	}
	if e.lookup("m@0") != nil {
		t.Fatal("new request got model during closeAll")
	}
	e.release(m)
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("closeAll did not finish after request released model")
	}
}
