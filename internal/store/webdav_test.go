package store

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/internal/cache"
	"golang.org/x/net/webdav"
)

// davServer is a WebDAV server (golang.org/x/net/webdav) of a folder under /dav/, signed in to
// as user u, password p, that refuses PROPFIND of depth infinity as servers often do.
type davServer struct {
	*httptest.Server
	root     string
	requests atomic.Int64
	ranges   atomic.Int64
	conns    atomic.Int64 // the connections made to it
}

func newDAV(t *testing.T) *davServer {
	t.Helper()
	d := &davServer{root: t.TempDir()}
	h := &webdav.Handler{Prefix: "/dav", FileSystem: webdav.Dir(d.root), LockSystem: webdav.NewMemLS()}
	d.Server = httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		d.requests.Add(1)
		if u, p, ok := r.BasicAuth(); !ok || u != "u" || p != "p" {
			w.Header().Set("WWW-Authenticate", `Basic realm="dav"`)
			http.Error(w, "sign in", http.StatusUnauthorized)
			return
		}
		if r.Method == "PROPFIND" && r.Header.Get("Depth") != "0" && r.Header.Get("Depth") != "1" {
			http.Error(w, "no depth infinity", http.StatusForbidden)
			return
		}
		if r.Header.Get("Range") != "" {
			d.ranges.Add(1)
		}
		h.ServeHTTP(w, r)
	}))
	d.Config.ConnState = func(_ net.Conn, s http.ConnState) {
		if s == http.StateNew {
			d.conns.Add(1)
		}
	}
	d.Start()
	t.Cleanup(d.Close)
	return d
}

func openDAV(t *testing.T, d *davServer) *WebDAV {
	t.Helper()
	t.Setenv("NETRC", filepath.Join(t.TempDir(), "none"))
	t.Setenv(WebDAVUserEnv, "u")
	t.Setenv(WebDAVPasswordEnv, "p")
	st, err := OpenWebDAV(WebDAVConfig{Name: "nas", URL: d.URL + "/dav/hwnpu"})
	if err != nil {
		t.Fatal(err)
	}
	return st
}

// TestWebDAV puts, reads, lists and deletes files as TestStores does, and goes on with a
// download cut off.
func TestWebDAV(t *testing.T) {
	ctx := context.Background()
	d := newDAV(t)
	st := openDAV(t, d)
	big := make([]byte, 3<<20+17)
	rand.Read(big)
	src := filepath.Join(t.TempDir(), "big")
	os.WriteFile(src, big, 0o644)
	if err := st.Put(ctx, "blobs/sha256/ab/big", src); err != nil {
		t.Fatal(err)
	}
	if err := st.Write(ctx, "entries/amd/k@1.json", []byte(`{}`)); err != nil {
		t.Fatal(err)
	}
	if err := st.Write(ctx, "top", nil); err != nil {
		t.Fatal(err)
	}
	if b, err := os.ReadFile(filepath.Join(d.root, "hwnpu", "blobs", "sha256", "ab", "big")); err != nil || !bytes.Equal(b, big) {
		t.Fatalf("on the server: %d bytes, %v", len(b), err)
	}
	if n, ok, err := st.Has(ctx, "blobs/sha256/ab/big"); err != nil || !ok || n != int64(len(big)) {
		t.Errorf("Has: %d %v %v", n, ok, err)
	}
	if n, ok, err := st.Has(ctx, "top"); err != nil || !ok || n != 0 {
		t.Errorf("Has of an empty file: %d %v %v", n, ok, err)
	}
	for _, k := range []string{"blobs/none", "none/at/all", "blobs/sha256"} { // a folder is not a file
		if _, ok, err := st.Has(ctx, k); err != nil || ok {
			t.Errorf("Has of %s: %v %v", k, ok, err)
		}
	}
	dst := filepath.Join(t.TempDir(), "x", "big")
	os.MkdirAll(filepath.Dir(dst), 0o755)
	os.WriteFile(dst+".part", big[:1<<20], 0o644)
	if n, err := st.Get(ctx, "blobs/sha256/ab/big", dst, nil); err != nil || n != int64(len(big)) {
		t.Fatalf("Get: %d %v", n, err)
	}
	if b, _ := os.ReadFile(dst); !bytes.Equal(b, big) {
		t.Fatalf("got %d bytes", len(b))
	}
	if d.ranges.Load() == 0 {
		t.Error("the cut-off download started over")
	}
	if b, err := st.Read(ctx, "entries/amd/k@1.json"); err != nil || string(b) != "{}" {
		t.Errorf("Read: %q %v", b, err)
	}
	if _, err := st.Get(ctx, "none", dst+"2", nil); !isNotFound(err) {
		t.Errorf("Get of none: %v", err)
	}
	if _, err := st.Read(ctx, "a/none"); !isNotFound(err) {
		t.Errorf("Read of none: %v", err)
	}
	os.WriteFile(filepath.Join(d.root, "hwnpu", "blobs", ".put-left"), []byte("x"), 0o644)
	keys, err := st.List(ctx, "")
	if err != nil || strings.Join(keys, " ") != "blobs/sha256/ab/big entries/amd/k@1.json top" {
		t.Errorf("List: %v %v", keys, err)
	}
	if keys, err := st.List(ctx, "entries/"); err != nil || strings.Join(keys, " ") != "entries/amd/k@1.json" {
		t.Errorf("List entries/: %v %v", keys, err)
	}
	if keys, err := st.List(ctx, "nothing/"); err != nil || len(keys) != 0 {
		t.Errorf("List of nothing: %v %v", keys, err)
	}
	if err := st.Delete(ctx, "top"); err != nil {
		t.Fatal(err)
	}
	if err := st.Delete(ctx, "top"); err != nil {
		t.Errorf("Delete of none: %v", err)
	}
	if _, ok, _ := st.Has(ctx, "top"); ok {
		t.Error("deleted, still there")
	}
	if _, err := st.Read(ctx, "../x"); err == nil {
		t.Error("a key out of the store was taken")
	}
}

// TestWebDAVSource puts an entry and a repository's file in a WebDAV source and reads them
// back, also as a server read only.
func TestWebDAVSource(t *testing.T) {
	t.Setenv(cache.RootEnv, t.TempDir())
	ctx := context.Background()
	d := newDAV(t)
	l := Source{Store: openDAV(t, d), Writeback: true}
	dir := filepath.Join(t.TempDir(), "x.amd-npu")
	write(t, filepath.Join(dir, "a.bin"), "aaaa")
	write(t, filepath.Join(dir, "sub", "b.bin"), "bb")
	rel := EntryPath("x", "amd-xdna2", "x.amd-npu", "v1")
	if _, err := Describe(dir, Entry{Kind: "amd", Key: "x@1", Scope: ScopePortable, Path: rel}); err != nil {
		t.Fatal(err)
	}
	if err := PutEntry(ctx, l, dir); err != nil {
		t.Fatal(err)
	}
	if es, err := ListEntries(ctx, l); err != nil || len(es) != 1 || es[0] != rel {
		t.Errorf("ListEntries: %v %v", es, err)
	}
	all := func(Entry) bool { return true }
	dst := filepath.Join(t.TempDir(), "y")
	if i, err := FetchEntry(ctx, []Source{l}, rel, all, dst, nil); err != nil || i != 0 {
		t.Fatalf("FetchEntry: %d %v", i, err)
	}
	if b, _ := os.ReadFile(filepath.Join(dst, "sub", "b.bin")); string(b) != "bb" {
		t.Errorf("got %q", b)
	}
	c := strings.Repeat("c", 40)
	if err := PutOriginal(ctx, l, "lab/m", c, "config.json", filepath.Join(dir, "a.bin"), ""); err != nil {
		t.Fatal(err)
	}
	// the same server read only, as plain HTTP
	ro, err := ParseSource("ro", strings.Replace(d.URL, "//", "//u:p@", 1)+"/dav/hwnpu")
	if err != nil {
		t.Fatal(err)
	}
	clearLocks()
	if i, lk, err := FetchOriginal(ctx, []Source{ro}, "lab/m", c, "config.json", "", filepath.Join(t.TempDir(), "c.json"), nil); err != nil || i != 0 || lk.Commit != c {
		t.Errorf("read only: %d %+v %v", i, lk, err)
	}
	if i, err := FetchEntry(ctx, []Source{ro}, rel, all, filepath.Join(t.TempDir(), "z"), nil); err != nil || i != 0 {
		t.Errorf("read only entry: %v", err)
	}
	if err := ro.Store.Write(ctx, "x", nil); err == nil || !strings.Contains(err.Error(), "writeback") {
		t.Errorf("a write to a source read only: %v", err)
	}
}

// TestWebDAVGC removes an entry of a WebDAV source with its folders (a folder of WebDAV stays
// when its files are deleted), up to the model's, and leaves the others.
func TestWebDAVGC(t *testing.T) {
	t.Setenv(cache.RootEnv, t.TempDir())
	ctx := context.Background()
	d := newDAV(t)
	l := Source{Store: openDAV(t, d), Writeback: true}
	put := func(model string) {
		dir := filepath.Join(t.TempDir(), model)
		write(t, filepath.Join(dir, "sub", "b.bin"), model)
		rel := EntryPath(model, "amd-xdna2", model+".amd-npu", "v1")
		if _, err := Describe(dir, Entry{Kind: "amd", Key: model + "@1", Model: model, Scope: ScopePortable, Path: rel}); err != nil {
			t.Fatal(err)
		}
		if err := PutEntry(ctx, l, dir); err != nil {
			t.Fatal(err)
		}
	}
	put("x")
	put("y")
	g, err := PlanGC(ctx, l, func(e Entry) bool { return e.Model == "x" })
	if err != nil || len(g.Entries) != 1 {
		t.Fatalf("plan: %+v %v", g, err)
	}
	if err := g.Apply(ctx, l); err != nil {
		t.Fatal(err)
	}
	conv := filepath.Join(d.root, "hwnpu", "converted")
	if _, err := os.Stat(filepath.Join(conv, "x")); err == nil {
		t.Error("x's folders are left")
	}
	if _, err := os.Stat(filepath.Join(conv, "y", "amd-xdna2", "y.amd-npu", "v1", EntryFile)); err != nil {
		t.Errorf("y: %v", err)
	}
}

// TestWebDAVSignIn takes the login from the netrc and says what to do without one.
func TestWebDAVSignIn(t *testing.T) {
	ctx := context.Background()
	d := newDAV(t)
	host := strings.TrimPrefix(d.URL, "http://")
	host = host[:strings.LastIndex(host, ":")]
	netrc := filepath.Join(t.TempDir(), "netrc")
	os.WriteFile(netrc, []byte("machine other login x password y\nmachine "+host+"\n  login u\n  password p\ndefault login z password w\n"), 0o600)
	t.Setenv("NETRC", netrc)
	t.Setenv(WebDAVUserEnv, "")
	t.Setenv(WebDAVPasswordEnv, "")
	st, err := OpenWebDAV(WebDAVConfig{Name: "nas", URL: d.URL + "/dav/"})
	if err != nil {
		t.Fatal(err)
	}
	if err := st.Write(ctx, "f", []byte("x")); err != nil {
		t.Fatal(err)
	}
	st, _ = OpenWebDAV(WebDAVConfig{Name: "nas", URL: strings.Replace(d.URL, "//", "//z@", 1) + "/dav/"})
	if err := st.Write(ctx, "f", []byte("x")); err == nil || !strings.Contains(err.Error(), "~/.netrc") {
		t.Errorf("a wrong password: %v", err)
	}
	st, _ = OpenWebDAV(WebDAVConfig{Name: "nas", URL: strings.Replace(d.URL, "//", "//u:p@", 1) + "/dav/"})
	if err := st.Write(ctx, "f", []byte("x")); err != nil {
		t.Errorf("the URL's password: %v", err)
	}
	if strings.Contains(st.String(), "p@") || strings.Contains(st.String(), ":p") {
		t.Errorf("the password is shown: %s", st)
	}
	if l, p := parseNetrc("machine h login a password b\nmachine h login c password d\n", "h", "c"); l != "c" || p != "d" {
		t.Errorf("netrc by login: %s %s", l, p)
	}
	if l, p := parseNetrc("default login a password b", "h", ""); l != "a" || p != "b" {
		t.Errorf("netrc default: %s %s", l, p)
	}
}

// TestWebDAVSelfSigned reads over https from a server whose certificate no authority signed: an
// error by default, read with ?tls_sha256= of its certificate or with ?insecure=true.
func TestWebDAVSelfSigned(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	write(t, filepath.Join(root, "hwnpu", "a.txt"), "aaa")
	h := &webdav.Handler{Prefix: "/dav", FileSystem: webdav.Dir(root), LockSystem: webdav.NewMemLS()}
	srv := httptest.NewTLSServer(h)
	defer srv.Close()
	sum := sha256.Sum256(srv.Certificate().Raw)
	for _, c := range []struct {
		query string
		ok    bool
	}{
		{"", false},
		{"?insecure=false", false},
		{"?tls_sha256=" + hex.EncodeToString(sum[:]), true},
		{"?tls_sha256=" + strings.Repeat("00", sha256.Size), false},
		{"?insecure=true", true},
		{"?insecure=true&writeback=true", true},
	} {
		s, err := ParseSource("nas", srv.URL+"/dav/hwnpu"+c.query)
		if err != nil {
			t.Fatalf("%q: %v", c.query, err)
		}
		b, err := s.Store.Read(ctx, "a.txt")
		if ok := err == nil && string(b) == "aaa"; ok != c.ok {
			t.Errorf("%q: read %q %v", c.query, b, err)
		}
	}
	if _, err := ParseSource("nas", srv.URL+"/dav?insecure=maybe"); err == nil {
		t.Error("insecure=maybe: no error")
	}
}

// TestWebDAVDown reads past a server that is off as if it were not set: the source after it
// answers, at once, and nothing more is asked of the one off.
func TestWebDAVDown(t *testing.T) {
	t.Setenv(cache.RootEnv, t.TempDir())
	ctx := context.Background()
	dead := httptest.NewServer(http.NotFoundHandler())
	dead.Close() // its port refuses connections now
	nas, err := ParseSource("nas", dead.URL+"/dav/hwnpu?writeback=true")
	if err != nil {
		t.Fatal(err)
	}
	folder := Source{Store: Dir{Name: "folder", Root: t.TempDir()}, Writeback: true}
	c := strings.Repeat("c", 40)
	src := filepath.Join(t.TempDir(), "config.json")
	write(t, src, "{}")
	if err := PutOriginal(ctx, folder, "lab/m", c, "config.json", src, ""); err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	clearLocks()
	i, _, err := FetchOriginal(ctx, []Source{nas, folder}, "lab/m", c, "config.json", "", filepath.Join(t.TempDir(), "c.json"), nil)
	if err != nil || i != 1 {
		t.Fatalf("FetchOriginal: %d %v", i, err)
	}
	if err := PutOriginal(ctx, nas, "lab/m", c, "config.json", src, ""); !unreachable(err) {
		t.Errorf("a write to the server off: %v", err)
	}
	if d := time.Since(start); d > 2*time.Second {
		t.Errorf("took %v: the server off was tried again", d)
	}
}

// TestWebDAVConnections writes files in new folders (MKCOL, PUT and MOVE, whose answers are not
// read) and reads them back over one connection: each new one asked for a NAS's .local address
// again, 5 seconds a request (docs/design/distribution.md).
func TestWebDAVConnections(t *testing.T) {
	ctx := context.Background()
	d := newDAV(t)
	st := openDAV(t, d)
	for i := range 20 {
		if err := st.Write(ctx, fmt.Sprintf("converted/m/t/e%d/v0/f", i), []byte("x")); err != nil {
			t.Fatal(err)
		}
		if _, ok, err := st.Has(ctx, fmt.Sprintf("converted/m/t/e%d/v0/f", i)); err != nil || !ok {
			t.Fatal(ok, err)
		}
	}
	if err := st.Delete(ctx, "converted/m/t/e0/v0/f"); err != nil {
		t.Fatal(err)
	}
	if n := d.conns.Load(); n > 1 {
		t.Errorf("%d connections for %d requests", n, d.requests.Load())
	}
}

// TestWebDAVStall cuts a download whose server stops sending after half of it (and one whose
// answer does not come) after davStall, and goes on from where it stopped.
func TestWebDAVStall(t *testing.T) {
	defer func(d time.Duration) { davStall = d }(davStall)
	davStall = 200 * time.Millisecond
	data := make([]byte, 1<<20)
	rand.Read(data)
	var gets atomic.Int64
	release := make(chan struct{})
	defer close(release)
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "f"), data, 0o644); err != nil {
		t.Fatal(err)
	}
	h := &webdav.Handler{Prefix: "/dav", FileSystem: webdav.Dir(root), LockSystem: webdav.NewMemLS()}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			h.ServeHTTP(w, r)
			return
		}
		switch gets.Add(1) {
		case 1: // its answer does not come
			select {
			case <-release:
			case <-r.Context().Done():
			}
			return
		case 2: // half, then nothing
			w.Header().Set("Content-Length", fmt.Sprint(len(data)))
			w.Write(data[:len(data)/2])
			w.(http.Flusher).Flush()
			select {
			case <-release:
			case <-r.Context().Done():
			}
			return
		}
		h.ServeHTTP(w, r)
	}))
	defer srv.Close()
	t.Setenv("NETRC", filepath.Join(t.TempDir(), "none"))
	st, err := OpenWebDAV(WebDAVConfig{Name: "nas", URL: srv.URL + "/dav"})
	if err != nil {
		t.Fatal(err)
	}
	dst := filepath.Join(t.TempDir(), "f")
	start := time.Now()
	n, err := st.Get(context.Background(), "f", dst, nil)
	if err != nil || n != int64(len(data)) {
		t.Fatalf("Get: %d %v", n, err)
	}
	if got, _ := os.ReadFile(dst); !bytes.Equal(got, data) {
		t.Error("the file is not the server's")
	}
	if g := gets.Load(); g != 3 {
		t.Errorf("%d GETs, want 3", g)
	}
	if d := time.Since(start); d > 10*time.Second {
		t.Errorf("took %v", d)
	}
}

// slowServer serves data's GETs from their Range: the first slowGets send fast bytes, then
// trickle bytes every tick until the client goes; the rest are the WebDAV handler's. trickle
// alone (fast 0) sends every GET at one slow rate.
func slowServer(t *testing.T, data []byte, slowGets int64, fast, trickle int, tick time.Duration) (*WebDAV, *atomic.Int64) {
	t.Helper()
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "f"), data, 0o644); err != nil {
		t.Fatal(err)
	}
	var gets atomic.Int64
	h := &webdav.Handler{Prefix: "/dav", FileSystem: webdav.Dir(root), LockSystem: webdav.NewMemLS()}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || gets.Add(1) > slowGets {
			h.ServeHTTP(w, r)
			return
		}
		var off int
		fmt.Sscanf(r.Header.Get("Range"), "bytes=%d-", &off)
		w.Header().Set("Content-Length", fmt.Sprint(len(data)-off))
		if off > 0 {
			w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", off, len(data)-1, len(data)))
			w.WriteHeader(http.StatusPartialContent)
		}
		end := min(off+fast, len(data))
		w.Write(data[off:end])
		w.(http.Flusher).Flush()
		for off = end; off < len(data); off += trickle {
			select {
			case <-r.Context().Done():
				return
			case <-time.After(tick):
			}
			w.Write(data[off:min(off+trickle, len(data))])
			w.(http.Flusher).Flush()
		}
	}))
	t.Cleanup(srv.Close)
	t.Setenv("NETRC", filepath.Join(t.TempDir(), "none"))
	st, err := OpenWebDAV(WebDAVConfig{Name: "nas", URL: srv.URL + "/dav"})
	if err != nil {
		t.Fatal(err)
	}
	return st, &gets
}

// TestWebDAVSlow cuts a download that slows to a trickle after davStall and goes on from where
// it stopped, more times than a round's tries: each cut brought more of the file.
func TestWebDAVSlow(t *testing.T) {
	defer func(d time.Duration, f int64) { davStall, davSlowFloor = d, f }(davStall, davSlowFloor)
	davStall, davSlowFloor = 200*time.Millisecond, 32<<10
	data := make([]byte, 1<<20)
	rand.Read(data)
	// four GETs send 128 KiB, then 100 bytes every 50 ms (0.4 KiB a davStall)
	st, gets := slowServer(t, data, 4, 128<<10, 100, 50*time.Millisecond)
	dst := filepath.Join(t.TempDir(), "f")
	start := time.Now()
	n, err := st.Get(context.Background(), "f", dst, nil)
	if err != nil || n != int64(len(data)) {
		t.Fatalf("Get: %d %v", n, err)
	}
	if got, _ := os.ReadFile(dst); !bytes.Equal(got, data) {
		t.Error("the file is not the server's")
	}
	if g := gets.Load(); g != 5 {
		t.Errorf("%d GETs, want 5", g)
	}
	if d := time.Since(start); d > 20*time.Second {
		t.Errorf("took %v", d)
	}
}

// TestWebDAVSlowAllAlong leaves a download that is slow from its start: it is not cut.
func TestWebDAVSlowAllAlong(t *testing.T) {
	defer func(d time.Duration, f int64) { davStall, davSlowFloor = d, f }(davStall, davSlowFloor)
	davStall, davSlowFloor = 200*time.Millisecond, 32<<10
	data := make([]byte, 256<<10)
	rand.Read(data)
	// 4 KiB every 20 ms: 40 KiB a davStall, over the floor, the same all along
	st, gets := slowServer(t, data, 1, 0, 4<<10, 20*time.Millisecond)
	dst := filepath.Join(t.TempDir(), "f")
	n, err := st.Get(context.Background(), "f", dst, nil)
	if err != nil || n != int64(len(data)) {
		t.Fatalf("Get: %d %v", n, err)
	}
	if g := gets.Load(); g != 1 {
		t.Errorf("%d GETs, want 1", g)
	}
}
