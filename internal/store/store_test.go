package store

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/store/s3fake"
)

func write(t *testing.T, p, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func fakeS3(t *testing.T, prefix string) (*s3fake.Server, *S3) {
	t.Helper()
	srv := s3fake.New(t)
	t.Setenv("AWS_ACCESS_KEY_ID", "id")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "secret")
	t.Setenv("AWS_PROFILE", "")
	st, err := OpenS3(S3Config{Name: "lab", Bucket: "models", Prefix: prefix, Endpoint: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	return srv, st
}

// TestStores puts, reads, lists and deletes objects in a folder and in S3 alike.
func TestStores(t *testing.T) {
	ctx := context.Background()
	srv, s3 := fakeS3(t, "cache")
	for _, st := range []Store{Dir{Name: "share", Root: t.TempDir()}, s3} {
		t.Run(st.String(), func(t *testing.T) {
			src := filepath.Join(t.TempDir(), "f")
			write(t, src, "hello")
			if err := st.Put(ctx, "a/b/f", src); err != nil {
				t.Fatal(err)
			}
			if err := st.Write(ctx, "a/c.json", []byte(`{}`)); err != nil {
				t.Fatal(err)
			}
			if n, ok, err := st.Has(ctx, "a/b/f"); err != nil || !ok || n != 5 {
				t.Errorf("Has: %d %v %v", n, ok, err)
			}
			if _, ok, err := st.Has(ctx, "a/none"); err != nil || ok {
				t.Errorf("Has of none: %v %v", ok, err)
			}
			dst := filepath.Join(t.TempDir(), "x", "f")
			if n, err := st.Get(ctx, "a/b/f", dst, nil); err != nil || n != 5 {
				t.Fatalf("Get: %d %v", n, err)
			}
			if b, _ := os.ReadFile(dst); string(b) != "hello" {
				t.Errorf("got %q", b)
			}
			if _, err := st.Get(ctx, "a/none", dst+"2", nil); !isNotFound(err) {
				t.Errorf("Get of none: %v", err)
			}
			keys, err := st.List(ctx, "a/")
			if err != nil || strings.Join(keys, " ") != "a/b/f a/c.json" {
				t.Errorf("List: %v %v", keys, err)
			}
			if err := st.Delete(ctx, "a/b/f"); err != nil {
				t.Fatal(err)
			}
			if _, ok, _ := st.Has(ctx, "a/b/f"); ok {
				t.Error("deleted, still there")
			}
			if _, err := st.Read(ctx, "../x"); err == nil {
				t.Error("a key out of the store was taken")
			}
		})
	}
	if _, ok := srv.Get("models", "cache/a/c.json"); !ok {
		t.Error("the prefix was not used")
	}
	if srv.Signed == 0 {
		t.Error("nothing was signed")
	}
}

func isNotFound(err error) bool {
	return err != nil && strings.Contains(err.Error(), ErrNotFound.Error())
}

// TestS3Large puts a file above the part size in parts, and goes on with a download cut off.
func TestS3Large(t *testing.T) {
	old := partSize
	partSize = 1 << 10
	t.Cleanup(func() { partSize = old })
	ctx := context.Background()
	srv, st := fakeS3(t, "")
	big := bytes.Repeat([]byte("0123456789abcdef"), 200) // 3200 bytes: 4 parts
	src := filepath.Join(t.TempDir(), "big")
	write(t, src, string(big))
	if err := st.Put(ctx, "big", src); err != nil {
		t.Fatal(err)
	}
	if b, _ := srv.Get("models", "big"); !bytes.Equal(b, big) {
		t.Fatalf("the parts made %d bytes", len(b))
	}
	if srv.Requests["POST"] != 2 {
		t.Errorf("%d POST (create and complete)", srv.Requests["POST"])
	}
	dst := filepath.Join(t.TempDir(), "big")
	write(t, dst+".part", string(big[:1000]))
	gets := srv.Requests["GET"]
	if _, err := st.Get(ctx, "big", dst, nil); err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(dst); !bytes.Equal(b, big) {
		t.Errorf("went on to %d bytes", len(b))
	}
	if srv.Requests["GET"]-gets != 1 {
		t.Errorf("%d GETs to go on", srv.Requests["GET"]-gets)
	}
}

// TestS3Unsigned reads a public bucket without keys.
func TestS3Unsigned(t *testing.T) {
	srv := s3fake.New(t)
	t.Setenv("AWS_ACCESS_KEY_ID", "")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "")
	t.Setenv("AWS_PROFILE", "none")
	t.Setenv("AWS_SHARED_CREDENTIALS_FILE", filepath.Join(t.TempDir(), "none"))
	if _, err := OpenS3(S3Config{Name: "lab", Bucket: "models", Endpoint: srv.URL}); err == nil || !strings.Contains(err.Error(), "no keys") {
		t.Errorf("no keys: %v", err)
	}
	srv.Put("models", "p/f", []byte("public"))
	st, err := OpenS3(S3Config{Name: "lab", Bucket: "models", Prefix: "p", Endpoint: srv.URL, NoSign: true})
	if err != nil {
		t.Fatal(err)
	}
	b, err := st.Read(context.Background(), "f")
	if err != nil || string(b) != "public" {
		t.Errorf("%q %v", b, err)
	}
	if srv.Signed != 0 {
		t.Error("signed")
	}
	if err := st.Write(context.Background(), "g", nil); err == nil {
		t.Error("a public bucket was written")
	}
}

// TestProfile reads the static keys, the region and the endpoint of a profile.
func TestProfile(t *testing.T) {
	d := t.TempDir()
	write(t, filepath.Join(d, "credentials"), "[default]\naws_access_key_id = D\naws_secret_access_key = DS\n[rustfs]\naws_access_key_id = R\naws_secret_access_key = RS\n")
	write(t, filepath.Join(d, "config"), "[profile rustfs]\nregion = lab-1\ns3 =\n  endpoint_url = http://rustfs:9000\n  addressing_style = path\n")
	t.Setenv("AWS_SHARED_CREDENTIALS_FILE", filepath.Join(d, "credentials"))
	t.Setenv("AWS_CONFIG_FILE", filepath.Join(d, "config"))
	t.Setenv("AWS_PROFILE", "")
	t.Setenv("AWS_ACCESS_KEY_ID", "")
	p := readProfile("rustfs")
	if p.creds.AccessKeyID != "R" || p.creds.SecretAccessKey != "RS" || p.region != "lab-1" || p.endpoint != "http://rustfs:9000" {
		t.Errorf("rustfs: %+v", p)
	}
	if p := readProfile(""); p.creds.AccessKeyID != "D" {
		t.Errorf("default: %+v", p)
	}
	t.Setenv("AWS_ACCESS_KEY_ID", "E")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "ES")
	if p := readProfile(""); p.creds.AccessKeyID != "E" {
		t.Errorf("the environment's keys: %+v", p)
	}
}

// TestEntries puts an export in one layer and makes it from there, checks it, and finds a
// checkpoint's file by its SHA-256 and by its commit.
func TestEntries(t *testing.T) {
	t.Setenv(cache.RootEnv, t.TempDir())
	ctx := context.Background()
	exp := filepath.Join(t.TempDir(), "m.amd-npu2")
	write(t, filepath.Join(exp, "manifest.json"), `{"version":1}`)
	write(t, filepath.Join(exp, "bank.bin"), "weights")
	write(t, filepath.Join(exp, "check", "x.bin"), "weights") // the same bytes: one blob
	recipe := map[string]string{"model": "m", "commit": "abc"}
	e, err := Describe(exp, Entry{Kind: "iron-lm", Scope: ScopePortable, Recipe: recipe})
	if err != nil {
		t.Fatal(err)
	}
	if e.Key != Fingerprint(recipe) || len(e.Files) != 3 {
		t.Fatalf("%+v", e)
	}
	if err := Verify(exp, true); err != nil {
		t.Fatal(err)
	}
	near, far := Dir{Name: "near", Root: t.TempDir()}, Dir{Name: "far", Root: t.TempDir()}
	layers := []Layer{{Store: near, Fill: true}, {Store: far, Write: true}}
	if err := PutEntry(ctx, layers[1], exp); err != nil {
		t.Fatal(err)
	}
	blobs, _ := far.List(ctx, "blobs/")
	if len(blobs) != 2 {
		t.Errorf("%d blobs for 2 contents", len(blobs))
	}
	out := filepath.Join(t.TempDir(), "m.amd-npu2")
	i, err := FetchEntry(ctx, layers, ScopePortable, "iron-lm", e.Key, out, nil)
	if err != nil || i != 1 {
		t.Fatalf("%d %v", i, err)
	}
	if err := Verify(out, true); err != nil {
		t.Fatal(err)
	}
	if _, err := FetchEntry(ctx, layers, ScopePortable, "iron-lm", Fingerprint(map[string]string{"model": "other"}), out+"2", nil); err != ErrNotFound {
		t.Errorf("another key: %v", err)
	}
	// a layer of other scopes is not looked in
	if _, err := FetchEntry(ctx, []Layer{{Store: far, Scopes: []string{ScopeUpstream}}}, ScopePortable, "iron-lm", e.Key, out+"3", nil); err != ErrNotFound {
		t.Errorf("a layer of upstream only: %v", err)
	}
	// a blob that changed in the layer is found out
	b0 := e.Files[0].SHA256
	write(t, filepath.Join(far.Root, "blobs", "sha256", b0[:2], b0), "tampered")
	if _, err := FetchEntry(ctx, layers, ScopePortable, "iron-lm", e.Key, out+"4", nil); err == nil {
		t.Error("a changed blob was taken")
	}

	ck := filepath.Join(t.TempDir(), "config.json")
	write(t, ck, `{"a":1}`)
	sum, _ := FileSHA256(ck)
	if err := PutUpstream(ctx, layers[1], "lab/m", "c0ffee", "config.json", ck, ""); err != nil {
		t.Fatal(err)
	}
	dst := filepath.Join(t.TempDir(), "config.json")
	if i, got, err := FetchUpstream(ctx, layers, "lab/m", "c0ffee", "config.json", "", dst, nil); err != nil || i != 1 || got != sum {
		t.Errorf("by commit: %d %s %v", i, got, err)
	}
	os.Remove(dst)
	if _, _, err := FetchUpstream(ctx, layers, "lab/m", "", "config.json", sum, dst, nil); err != nil {
		t.Errorf("by SHA-256: %v", err)
	}
	os.Remove(dst)
	if _, _, err := FetchUpstream(ctx, layers, "lab/m", "other", "config.json", "", dst, nil); err != ErrNotFound {
		t.Errorf("another commit: %v", err)
	}
}

// TestQueue puts an entry in a layer in the background and leaves work that fails for later.
func TestQueue(t *testing.T) {
	t.Setenv(cache.RootEnv, t.TempDir())
	ctx := context.Background()
	exp := filepath.Join(t.TempDir(), "e")
	write(t, filepath.Join(exp, "a"), "a")
	if _, err := Describe(exp, Entry{Kind: "k", Recipe: map[string]string{"x": "1"}}); err != nil {
		t.Fatal(err)
	}
	l := Layer{Store: Dir{Name: "share", Root: t.TempDir()}, Write: true}
	SetLayers([]Layer{l})
	t.Cleanup(func() { SetLayers(nil) })
	PutLater(l, exp)
	PutLater(Layer{Store: Dir{Name: "gone", Root: "/nonexistent"}}, exp) // no such layer now: kept
	Flush(ctx)
	if es, _ := ListEntries(ctx, l); len(es) != 1 {
		t.Errorf("entries %v", es)
	}
	left, _ := filepath.Glob(filepath.Join(cache.Sub("work", "queue"), "*.json"))
	if len(left) != 1 {
		t.Errorf("%d jobs left", len(left))
	}
	if b, _ := os.ReadFile(cache.Sub("work", "queue.log")); !strings.Contains(string(b), "no layer") {
		t.Errorf("log: %s", b)
	}
}
