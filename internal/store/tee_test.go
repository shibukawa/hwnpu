package store

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/cache"
)

// TestTee puts a file in a folder and on WebDAV as it is "downloaded": in place and named in the
// lock when whole and as expected; nothing (no temporary file either) when not.
func TestTee(t *testing.T) {
	t.Setenv(cache.RootEnv, t.TempDir())
	ctx := context.Background()
	body := []byte(strings.Repeat("0123456789", 100_000)) // 1 MB
	h := sha256.Sum256(body)
	sum := hex.EncodeToString(h[:])
	c := strings.Repeat("c", 40)
	sources := map[string]func() (Source, func() []string){
		"folder": func() (Source, func() []string) {
			root := t.TempDir()
			return Source{Store: Dir{Name: "nas", Root: root}, Writeback: true}, func() []string { return files(t, root) }
		},
		"webdav": func() (Source, func() []string) {
			d := newDAV(t)
			st := openDAV(t, d)
			return Source{Store: st, Writeback: true}, func() []string { return files(t, filepath.Join(d.root, "hwnpu")) }
		},
	}
	for kind, mk := range sources {
		for _, tc := range []struct {
			name    string
			sum     string
			dlErr   error
			want    bool
			written int // bytes of body written
		}{
			{"whole", sum, nil, true, len(body)},
			{"no sum known", "", nil, true, len(body)},
			{"another sum", strings.Repeat("0", 64), nil, false, len(body)},
			{"cut off", "", errors.New("connection reset"), false, len(body) / 3},
			{"short", "", nil, false, len(body) / 2},
		} {
			t.Run(kind+"/"+tc.name, func(t *testing.T) {
				clearLocks()
				s, ls := mk()
				tee := StartTee(ctx, s, "lab/m", c, "w.bin", int64(len(body)))
				if tee == nil {
					t.Fatal("no tee")
				}
				for b := body[:tc.written]; len(b) > 0; {
					n := min(len(b), 32<<10)
					tee.Write(b[:n])
					b = b[n:]
				}
				if got := tee.Finish(ctx, tc.sum, tc.dlErr); got != tc.want {
					t.Fatalf("Finish: %v", got)
				}
				clearLocks()
				l, err := ReadLock(ctx, s, "lab/m")
				in := ls()
				if tc.want {
					if err != nil || l.Files["w.bin"].SHA256 != sum {
						t.Errorf("lock: %+v %v", l, err)
					}
					if !slicesHas(in, "original/lab/m/w.bin") {
						t.Errorf("files %v", in)
					}
				} else if err == nil || len(in) != 0 {
					t.Errorf("left: lock %+v %v, files %v", l, err, in)
				}
			})
		}
	}
}

// TestTeeBehind drops a source that falls behind, without holding the download back, and starts
// no tee where the file is already or the lock is of another commit.
func TestTeeBehind(t *testing.T) {
	t.Setenv(cache.RootEnv, t.TempDir())
	ctx := context.Background()
	root := t.TempDir()
	s := Source{Store: Dir{Name: "nas", Root: root}, Writeback: true}
	c := strings.Repeat("c", 40)
	old := TeeBuffer
	TeeBuffer = 10
	t.Cleanup(func() { TeeBuffer = old })
	tee := StartTee(ctx, s, "lab/m", c, "w.bin", 100)
	tee.Write(make([]byte, 100))
	if tee.Finish(ctx, "", nil) {
		t.Error("a source behind put the file")
	}
	if in := files(t, root); len(in) != 0 {
		t.Errorf("left %v", in)
	}
	TeeBuffer = old
	f := filepath.Join(t.TempDir(), "w.bin")
	write(t, f, strings.Repeat("x", 100))
	if err := PutOriginal(ctx, s, "lab/m", c, "w.bin", f, ""); err != nil {
		t.Fatal(err)
	}
	if StartTee(ctx, s, "lab/m", c, "w.bin", 100) != nil {
		t.Error("a tee of a file there")
	}
	if StartTee(ctx, s, "lab/m", strings.Repeat("d", 40), "x.bin", 100) != nil {
		t.Error("a tee into a lock of another commit")
	}
	if StartTee(ctx, Source{Store: s.Store}, "lab/m", c, "x.bin", 100) != nil {
		t.Error("a tee into a source not written back to")
	}
}

// files are the files under root, slash paths, but the queue's log.
func files(t *testing.T, root string) []string {
	t.Helper()
	var out []string
	filepath.WalkDir(root, func(p string, d os.DirEntry, err error) error {
		if err == nil && !d.IsDir() {
			r, _ := filepath.Rel(root, p)
			out = append(out, filepath.ToSlash(r))
		}
		return nil
	})
	return out
}

func slicesHas(ss []string, s string) bool {
	for _, x := range ss {
		if x == s {
			return true
		}
	}
	return false
}
