package store

import (
	"cmp"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"io"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"sync"
	"time"
)

// What is downloaded from Hugging Face goes in the sources written back to as it comes (a tee),
// besides the file here, so that a large checkpoint is not put after it is whole
// (docs/cache-sharing.md 7.1). It goes under a temporary name and is moved into place, and named
// in the lock, only when the download is whole and its SHA-256 the one expected. A source that
// takes no stream (S3), one that falls behind (TeeBuffer) or fails drops out of the tee: the file
// goes in it from the queue after the download, as before.

// streamer is a Store that takes an object as its bytes are written (a folder, WebDAV).
type streamer interface {
	// stream starts putting the object of key, of size bytes: what is written to it goes under
	// a temporary name; commit moves it into place, abort removes it.
	stream(ctx context.Context, key string, size int64) (putStream, error)
}

type putStream interface {
	io.Writer
	commit() error
	abort()
}

// TeeBuffer is how far a source may fall behind the download (bytes) before it drops out.
var TeeBuffer int64 = 64 << 20

// errBehind is a source that fell behind the download.
var errBehind = errors.New("fell behind the download")

// Tee puts a file of a repository in a source as it is downloaded here.
type Tee struct {
	s                  Source
	repo, commit, name string
	size               int64
	start              time.Time
	ps                 putStream
	h                  hash.Hash
	ch                 chan []byte
	done               chan struct{}
	mu                 sync.Mutex
	pending, written   int64
	err                error // why it dropped out
}

// StartTee starts putting the file name of repo of commit, of size bytes, in s as it is
// downloaded (Write). It is nil when s takes no stream, holds the file already, or names another
// commit in its lock (the file then goes as it went before: FillLater, which says why).
func StartTee(ctx context.Context, s Source, repo, commit, name string, size int64) *Tee {
	st, ok := s.Store.(streamer)
	if !ok || !s.Writeback || commit == "" || size < 0 {
		return nil
	}
	switch l, err := readLock(ctx, s, repo); {
	case err == nil && (l.Commit != commit || l.Files[name].Size == size):
		return nil
	case err != nil && !errors.Is(err, ErrNotFound):
		return nil
	}
	ps, err := st.stream(ctx, OriginalKey(repo, name), size)
	if err != nil {
		logQueue("%s/%s: not put in %s as it is downloaded: %v", repo, name, s, err)
		return nil
	}
	t := &Tee{s: s, repo: repo, commit: commit, name: name, size: size, start: time.Now(), ps: ps,
		h: sha256.New(), ch: make(chan []byte, 4096), done: make(chan struct{})}
	go t.run()
	return t
}

func (t *Tee) run() {
	defer close(t.done)
	for b := range t.ch {
		t.mu.Lock()
		failed := t.err != nil
		t.mu.Unlock()
		if failed {
			continue // drain: the download goes on
		}
		_, err := t.ps.Write(b)
		t.mu.Lock()
		t.pending -= int64(len(b))
		t.written += int64(len(b))
		if err != nil && t.err == nil {
			t.err = err
		}
		t.mu.Unlock()
	}
}

// Write hands p to the source; it never blocks the download long nor fails it: a source that
// falls TeeBuffer behind drops out.
func (t *Tee) Write(p []byte) (int, error) {
	if t == nil {
		return len(p), nil
	}
	t.h.Write(p)
	t.mu.Lock()
	if t.err == nil && t.pending+int64(len(p)) > TeeBuffer {
		t.err = errBehind
	}
	ok := t.err == nil
	if ok {
		t.pending += int64(len(p))
	}
	t.mu.Unlock()
	if ok {
		select {
		case t.ch <- append([]byte(nil), p...):
		default: // the source holds the chunks back: it is behind
			t.mu.Lock()
			t.pending -= int64(len(p))
			if t.err == nil {
				t.err = errBehind
			}
			t.mu.Unlock()
		}
	}
	return len(p), nil
}

// Source is the source the tee puts the file in.
func (t *Tee) Source() Source { return t.s }

// Finish ends the tee when the download has ended: whole (err nil) with the SHA-256 sum ("" when
// none is known before), or not. A file whole and as expected is moved into place and named in
// the lock; else it is removed. It reports whether the file is in the source; if not, the caller
// puts it from the queue (FillLater) as before.
func (t *Tee) Finish(ctx context.Context, sum string, downloadErr error) bool {
	if t == nil {
		return false
	}
	close(t.ch)
	t.mu.Lock()
	err := t.err
	t.mu.Unlock()
	err = cmp.Or(downloadErr, err)
	if err == nil {
		select {
		case <-t.done: // the rest of the buffer went
		case <-time.After(teeDrain):
			err = fmt.Errorf("the rest not put in %v", teeDrain)
		}
	}
	if err == nil {
		t.mu.Lock()
		err = t.err
		t.mu.Unlock()
	}
	got := hex.EncodeToString(t.h.Sum(nil))
	switch {
	case err != nil:
	case t.written != t.size:
		err = fmt.Errorf("%d bytes of %d", t.written, t.size)
	case sum != "" && got != sum:
		err = fmt.Errorf("SHA-256 %s, not %s", got, sum)
	}
	if err != nil {
		t.ps.abort() // ends a write that waits on the source too
		<-t.done
	} else {
		err = t.ps.commit()
	}
	if err == nil {
		err = Record(ctx, t.s, t.repo, t.commit, t.name, LockedFile{SHA256: got, Size: t.size}, false)
	}
	if err != nil {
		if downloadErr == nil {
			logQueue("%s/%s: not put in %s as it was downloaded: %v (put after it)", t.repo, t.name, t.s, err)
		}
		return false
	}
	logQueue("%s/%s: put in %s as it was downloaded (%s)", t.repo, t.name, t.s, rate(t.size, time.Since(t.start)))
	return true
}

// teeDrain bounds the wait, after the download, for what a source has not taken yet.
var teeDrain = 2 * time.Minute

// a folder's stream: a temporary file beside the object, renamed into place

type dirStream struct {
	f   *os.File
	dst string
}

func (d Dir) stream(_ context.Context, key string, _ int64) (putStream, error) {
	p, err := d.file(key)
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return nil, err
	}
	f, err := os.CreateTemp(filepath.Dir(p), ".put-*")
	if err != nil {
		return nil, err
	}
	return &dirStream{f: f, dst: p}, nil
}

func (s *dirStream) Write(p []byte) (int, error) { return s.f.Write(p) }

func (s *dirStream) commit() error {
	if err := s.f.Close(); err != nil {
		os.Remove(s.f.Name())
		return err
	}
	return os.Rename(s.f.Name(), s.dst)
}

func (s *dirStream) abort() {
	s.f.Close()
	os.Remove(s.f.Name())
}

// WebDAV's stream: one PUT of a temporary name whose body is what is written (not tried again:
// the bytes are gone), then a MOVE into place

type davStream struct {
	w        *WebDAV
	ctx      context.Context
	tmp, key string
	pw       *io.PipeWriter
	res      chan error
}

func (w *WebDAV) stream(ctx context.Context, key string, size int64) (putStream, error) {
	if err := w.writable(); err != nil {
		return nil, err
	}
	if err := w.checkKey(key); err != nil {
		return nil, err
	}
	dir := path.Dir(key)
	if dir == "." {
		dir = ""
	}
	if err := w.mkdirs(ctx, dir); err != nil {
		return nil, err
	}
	var rnd [8]byte
	rand.Read(rnd[:])
	s := &davStream{w: w, ctx: ctx, tmp: path.Join(dir, ".put-"+hex.EncodeToString(rnd[:])), key: key, res: make(chan error, 1)}
	pr, pw := io.Pipe()
	s.pw = pw
	go func() {
		err := w.putOnce(ctx, w.url(s.tmp), pr, size)
		pr.CloseWithError(cmp.Or(err, io.ErrClosedPipe)) // a PUT that ended early: the writes fail
		s.res <- err
	}()
	return s, nil
}

func (s *davStream) Write(p []byte) (int, error) { return s.pw.Write(p) }

func (s *davStream) commit() error {
	s.pw.Close()
	if err := <-s.res; err != nil {
		s.remove()
		return err
	}
	hdr := http.Header{"Destination": {s.w.url(s.key)}, "Overwrite": {"T"}}
	if _, err := s.w.do(s.ctx, "MOVE", s.w.url(s.tmp), hdr, nil, http.StatusCreated, http.StatusNoContent, http.StatusOK); err != nil {
		s.remove()
		return err
	}
	return nil
}

func (s *davStream) abort() {
	s.pw.CloseWithError(errors.New("store: the download stopped"))
	<-s.res
	s.remove()
}

func (s *davStream) remove() {
	s.w.do(context.WithoutCancel(s.ctx), http.MethodDelete, s.w.url(s.tmp), nil, nil, http.StatusNoContent, http.StatusOK, http.StatusNotFound)
}

// putOnce PUTs body (of size bytes) to u in one request, signed in as the store's requests are.
func (w *WebDAV) putOnce(ctx context.Context, u string, body io.Reader, size int64) error {
	w.mu.Lock()
	down := w.down
	w.mu.Unlock()
	if down != nil {
		return down
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, u, body)
	if err != nil {
		return err
	}
	req.ContentLength = size
	if size == 0 {
		req.Body = http.NoBody
	}
	if w.user != "" || w.pass != "" {
		req.SetBasicAuth(w.user, w.pass)
	}
	resp, err := w.client.Do(req)
	if err != nil {
		return err
	}
	io.Copy(io.Discard, io.LimitReader(resp.Body, 64<<10))
	resp.Body.Close()
	switch resp.StatusCode {
	case http.StatusCreated, http.StatusNoContent, http.StatusOK:
		return nil
	}
	return &davError{Method: http.MethodPut, URL: u, StatusCode: resp.StatusCode, Status: resp.Status}
}
