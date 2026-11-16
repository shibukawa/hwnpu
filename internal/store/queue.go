package store

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu/internal/cache"
)

// What is written back to the sources goes in the background, after it is whole here
// (docs/cache-sharing.md 7.1): a job is written to work/queue under the cache's root first and removed when done, so
// that one cut off (the process ended, the network went) is done by the next process that runs.
// A job that fails is kept for the next one and logged to work/queue.log; it never stops a run.

// job is a piece of work on a source.
type job struct {
	Op     string `json:"op"`     // "entry": PutEntry of Dir; "original": PutOriginal
	Source string `json:"source"` // the source's String()
	Dir    string `json:"dir,omitempty"`
	Repo   string `json:"repo,omitempty"`
	// Commit, Name, Path and SHA256 are PutOriginal's.
	Commit string `json:"commit,omitempty"`
	Name   string `json:"name,omitempty"`
	Path   string `json:"path,omitempty"`
	SHA256 string `json:"sha256,omitempty"`
}

func (j job) id() string {
	b, _ := json.Marshal(j)
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:8])
}

var (
	queueMu   sync.Mutex
	queueWake chan struct{}
	queueBusy sync.WaitGroup
	resumed   sync.Once
)

func queueDir() string { return cache.Sub("work", "queue") }

// the units of the cache work is queued for are not removed (cache.Room, hwnpu cache prune)
func init() { cache.SetBusy(queuedPaths) }

// queuedPaths are the paths here the jobs of the queue put in the sources.
func queuedPaths() []string {
	files, _ := filepath.Glob(filepath.Join(queueDir(), "*.json"))
	var out []string
	for _, f := range files {
		var j job
		if b, err := os.ReadFile(f); err == nil && json.Unmarshal(b, &j) == nil {
			for _, p := range []string{j.Dir, j.Path} {
				if p != "" {
					out = append(out, p)
				}
			}
		}
	}
	return out
}

// PutLater puts the entry of dir in source s in the background (PutEntry).
func PutLater(s Source, dir string) { enqueue(job{Op: "entry", Source: s.String(), Dir: dir}) }

// FillLater puts a file of a repository in source s in the background (PutOriginal).
func FillLater(s Source, repo, commit, name, path, sum string) {
	enqueue(job{Op: "original", Source: s.String(), Repo: repo, Commit: commit, Name: name, Path: path, SHA256: sum})
}

func enqueue(j job) {
	dir := queueDir()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		logQueue("queue: %v", err)
		return
	}
	b, _ := json.Marshal(j)
	if err := os.WriteFile(filepath.Join(dir, j.id()+".json"), b, 0o644); err != nil {
		logQueue("queue: %v", err)
		return
	}
	Resume()
	wake()
}

func wake() {
	queueMu.Lock()
	defer queueMu.Unlock()
	if queueWake == nil {
		queueWake = make(chan struct{}, 1)
		queueBusy.Add(1)
		go worker(queueWake)
	}
	select {
	case queueWake <- struct{}{}:
	default:
	}
}

// Resume starts the work left in the queue by a process before, once per process.
func Resume() {
	resumed.Do(func() {
		if files, _ := filepath.Glob(filepath.Join(queueDir(), "*.json")); len(files) > 0 {
			wake()
		}
	})
}

// worker does the jobs of the queue, PutStreams at a time, until none is left.
func worker(wakeup chan struct{}) {
	defer queueBusy.Done()
	var (
		mu      sync.Mutex
		failed  = map[string]bool{}
		running = map[string]bool{}
	)
	done := make(chan struct{}, PutStreams)
	for {
		files, _ := filepath.Glob(filepath.Join(queueDir(), "*.json"))
		mu.Lock()
		for _, f := range files {
			if failed[f] || running[f] || len(running) >= PutStreams {
				continue
			}
			running[f] = true
			go func() {
				err := runJob(f)
				mu.Lock()
				delete(running, f)
				if err != nil {
					failed[f] = true
				}
				mu.Unlock()
				if err != nil {
					logQueue("%s: %v (left for the next run)", filepath.Base(f), err)
				}
				done <- struct{}{}
			}()
		}
		busy := len(running) > 0
		mu.Unlock()
		select {
		case <-done:
		case <-wakeup:
		case <-time.After(2 * time.Second):
			if busy {
				continue
			}
			queueMu.Lock()
			if len(wakeup) == 0 {
				queueWake = nil
				queueMu.Unlock()
				return
			}
			queueMu.Unlock()
		}
	}
}

// Flush waits until the queue's work is done (or failed) or ctx ends.
func Flush(ctx context.Context) {
	Resume()
	done := make(chan struct{})
	go func() { queueBusy.Wait(); close(done) }()
	select {
	case <-done:
	case <-ctx.Done():
	}
}

// Wait is Flush at the end of a command: when the queue holds work (put by this process or left
// by one before), it says so on w and waits until it is done or Ctrl-C, which leaves the rest
// for the next run. With none (a machine that writes nothing back), it returns at once.
func Wait(w io.Writer) {
	files, _ := filepath.Glob(filepath.Join(queueDir(), "*.json"))
	if len(files) == 0 {
		return
	}
	fmt.Fprintf(w, "writing %d back to the sources before ending (Ctrl-C leaves them for the next run)\n", len(files))
	wake() // the queue's worker may have ended (or Resume found none) before this work was there
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	Flush(ctx)
}

func runJob(file string) error {
	ctx := context.Background()
	unlock, err := lockWork(ctx, file)
	if err != nil {
		return err
	}
	defer unlock()
	start := time.Now() // after the lock: the time of the work, not of waiting for another process
	b, err := os.ReadFile(file)
	if os.IsNotExist(err) {
		return nil // another process did it
	}
	if err != nil {
		return err
	}
	var j job
	if err := json.Unmarshal(b, &j); err != nil {
		os.Remove(file)
		return err
	}
	var src *Source
	for _, x := range Sources() {
		if x.String() == j.Source {
			src = &x
			break
		}
	}
	if src == nil {
		return fmt.Errorf("no source %s now", j.Source)
	}
	switch j.Op {
	case "entry":
		err = PutEntry(ctx, *src, j.Dir)
	case "original":
		err = PutOriginal(ctx, *src, j.Repo, j.Commit, j.Name, j.Path, j.SHA256)
		if errors.Is(err, ErrOtherCommit) {
			logQueue("%s: %v (left as it is: hwnpu-dev cache update moves it)", describe(j), err)
			err = nil
		}
	default:
		err = fmt.Errorf("a job %q this hwnpu does not know", j.Op)
	}
	if os.IsNotExist(err) {
		err = nil // what was to go is gone here
	}
	if err == nil {
		os.Remove(file)
		logQueue("%s: put in %s (%s)", describe(j), j.Source, rate(size(j), time.Since(start)))
	}
	return err
}

// size is the bytes job j puts: its file, or the files of its entry's directory.
func size(j job) int64 {
	if j.Op != "entry" {
		fi, err := os.Stat(j.Path)
		if err != nil {
			return 0
		}
		return fi.Size()
	}
	var n int64
	filepath.WalkDir(j.Dir, func(_ string, d os.DirEntry, err error) error {
		if err == nil && !d.IsDir() {
			if fi, err := d.Info(); err == nil {
				n += fi.Size()
			}
		}
		return nil
	})
	return n
}

// rate says how much was put in d: "0.62 GB in 24.1s, 25.7 MB/s".
func rate(n int64, d time.Duration) string {
	s := fmt.Sprintf("%.2f GB in %.1fs", float64(n)/1e9, d.Seconds())
	if d > 0 {
		s += fmt.Sprintf(", %.1f MB/s", float64(n)/1e6/d.Seconds())
	}
	return s
}

func describe(j job) string {
	if j.Op == "entry" {
		return filepath.Base(j.Dir)
	}
	return j.Repo + "/" + j.Name
}

// logQueue appends a line to work/queue.log (the background work writes nothing on the
// terminal, where a chat is).
func logQueue(format string, args ...any) {
	p := cache.Sub("work", "queue.log")
	os.MkdirAll(filepath.Dir(p), 0o755)
	f, err := os.OpenFile(p, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return
	}
	defer f.Close()
	fmt.Fprintf(f, "%s %s\n", time.Now().Format(time.RFC3339), fmt.Sprintf(format, args...))
}
