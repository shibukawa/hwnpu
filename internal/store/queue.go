package store

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu/internal/cache"
)

// What is put in the layers goes in the background, after it is whole here (docs/cache-sharing.md
// 7.1): a job is written to work/queue under the cache's root first and removed when done, so
// that one cut off (the process ended, the network went) is done by the next process that runs.
// A job that fails is kept for the next one and logged to work/queue.log; it never stops a run.

// Layers are the layers in the order they are looked in (set from the definitions: chat).
var (
	layersMu sync.RWMutex
	layers   []Layer
)

// SetLayers sets the layers.
func SetLayers(ls []Layer) {
	layersMu.Lock()
	layers = ls
	layersMu.Unlock()
}

// Layers are the layers set.
func Layers() []Layer {
	layersMu.RLock()
	defer layersMu.RUnlock()
	return layers
}

// job is a piece of work on a layer.
type job struct {
	Op    string `json:"op"`    // "entry": PutEntry of Dir; "upstream": PutUpstream
	Layer string `json:"layer"` // the layer's String()
	Dir   string `json:"dir,omitempty"`
	Repo  string `json:"repo,omitempty"`
	// Commit, Name, Path and SHA256 are PutUpstream's.
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

// PutLater puts the entry of dir in layer l in the background.
func PutLater(l Layer, dir string) { enqueue(job{Op: "entry", Layer: l.String(), Dir: dir}) }

// FillLater puts a file of a checkpoint in layer l in the background (PutUpstream).
func FillLater(l Layer, repo, commit, name, path, sum string) {
	enqueue(job{Op: "upstream", Layer: l.String(), Repo: repo, Commit: commit, Name: name, Path: path, SHA256: sum})
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

// worker does the jobs of the queue one at a time until none is left.
func worker(wakeup chan struct{}) {
	defer queueBusy.Done()
	failed := map[string]bool{}
	for {
		files, _ := filepath.Glob(filepath.Join(queueDir(), "*.json"))
		did := false
		for _, f := range files {
			if failed[f] {
				continue
			}
			did = true
			if err := runJob(f); err != nil {
				failed[f] = true
				logQueue("%s: %v (left for the next run)", filepath.Base(f), err)
			}
		}
		if did {
			continue
		}
		select {
		case <-wakeup:
		case <-time.After(2 * time.Second):
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

func runJob(file string) error {
	ctx := context.Background()
	unlock, err := lockWork(ctx, file)
	if err != nil {
		return err
	}
	defer unlock()
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
	var l *Layer
	for _, x := range Layers() {
		if x.String() == j.Layer {
			l = &x
			break
		}
	}
	if l == nil {
		return fmt.Errorf("no layer %s now", j.Layer)
	}
	switch j.Op {
	case "entry":
		err = PutEntry(ctx, *l, j.Dir)
	case "upstream":
		err = PutUpstream(ctx, *l, j.Repo, j.Commit, j.Name, j.Path, j.SHA256)
	default:
		err = fmt.Errorf("a job %q this hwnpu does not know", j.Op)
	}
	if os.IsNotExist(err) {
		err = nil // what was to go is gone here
	}
	if err == nil {
		os.Remove(file)
		logQueue("%s: put in %s", describe(j), j.Layer)
	}
	return err
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
