package cache

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

// The locks of what is being prepared (docs/cache-sharing.md 6, model-table.md 11.2): a file in
// Root/locks held with flock (LockFileEx on Windows) while a process downloads, converts or
// exports one thing, so that two processes do not write it at once. The OS lets it go when the
// process ends, so that no stale lock is left. Within a process a mutex of the same name keeps
// goroutines apart.

var (
	inProcMu sync.Mutex
	inProc   = map[string]*sync.Mutex{}
)

var unsafeChars = regexp.MustCompile(`[^A-Za-z0-9._-]+`)

// lockFile is the file of a lock's name: readable, and unique by a hash of the whole name.
func lockFile(name string) string {
	h := sha256.Sum256([]byte(name))
	base := unsafeChars.ReplaceAllString(filepath.Base(name), "_")
	if len(base) > 60 {
		base = base[:60]
	}
	return Sub("locks", base+"-"+hex.EncodeToString(h[:6])+".lock")
}

// Lock takes the lock called name (a path of what is prepared, or any name), waiting until ctx
// ends; what says on log what is waited for once. It returns the function that lets it go.
func Lock(ctx context.Context, name, what string, log io.Writer) (func(), error) {
	inProcMu.Lock()
	mu := inProc[name]
	if mu == nil {
		mu = &sync.Mutex{}
		inProc[name] = mu
	}
	inProcMu.Unlock()
	got := make(chan struct{})
	go func() { mu.Lock(); close(got) }()
	select {
	case <-got:
	case <-ctx.Done():
		go func() { <-got; mu.Unlock() }()
		return nil, ctx.Err()
	}
	path := lockFile(name)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		mu.Unlock()
		return nil, err
	}
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o644)
	if err != nil {
		mu.Unlock()
		return nil, err
	}
	told := false
	for {
		ok, err := tryLock(f)
		if err != nil {
			f.Close()
			mu.Unlock()
			return nil, fmt.Errorf("cache: lock %s: %w", path, err)
		}
		if ok {
			break
		}
		if !told && log != nil {
			who := strings.TrimSpace(readHolder(path))
			if who == "" {
				who = "another process"
			}
			fmt.Fprintf(log, "waiting: %s is preparing %s\n", who, what)
			told = true
		}
		select {
		case <-ctx.Done():
			f.Close()
			mu.Unlock()
			return nil, ctx.Err()
		case <-time.After(200 * time.Millisecond):
		}
	}
	f.Truncate(0)
	f.WriteAt([]byte("process "+strconv.Itoa(os.Getpid())+" ("+filepath.Base(os.Args[0])+")\n"), 0)
	return func() {
		unlock(f)
		f.Close()
		mu.Unlock()
	}, nil
}

// readHolder is what the holder of a lock wrote in it. On Windows the locked range cannot be
// read: none is told then.
func readHolder(path string) string {
	b, _ := os.ReadFile(path)
	return string(b)
}
