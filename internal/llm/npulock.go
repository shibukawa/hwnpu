package llm

import (
	"context"
	"fmt"
	"github.com/shibukawa/hwnpu/internal/cache"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

// The Neural Engine is used by one hwnpu process at a time: two at once have frozen a Mac (M1:
// predictions that never return, processes that neither Ctrl+C nor kill -9 ends, the next
// commands stuck behind them). OpenDevice takes an exclusive lock on a file for the Core ML
// backend and keeps it while the process lives (the kernel drops it when the process ends); a
// second process waits for it, saying which process holds it. HWNPU_NPU_LOCK=off skips it.
const npuLockEnv = "HWNPU_NPU_LOCK"

var npuLock struct {
	sync.Mutex
	f *os.File
}

func npuLockPath() string {
	return cache.Sub("neural-engine.lock")
}

// lockNPU takes the lock unless this process has it, waiting until ctx ends.
func lockNPU(ctx context.Context) error {
	if strings.EqualFold(os.Getenv(npuLockEnv), "off") {
		return nil
	}
	npuLock.Lock()
	defer npuLock.Unlock()
	if npuLock.f != nil {
		return nil
	}
	path := npuLockPath()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o644)
	if err != nil {
		return err
	}
	told := false
	for {
		ok, err := tryLock(f)
		if err != nil {
			f.Close()
			return fmt.Errorf("npu lock %s: %w", path, err)
		}
		if ok {
			break
		}
		if !told {
			holder, _ := os.ReadFile(path)
			who := strings.TrimSpace(string(holder))
			if who == "" {
				who = "another process"
			}
			fmt.Fprintf(os.Stderr, "waiting for the Neural Engine: %s is using it (%s=off to skip the wait)\n", who, npuLockEnv)
			told = true
		}
		select {
		case <-ctx.Done():
			f.Close()
			return ctx.Err()
		case <-time.After(200 * time.Millisecond):
		}
	}
	f.Truncate(0)
	f.WriteAt([]byte("process "+strconv.Itoa(os.Getpid())+" ("+filepath.Base(os.Args[0])+")\n"), 0)
	npuLock.f = f
	return nil
}

// LockNPU takes the Neural Engine's lock as OpenDevice does, for a tool that opens the device
// by other means (hwnpu-dev probe cache, whose measurements another process compiling for the
// Neural Engine would spoil). It is held until the process ends.
func LockNPU(ctx context.Context) error { return lockNPU(ctx) }
