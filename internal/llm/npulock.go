package llm

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/mem"
)

// The Neural Engine is used by one hwnpu process at a time: two at once have frozen a Mac (M1:
// predictions that never return, processes that neither Ctrl+C nor kill -9 ends, the next
// commands stuck behind them). OpenDevice takes an exclusive lock on a file for the Core ML
// backend and keeps it while the process lives (the kernel drops it when the process ends); a
// second process waits for it, saying which process holds it. It is not skipped: hwnpud
// (docs/design/hwnpud.md) is how several apps share the Neural Engine. A process started by one
// that holds the lock runs under it (NPULockEnv: tools/coreai/aicheck locked, for a series of
// commands no other process is to come between).
//
// AMD's NPU takes the same lock for a chat model's IRON export (chat's loadIRON): one chat model
// uses all of its hardware contexts, a second process's CREATE_HWCTX right after another's
// contexts ended fails now and then, and hwnpud's engine holds the NPU for every client.

var npuLock struct {
	sync.Mutex
	f *os.File
}

func npuLockPath() string {
	return cache.Sub("neural-engine.lock")
}

// lockNPU takes the lock unless this process has it, waiting until ctx ends.
func lockNPU(ctx context.Context) error {
	npuLock.Lock()
	defer npuLock.Unlock()
	if npuLock.f != nil {
		return nil
	}
	path := npuLockPath()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	if f := inheritedNPULock(path); f != nil {
		npuLock.f = f
		return nil
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
			who, _, _ := strings.Cut(strings.TrimSpace(string(holder)), "\n") // its first line
			if who == "" {
				who = "another process"
			}
			fmt.Fprintf(os.Stderr, "waiting for the %s: %s is using it\n", npuName(), who)
			told = true
		}
		select {
		case <-ctx.Done():
			f.Close()
			return ctx.Err()
		case <-time.After(200 * time.Millisecond):
		}
	}
	// who has it now, for a process that waits; then, once what the process before left on the
	// Neural Engine is gone (settle), what else is on it, for the process after
	before, _ := os.ReadFile(path)
	who := "process " + strconv.Itoa(os.Getpid()) + " (" + filepath.Base(os.Args[0]) + ")\n"
	f.Truncate(0)
	f.WriteAt([]byte(who), 0)
	npuLock.f = f
	if left, _, ok := settle(ctx, mem.NPUWired, lockBase(before), npuPoll, func(over int64) {
		fmt.Fprintf(os.Stderr, "waiting for the Neural Engine: %.1f GiB of the process before are still on it\n", float64(over)/(1<<30))
	}); ok {
		f.WriteAt([]byte(who+lockBaseKey+strconv.FormatInt(left, 10)+"\n"), 0)
	}
	return nil
}

// A process that ends leaves its programs on the Neural Engine for a while: 2.5 GiB of one that
// was killed for a tenth of a second to a second, 1.5 GiB of a chat for seven seconds on a busy
// machine. A program loaded and run meanwhile may be refused: Chroma's T5 failed twice with
// "Request cancelled", started the moment that chat's lock was free (docs/design/apple-neural-engine.md). So the
// lock is followed by a wait for that memory to go (settle).
//
// What is on the Neural Engine is known for the whole machine alone (mem.NPUWired), other
// programs' share included. To tell the two apart, a process writes into the lock file what was
// on it when its own wait ended (base: nothing of hwnpu's): the next process waits for what is
// over that.
const (
	lockBaseKey = "base "                // the lock file's second line: lockBaseKey and the bytes
	npuSlack    = 128 << 20              // more than the base by this much is the process before's
	npuFall     = 16 << 20               // a fall of this much says it is still being freed
	npuPoll     = 100 * time.Millisecond // between two looks
	npuStall    = 100                    // looks without a fall: what is left is another program's
	npuSettle   = 300                    // looks at most
	npuTell     = 10                     // looks before saying what is waited for
)

// lockBase is the base a lock file's content names, 0 when it names none (a file of an older
// hwnpu, or of a process that could not read the memory).
func lockBase(content []byte) int64 {
	for _, line := range strings.Split(string(content), "\n") {
		if v, ok := strings.CutPrefix(line, lockBaseKey); ok {
			n, _ := strconv.ParseInt(strings.TrimSpace(v), 10, 64)
			return max(n, 0)
		}
	}
	return 0
}

// settle waits until wired (mem.NPUWired: the memory wired for the Neural Engine, false where it
// cannot be read) is at most npuSlack over base, looking every poll: npuSettle looks at most, and
// npuStall looks without a fall end the wait, since what is left is then in use by another
// program. tell is told once how much is over the base when the wait is a long one. It returns
// what is wired when the wait ends (the base for the next process) and the looks it waited; ok is
// false where the memory cannot be read.
func settle(ctx context.Context, wired func() (int64, bool), base int64, poll time.Duration, tell func(over int64)) (left int64, looks int, ok bool) {
	n, ok := wired()
	if !ok {
		return 0, 0, false
	}
	low, still := n, 0
	for i := 1; n > base+npuSlack && i <= npuSettle; i++ {
		if i == npuTell && tell != nil {
			tell(n - base)
		}
		if ctx.Err() != nil {
			return n, looks, true
		}
		select {
		case <-ctx.Done():
			return n, looks, true
		case <-time.After(poll):
		}
		looks = i
		v, ok := wired()
		if !ok {
			return n, looks, false
		}
		n = v
		if n <= low-npuFall {
			low, still = n, 0
		} else if still++; still >= npuStall {
			break
		}
	}
	return n, looks, true
}

// npuName names the NPU the lock is for in what a waiting process says.
func npuName() string {
	if runtime.GOOS == "darwin" {
		return "Neural Engine"
	}
	return "NPU"
}

// NPULockEnv names the descriptor of the lock a process was started under (RunLocked): the
// process that started it holds the lock, and keeps it until it ends itself.
const NPULockEnv = "HWNPU_NPU_LOCK_FD"

// inheritedNPULock is the lock's file passed on by the process that started this one holding it
// (NPULockEnv), nil without one: a descriptor of the lock's own file (path) that the lock is
// taken on. The lock is of the open file, which the two processes share, so no other process
// gets it between the commands of the one that holds it.
func inheritedNPULock(path string) *os.File {
	fd, err := strconv.Atoi(os.Getenv(NPULockEnv))
	if err != nil || fd < 3 {
		return nil
	}
	f := os.NewFile(uintptr(fd), path)
	if f == nil {
		return nil
	}
	got, err1 := f.Stat()
	want, err2 := os.Stat(path)
	if err1 != nil || err2 != nil || !os.SameFile(got, want) {
		return nil // not the lock's file: take the lock as any process
	}
	if ok, err := tryLock(f); err != nil || !ok {
		return nil
	}
	return f
}

// NPULockFile is the lock's file once this process holds it (LockNPU), for a command started
// under it: passed as the command's descriptor 3 with NPULockEnv set to 3.
func NPULockFile() *os.File {
	npuLock.Lock()
	defer npuLock.Unlock()
	return npuLock.f
}

// LockNPU takes the NPU's lock as OpenDevice does, for what opens the device by other means:
// hwnpu-dev probe cache, whose measurements another process compiling for the Neural Engine
// would spoil, and the chat models' IRON exports on AMD. It is held until the process ends.
func LockNPU(ctx context.Context) error { return lockNPU(ctx) }
