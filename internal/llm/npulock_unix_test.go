//go:build unix

package llm

import (
	"os"
	"path/filepath"
	"strconv"
	"syscall"
	"testing"
)

// A process started under the lock (NPULockEnv: the descriptor of the lock's own file, which
// the process that holds it passed on) has it without waiting; a descriptor of another file, or
// none, is not taken for it.
func TestInheritedNPULock(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "neural-engine.lock")
	held, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	defer held.Close()
	if ok, err := tryLock(held); err != nil || !ok {
		t.Fatalf("lock: %v %v", ok, err)
	}
	// another open of the file does not get the lock: it is the open file's
	other, err := os.OpenFile(path, os.O_RDWR, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	defer other.Close()
	if ok, _ := tryLock(other); ok {
		t.Fatal("a second open of the file got the lock")
	}
	dup := func(f *os.File) int { // a descriptor of the same open file, as a child's descriptor 3
		fd, err := syscall.Dup(int(f.Fd()))
		if err != nil {
			t.Fatal(err)
		}
		for fd < 3 {
			if fd, err = syscall.Dup(fd); err != nil {
				t.Fatal(err)
			}
		}
		return fd
	}
	t.Setenv(NPULockEnv, "")
	if f := inheritedNPULock(path); f != nil {
		t.Error("a lock was inherited without the variable")
	}
	t.Setenv(NPULockEnv, strconv.Itoa(dup(held)))
	f := inheritedNPULock(path)
	if f == nil {
		t.Fatal("the lock's descriptor was not taken")
	}
	f.Close()
	stray, err := os.Create(filepath.Join(dir, "another"))
	if err != nil {
		t.Fatal(err)
	}
	defer stray.Close()
	t.Setenv(NPULockEnv, strconv.Itoa(dup(stray)))
	if f := inheritedNPULock(path); f != nil {
		t.Error("a descriptor of another file was taken for the lock")
	}
}
