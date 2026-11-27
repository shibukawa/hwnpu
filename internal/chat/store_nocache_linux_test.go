package chat

import (
	"os"
	"path/filepath"
	"testing"
	"unsafe"

	"golang.org/x/sys/unix"
)

// resident is how many of the file's pages are in the file cache.
func resident(t *testing.T, path string) (in, all int) {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	st, _ := f.Stat()
	b, err := unix.Mmap(int(f.Fd()), 0, int(st.Size()), unix.PROT_READ, unix.MAP_SHARED)
	if err != nil {
		t.Fatal(err)
	}
	defer unix.Munmap(b)
	pg := os.Getpagesize()
	vec := make([]byte, (len(b)+pg-1)/pg)
	if _, _, e := unix.Syscall(unix.SYS_MINCORE, uintptr(unsafe.Pointer(&b[0])), uintptr(len(b)), uintptr(unsafe.Pointer(&vec[0]))); e != 0 {
		t.Fatal(e)
	}
	for _, v := range vec {
		in += int(v & 1)
	}
	return in, len(vec)
}

// The store's files leave no pages in the file cache once written and once read. On a disk: a
// tmpfs keeps its files in those pages.
func TestStoreFilesUncached(t *testing.T) {
	dir, err := os.MkdirTemp(filepath.Join(os.Getenv("HOME"), ".cache"), "hwnpu-nocache-")
	if err != nil {
		t.Skip(err)
	}
	defer os.RemoveAll(dir)
	var fs unix.Statfs_t
	if unix.Statfs(dir, &fs) == nil && fs.Type == unix.TMPFS_MAGIC {
		t.Skip("a tmpfs")
	}
	b := make([]byte, 8<<20)
	for i := range b {
		b[i] = byte(i * 7)
	}
	path := filepath.Join(dir, "rows.bin")
	if err := writeFile(path, b); err != nil {
		t.Fatal(err)
	}
	dropping.Wait()
	if in, all := resident(t, path); in > all/8 {
		t.Errorf("written: %d of %d pages in the file cache", in, all)
	}
	got, err := readUncached(path)
	if err != nil || len(got) != len(b) || got[12345] != b[12345] {
		t.Fatalf("read %d bytes (%v)", len(got), err)
	}
	if in, all := resident(t, path); in > all/8 {
		t.Errorf("read: %d of %d pages in the file cache", in, all)
	}
}
