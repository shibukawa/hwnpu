package chat

import (
	"io"
	"os"
	"sync"
)

// The prompt store's files (rows, marks, meta) are read once when a sequence comes back and written
// once when it is kept, and the system would keep their pages in its file cache after both, where
// on a machine short of memory they push out the model's own pages (openset-llm, patches/0007:
// "a state written on an 8 GB box must not stay in the page cache"). A mark of Qwen3.5 holds its
// recurrent states, so the marks are not small either. setNoCache and dropPageCache are per
// system (store_nocache_*.go).

// dropping is the written files dropPageCache still writes back on Linux; tests wait for it.
var dropping sync.WaitGroup

// writeUncached writes a file as os.WriteFile does, leaving none of its pages in the file cache.
func writeUncached(path string, b []byte) error {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	setNoCache(f)
	_, err = f.Write(b)
	if err == nil {
		dropPageCache(f, true)
	}
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	return err
}

// readUncached reads a file as os.ReadFile does, leaving none of its pages in the file cache.
func readUncached(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	setNoCache(f)
	b, err := io.ReadAll(f)
	if err != nil {
		return nil, err
	}
	dropPageCache(f, false)
	return b, nil
}
