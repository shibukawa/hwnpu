//go:build !unix

package mapped

import "os"

// Open reads the file at path (no mapping here).
func Open(path string) (*File, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return &File{b: b}, nil
}

// Anon is n bytes to be written (Go's heap here).
func Anon(n int) (*File, error) { return &File{b: make([]byte, n)}, nil }

func unmap([]byte) error { return nil }
