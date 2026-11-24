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

func unmap([]byte) error { return nil }
