//go:build !unix

package safetensors

import "fmt"

// Map reads a tensor's raw data (no mapping here).
func (f *File) Map(name string) ([]byte, error) { return f.Bytes(name) }

func (f *File) unmap() {}

// MappedAt is false: nothing is mapped here.
func (f *File) MappedAt([]byte) (int64, bool) { return 0, false }

// MapFloat32 is false: nothing is mapped here (the caller reads the tensor).
func (f *File) MapFloat32(name string) ([]float32, bool, error) {
	if _, ok := f.tensors[name]; !ok {
		return nil, false, fmt.Errorf("safetensors: no tensor %q", name)
	}
	return nil, false, nil
}
