//go:build !unix

package safetensors

// Map reads a tensor's raw data (no mapping here).
func (f *File) Map(name string) ([]byte, error) { return f.Bytes(name) }

func (f *File) unmap() {}
