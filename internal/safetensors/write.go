package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// Section is a reader of a tensor's raw little endian data, for copying it without converting.
func (f *File) Section(name string) (*io.SectionReader, Tensor, error) {
	t, ok := f.tensors[name]
	if !ok {
		return nil, Tensor{}, fmt.Errorf("safetensors: no tensor %q", name)
	}
	return io.NewSectionReader(f.f, f.dataBase+t.begin, t.end-t.begin), t, nil
}

// Out is a tensor to write: Size bytes of raw little endian data read from Data.
type Out struct {
	Name  string
	DType string
	Shape []int
	Size  int64
	Data  io.Reader
}

// Write writes the tensors to path as one safetensors file, in their order, with metadata (nil
// for none). It writes path.tmp and renames it, so that a file cut off is never taken for whole.
func Write(path string, ts []Out, metadata map[string]string) error {
	hdr := map[string]any{}
	if len(metadata) > 0 {
		hdr["__metadata__"] = metadata
	}
	var off int64
	for _, t := range ts {
		if _, dup := hdr[t.Name]; dup || t.Name == "__metadata__" {
			return fmt.Errorf("safetensors: %s: the tensor %q twice", path, t.Name)
		}
		hdr[t.Name] = header{DType: t.DType, Shape: t.Shape, DataOffsets: []int64{off, off + t.Size}}
		off += t.Size
	}
	raw, err := json.Marshal(hdr)
	if err != nil {
		return err
	}
	// the data starts 8-byte aligned, as the reference writer pads the header with spaces
	for (8+len(raw))%8 != 0 {
		raw = append(raw, ' ')
	}
	tmp := path + ".tmp"
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	err = func() error {
		if err := binary.Write(f, binary.LittleEndian, uint64(len(raw))); err != nil {
			return err
		}
		if _, err := f.Write(raw); err != nil {
			return err
		}
		for _, t := range ts {
			n, err := io.Copy(f, io.LimitReader(t.Data, t.Size))
			if err != nil {
				return fmt.Errorf("%s: %w", t.Name, err)
			}
			if n != t.Size {
				return fmt.Errorf("%s: %d bytes of %d", t.Name, n, t.Size)
			}
		}
		return f.Close()
	}()
	if err != nil {
		f.Close()
		os.Remove(tmp)
		return fmt.Errorf("safetensors: %s: %w", path, err)
	}
	return os.Rename(tmp, path)
}
