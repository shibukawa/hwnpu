// Package safetensors reads Hugging Face safetensors checkpoints: a little endian uint64 header
// length, a JSON header that maps tensor names to dtype, shape and byte range, then the raw data.
package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"sort"

	"github.com/shibukawa/bonsaiai/internal/vecops"
)

// Tensor describes one tensor of the file.
type Tensor struct {
	Name  string
	DType string // "BF16", "F16", "F32", "I32", ...
	Shape []int
	begin int64
	end   int64
}

// Len is the number of elements.
func (t Tensor) Len() int {
	n := 1
	for _, d := range t.Shape {
		n *= d
	}
	return n
}

// File is an open checkpoint. Tensors are read on demand so a multi gigabyte checkpoint costs
// only the memory of the tensors in use.
type File struct {
	f        *os.File
	tensors  map[string]Tensor
	names    []string
	dataBase int64
	Metadata map[string]string // the "__metadata__" entry, if any
	mapped   []byte            // the file mapped by Map
}

type header struct {
	DType       string  `json:"dtype"`
	Shape       []int   `json:"shape"`
	DataOffsets []int64 `json:"data_offsets"`
}

// Open opens a safetensors file.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	var n uint64
	if err := binary.Read(f, binary.LittleEndian, &n); err != nil {
		f.Close()
		return nil, fmt.Errorf("safetensors: %s: %w", path, err)
	}
	if n > 100<<20 {
		f.Close()
		return nil, fmt.Errorf("safetensors: %s: header of %d bytes is not plausible", path, n)
	}
	raw := make([]byte, n)
	if _, err := io.ReadFull(f, raw); err != nil {
		f.Close()
		return nil, fmt.Errorf("safetensors: %s: %w", path, err)
	}
	var entries map[string]json.RawMessage
	if err := json.Unmarshal(raw, &entries); err != nil {
		f.Close()
		return nil, fmt.Errorf("safetensors: %s: %w", path, err)
	}
	sf := &File{f: f, tensors: map[string]Tensor{}, dataBase: int64(8 + n)}
	for name, msg := range entries {
		if name == "__metadata__" {
			json.Unmarshal(msg, &sf.Metadata)
			continue
		}
		var h header
		if err := json.Unmarshal(msg, &h); err != nil {
			f.Close()
			return nil, fmt.Errorf("safetensors: %s: %s: %w", path, name, err)
		}
		if len(h.DataOffsets) != 2 {
			f.Close()
			return nil, fmt.Errorf("safetensors: %s: %s: bad data_offsets", path, name)
		}
		sf.tensors[name] = Tensor{Name: name, DType: h.DType, Shape: h.Shape, begin: h.DataOffsets[0], end: h.DataOffsets[1]}
		sf.names = append(sf.names, name)
	}
	sort.Strings(sf.names)
	return sf, nil
}

// Close closes the file.
func (f *File) Close() error {
	f.unmap()
	return f.f.Close()
}

// ReadAt reads the file at off into p.
func (f *File) ReadAt(p []byte, off int64) (int, error) { return f.f.ReadAt(p, off) }

// Names lists the tensors in sorted order.
func (f *File) Names() []string { return append([]string(nil), f.names...) }

// Lookup returns a tensor's description.
func (f *File) Lookup(name string) (Tensor, bool) {
	t, ok := f.tensors[name]
	return t, ok
}

// Bytes reads a tensor's raw little endian data.
func (f *File) Bytes(name string) ([]byte, error) {
	t, ok := f.tensors[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: no tensor %q", name)
	}
	buf := make([]byte, t.end-t.begin)
	if _, err := f.f.ReadAt(buf, f.dataBase+t.begin); err != nil {
		return nil, fmt.Errorf("safetensors: %s: %w", name, err)
	}
	return buf, nil
}

// Float32 reads a tensor and converts BF16, F16 or F32 data to float32.
func (f *File) Float32(name string) ([]float32, error) {
	t, ok := f.tensors[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: no tensor %q", name)
	}
	return f.Float32Range(name, 0, t.Len())
}

// ScaleSuffix names the scales of a matrix stored in int8 (bonsaiai's own layout, not the
// reference's): an I8 matrix NAME of (rows, cols) and an F16 matrix NAME + ScaleSuffix of (rows,
// cols / group), each value standing for q * the scale of its row's group. Float32Range expands
// it (Gemma 4's per-layer embeddings in the tables file, docs/design/performance.md).
const ScaleSuffix = ".bonsaiai_scale"

// Float32Range reads elements [begin, end) of a tensor, in row major order, as float32.
func (f *File) Float32Range(name string, begin, end int) ([]float32, error) {
	t, ok := f.tensors[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: no tensor %q", name)
	}
	n := end - begin
	if begin < 0 || end > t.Len() || n < 0 {
		return nil, fmt.Errorf("safetensors: %s: range [%d, %d) outside %d elements", name, begin, end, t.Len())
	}
	if t.DType == "I8" {
		return f.int8Range(name, t, begin, end)
	}
	size := 2
	if t.DType == "F32" {
		size = 4
	}
	raw := make([]byte, n*size)
	if _, err := f.f.ReadAt(raw, f.dataBase+t.begin+int64(begin*size)); err != nil {
		return nil, fmt.Errorf("safetensors: %s: %w", name, err)
	}
	out := make([]float32, n)
	switch t.DType {
	case "BF16":
		if len(raw) != 2*n {
			return nil, fmt.Errorf("safetensors: %s: %d bytes for %d BF16 values", name, len(raw), n)
		}
		vecops.BF16ToF32(out, raw)
	case "F16":
		if len(raw) != 2*n {
			return nil, fmt.Errorf("safetensors: %s: %d bytes for %d F16 values", name, len(raw), n)
		}
		vecops.F16ToF32(out, raw)
	case "F32":
		if len(raw) != 4*n {
			return nil, fmt.Errorf("safetensors: %s: %d bytes for %d F32 values", name, len(raw), n)
		}
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
		}
	default:
		return nil, fmt.Errorf("safetensors: %s: cannot convert %s to float32", name, t.DType)
	}
	return out, nil
}

// int8Range is Float32Range of an I8 matrix with its scales (ScaleSuffix).
func (f *File) int8Range(name string, t Tensor, begin, end int) ([]float32, error) {
	st, ok := f.tensors[name+ScaleSuffix]
	if !ok || len(t.Shape) != 2 || len(st.Shape) != 2 || st.DType != "F16" || st.Shape[0] != t.Shape[0] ||
		st.Shape[1] == 0 || t.Shape[1]%st.Shape[1] != 0 {
		return nil, fmt.Errorf("safetensors: %s: an I8 tensor without its F16 scales %s", name, name+ScaleSuffix)
	}
	cols, groups := t.Shape[1], st.Shape[1]
	group := cols / groups
	q := make([]byte, end-begin)
	if _, err := f.f.ReadAt(q, f.dataBase+t.begin+int64(begin)); err != nil {
		return nil, fmt.Errorf("safetensors: %s: %w", name, err)
	}
	// the scales of the rows the range touches
	r0, r1 := begin/cols, (end+cols-1)/cols
	scales, err := f.Float32Range(name+ScaleSuffix, r0*groups, r1*groups)
	if err != nil {
		return nil, err
	}
	out := make([]float32, len(q))
	for i, b := range q {
		e := begin + i
		out[i] = float32(int8(b)) * scales[(e/cols-r0)*groups+e%cols/group]
	}
	return out, nil
}
