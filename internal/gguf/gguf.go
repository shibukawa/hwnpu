// Package gguf reads GGUF checkpoints (llama.cpp) and dequantizes the block formats hwnpu
// converts from: F32, F16, BF16, Q4_0, Q8_0 and Q6_K.
//
// Dimensions are stored innermost first, so a tensor listed as [2048, 128000] is a row major
// matrix of 128000 rows by 2048 columns. Shape returns the row major order.
package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
)

// Type is the ggml tensor type.
type Type uint32

const (
	F32  Type = 0
	F16  Type = 1
	Q4_0 Type = 2
	Q4_1 Type = 3
	Q8_0 Type = 8
	Q4_K Type = 12
	Q5_K Type = 13
	Q6_K Type = 14
	BF16 Type = 30
)

func (t Type) String() string {
	switch t {
	case F32:
		return "F32"
	case F16:
		return "F16"
	case BF16:
		return "BF16"
	case Q4_0:
		return "Q4_0"
	case Q4_1:
		return "Q4_1"
	case Q8_0:
		return "Q8_0"
	case Q4_K:
		return "Q4_K"
	case Q5_K:
		return "Q5_K"
	case Q6_K:
		return "Q6_K"
	}
	return fmt.Sprintf("type%d", uint32(t))
}

// blockOf returns the elements per block and the bytes per block.
func blockOf(t Type) (elems, bytes int, ok bool) {
	switch t {
	case F32:
		return 1, 4, true
	case F16, BF16:
		return 1, 2, true
	case Q4_0:
		return 32, 18, true
	case Q8_0:
		return 32, 34, true
	case Q6_K:
		return 256, 210, true
	}
	return 0, 0, false
}

// Tensor describes one tensor.
type Tensor struct {
	Name   string
	Shape  []int // row major (outermost first)
	Type   Type
	offset uint64 // from the start of the data section
}

// Len is the number of elements.
func (t Tensor) Len() int {
	n := 1
	for _, d := range t.Shape {
		n *= d
	}
	return n
}

// File is an open GGUF file.
type File struct {
	f        *os.File
	Version  uint32
	KV       map[string]any // metadata; arrays of strings become []string, numeric arrays []float64
	tensors  map[string]Tensor
	names    []string
	dataBase int64
}

type reader struct {
	r   io.Reader
	err error
}

func (r *reader) u32() uint32 {
	var v uint32
	if r.err == nil {
		r.err = binary.Read(r.r, binary.LittleEndian, &v)
	}
	return v
}

func (r *reader) u64() uint64 {
	var v uint64
	if r.err == nil {
		r.err = binary.Read(r.r, binary.LittleEndian, &v)
	}
	return v
}

func (r *reader) str() string {
	n := r.u64()
	if r.err != nil {
		return ""
	}
	if n > 64<<20 {
		r.err = fmt.Errorf("string of %d bytes is not plausible", n)
		return ""
	}
	b := make([]byte, n)
	_, r.err = io.ReadFull(r.r, b)
	return string(b)
}

// value reads one metadata value of the given type.
func (r *reader) value(t uint32) any {
	switch t {
	case 0: // uint8
		var v uint8
		r.err = binary.Read(r.r, binary.LittleEndian, &v)
		return float64(v)
	case 1:
		var v int8
		r.err = binary.Read(r.r, binary.LittleEndian, &v)
		return float64(v)
	case 2:
		var v uint16
		r.err = binary.Read(r.r, binary.LittleEndian, &v)
		return float64(v)
	case 3:
		var v int16
		r.err = binary.Read(r.r, binary.LittleEndian, &v)
		return float64(v)
	case 4:
		return float64(r.u32())
	case 5:
		return float64(int32(r.u32()))
	case 6:
		return float64(math.Float32frombits(r.u32()))
	case 7:
		var v uint8
		r.err = binary.Read(r.r, binary.LittleEndian, &v)
		return v != 0
	case 8:
		return r.str()
	case 9:
		et := r.u32()
		n := r.u64()
		if r.err != nil {
			return nil
		}
		if et == 8 {
			out := make([]string, 0, min(n, 1<<20))
			for i := uint64(0); i < n && r.err == nil; i++ {
				out = append(out, r.str())
			}
			return out
		}
		out := make([]float64, 0, min(n, 1<<20))
		for i := uint64(0); i < n && r.err == nil; i++ {
			v, _ := r.value(et).(float64)
			out = append(out, v)
		}
		return out
	case 10:
		return float64(r.u64())
	case 11:
		return float64(int64(r.u64()))
	case 12:
		return math.Float64frombits(r.u64())
	}
	r.err = fmt.Errorf("unknown metadata type %d", t)
	return nil
}

// Open opens a GGUF file and reads its metadata and tensor table.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	g := &File{f: f, KV: map[string]any{}, tensors: map[string]Tensor{}}
	if err := g.readHeader(); err != nil {
		f.Close()
		return nil, fmt.Errorf("gguf: %s: %w", path, err)
	}
	return g, nil
}

func (g *File) readHeader() error {
	br := &reader{r: &countingReader{r: g.f}}
	magic := make([]byte, 4)
	if _, err := io.ReadFull(br.r, magic); err != nil {
		return err
	}
	if string(magic) != "GGUF" {
		return fmt.Errorf("not a GGUF file")
	}
	g.Version = br.u32()
	if g.Version < 2 {
		return fmt.Errorf("GGUF version %d is not supported", g.Version)
	}
	nTensors := br.u64()
	nKV := br.u64()
	for i := uint64(0); i < nKV && br.err == nil; i++ {
		k := br.str()
		t := br.u32()
		g.KV[k] = br.value(t)
	}
	if br.err != nil {
		return br.err
	}
	for i := uint64(0); i < nTensors && br.err == nil; i++ {
		name := br.str()
		nd := br.u32()
		if nd > 8 {
			return fmt.Errorf("%s: %d dimensions", name, nd)
		}
		dims := make([]int, nd)
		for j := range dims {
			dims[j] = int(br.u64())
		}
		t := Type(br.u32())
		off := br.u64()
		// innermost first in the file; row major outermost first for callers
		shape := make([]int, nd)
		for j := range dims {
			shape[j] = dims[int(nd)-1-j]
		}
		g.tensors[name] = Tensor{Name: name, Shape: shape, Type: t, offset: off}
		g.names = append(g.names, name)
	}
	if br.err != nil {
		return br.err
	}
	align := int64(32)
	if a, ok := g.KV["general.alignment"].(float64); ok && a > 0 {
		align = int64(a)
	}
	pos := br.r.(*countingReader).n
	g.dataBase = (pos + align - 1) / align * align
	sort.Strings(g.names)
	return nil
}

type countingReader struct {
	r io.Reader
	n int64
}

func (c *countingReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n += int64(n)
	return n, err
}

// Close closes the file.
func (g *File) Close() error { return g.f.Close() }

// Names lists the tensors in sorted order.
func (g *File) Names() []string { return append([]string(nil), g.names...) }

// Lookup returns a tensor's description.
func (g *File) Lookup(name string) (Tensor, bool) {
	t, ok := g.tensors[name]
	return t, ok
}

// String returns a string metadata value.
func (g *File) String(key string) string {
	s, _ := g.KV[key].(string)
	return s
}

// Number returns a numeric metadata value.
func (g *File) Number(key string) (float64, bool) {
	v, ok := g.KV[key].(float64)
	return v, ok
}

// Bytes reads a tensor's raw quantized data.
func (g *File) Bytes(name string) ([]byte, error) {
	t, ok := g.tensors[name]
	if !ok {
		return nil, fmt.Errorf("gguf: no tensor %q", name)
	}
	elems, bytes, ok := blockOf(t.Type)
	if !ok {
		return nil, fmt.Errorf("gguf: %s: type %s is not supported", name, t.Type)
	}
	n := t.Len()
	if n%elems != 0 {
		return nil, fmt.Errorf("gguf: %s: %d elements is not a multiple of the %d element block", name, n, elems)
	}
	buf := make([]byte, n/elems*bytes)
	if _, err := g.f.ReadAt(buf, g.dataBase+int64(t.offset)); err != nil {
		return nil, fmt.Errorf("gguf: %s: %w", name, err)
	}
	return buf, nil
}

// Float32 reads a tensor and dequantizes it to float32 in row major order.
func (g *File) Float32(name string) ([]float32, error) {
	t, ok := g.tensors[name]
	if !ok {
		return nil, fmt.Errorf("gguf: no tensor %q", name)
	}
	return g.Float32Range(name, 0, t.Len())
}

// Float32Range reads and dequantizes elements [begin, end) of a tensor, in row major order. The
// range must start and end on block boundaries of the tensor's type.
func (g *File) Float32Range(name string, begin, end int) ([]float32, error) {
	t, ok := g.tensors[name]
	if !ok {
		return nil, fmt.Errorf("gguf: no tensor %q", name)
	}
	elems, bytes, ok := blockOf(t.Type)
	if !ok {
		return nil, fmt.Errorf("gguf: %s: type %s is not supported", name, t.Type)
	}
	if begin < 0 || end > t.Len() || end < begin || begin%elems != 0 || end%elems != 0 {
		return nil, fmt.Errorf("gguf: %s: range [%d, %d) is not block aligned within %d elements", name, begin, end, t.Len())
	}
	raw := make([]byte, (end-begin)/elems*bytes)
	if _, err := g.f.ReadAt(raw, g.dataBase+int64(t.offset)+int64(begin/elems*bytes)); err != nil {
		return nil, fmt.Errorf("gguf: %s: %w", name, err)
	}
	out := make([]float32, end-begin)
	switch t.Type {
	case F32:
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
		}
	case F16:
		for i := range out {
			out[i] = f16to32(binary.LittleEndian.Uint16(raw[2*i:]))
		}
	case BF16:
		for i := range out {
			out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[2*i:])) << 16)
		}
	case Q4_0:
		DequantQ4_0(raw, out)
	case Q8_0:
		dequantQ8_0(raw, out)
	case Q6_K:
		dequantQ6_K(raw, out)
	default:
		return nil, fmt.Errorf("gguf: %s: type %s is not supported", name, t.Type)
	}
	return out, nil
}

// Q4_0Block is one block of Q4_0: 32 values as (q - 8) * d, q being 4 bit unsigned.
type Q4_0Block struct {
	D float32
	Q [32]uint8 // 0..15
}

// ReadQ4_0 decodes the blocks of a Q4_0 tensor without dequantizing, so that the 4 bit values
// can be repacked losslessly. Element j of a block is the low nibble of byte j for j < 16 and the
// high nibble of byte j-16 otherwise.
func ReadQ4_0(raw []byte) []Q4_0Block {
	blocks := make([]Q4_0Block, len(raw)/18)
	for i := range blocks {
		b := raw[18*i : 18*i+18]
		blocks[i].D = f16to32(binary.LittleEndian.Uint16(b))
		for j := 0; j < 16; j++ {
			blocks[i].Q[j] = b[2+j] & 0xf
			blocks[i].Q[j+16] = b[2+j] >> 4
		}
	}
	return blocks
}

// DequantQ4_0 dequantizes Q4_0 blocks into out.
func DequantQ4_0(raw []byte, out []float32) {
	for i := 0; i < len(raw)/18; i++ {
		b := raw[18*i : 18*i+18]
		d := f16to32(binary.LittleEndian.Uint16(b))
		y := out[32*i : 32*i+32]
		for j := 0; j < 16; j++ {
			y[j] = (float32(b[2+j]&0xf) - 8) * d
			y[j+16] = (float32(b[2+j]>>4) - 8) * d
		}
	}
}

func dequantQ8_0(raw []byte, out []float32) {
	for i := 0; i < len(raw)/34; i++ {
		b := raw[34*i : 34*i+34]
		d := f16to32(binary.LittleEndian.Uint16(b))
		y := out[32*i : 32*i+32]
		for j := 0; j < 32; j++ {
			y[j] = float32(int8(b[2+j])) * d
		}
	}
}

// dequantQ6_K follows dequantize_row_q6_K in ggml-quants.c: a block of 256 values holds the low
// 4 bits in ql[128], the high 2 bits in qh[64], 16 int8 scales and an fp16 scale d.
func dequantQ6_K(raw []byte, out []float32) {
	for i := 0; i < len(raw)/210; i++ {
		b := raw[210*i : 210*i+210]
		ql := b[0:128]
		qh := b[128:192]
		sc := b[192:208]
		d := f16to32(binary.LittleEndian.Uint16(b[208:]))
		y := out[256*i : 256*i+256]
		for n := 0; n < 256; n += 128 {
			for l := 0; l < 32; l++ {
				is := l / 16
				q1 := int8((ql[l]&0xf)|((qh[l]>>0)&3)<<4) - 32
				q2 := int8((ql[l+32]&0xf)|((qh[l]>>2)&3)<<4) - 32
				q3 := int8((ql[l]>>4)|((qh[l]>>4)&3)<<4) - 32
				q4 := int8((ql[l+32]>>4)|((qh[l]>>6)&3)<<4) - 32
				y[l] = d * float32(int8(sc[is])) * float32(q1)
				y[l+32] = d * float32(int8(sc[is+2])) * float32(q2)
				y[l+64] = d * float32(int8(sc[is+4])) * float32(q3)
				y[l+96] = d * float32(int8(sc[is+6])) * float32(q4)
			}
			y = y[128:]
			ql = ql[64:]
			qh = qh[32:]
			sc = sc[8:]
		}
	}
}

func f16to32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h & 0x3ff)
	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		e := uint32(127 - 15 + 1)
		for mant&0x400 == 0 {
			mant <<= 1
			e--
		}
		mant &= 0x3ff
		return math.Float32frombits(sign | e<<23 | mant<<13)
	case 0x1f:
		return math.Float32frombits(sign | 0xff<<23 | mant<<13)
	}
	return math.Float32frombits(sign | (exp+127-15)<<23 | mant<<13)
}
