// Package tflite reads TensorFlow Lite models (FlatBuffers, schema.fbs) without dependencies and
// converts their graphs to internal/onnx's, whose host evaluator and MIL lowering then run them.
package tflite

import (
	"encoding/binary"
	"math"
)

// table is a FlatBuffers table: the buffer and the table's position.
type table struct {
	b   []byte
	pos int
}

func u32(b []byte, p int) int { return int(binary.LittleEndian.Uint32(b[p:])) }

// field returns the position of field i's value, or -1 when it is absent.
func (t table) field(i int) int {
	vt := t.pos - int(int32(binary.LittleEndian.Uint32(t.b[t.pos:])))
	vsize := int(binary.LittleEndian.Uint16(t.b[vt:]))
	off := 4 + 2*i
	if off >= vsize {
		return -1
	}
	o := int(binary.LittleEndian.Uint16(t.b[vt+off:]))
	if o == 0 {
		return -1
	}
	return t.pos + o
}

func (t table) uint8(i int, def int) int {
	p := t.field(i)
	if p < 0 {
		return def
	}
	return int(t.b[p])
}

func (t table) int32(i int, def int) int {
	p := t.field(i)
	if p < 0 {
		return def
	}
	return int(int32(binary.LittleEndian.Uint32(t.b[p:])))
}

func (t table) uint64(i int, def uint64) uint64 {
	p := t.field(i)
	if p < 0 {
		return def
	}
	return binary.LittleEndian.Uint64(t.b[p:])
}

func (t table) float32(i int, def float32) float32 {
	p := t.field(i)
	if p < 0 {
		return def
	}
	return math.Float32frombits(binary.LittleEndian.Uint32(t.b[p:]))
}

func (t table) bool(i int) bool { return t.uint8(i, 0) != 0 }

// vector returns the position of a vector's first element and its length.
func (t table) vector(i int) (int, int) {
	p := t.field(i)
	if p < 0 {
		return 0, 0
	}
	v := p + u32(t.b, p)
	return v + 4, u32(t.b, v)
}

func (t table) string(i int) string {
	p, n := t.vector(i)
	return string(t.b[p : p+n])
}

func (t table) bytes(i int) []byte {
	p, n := t.vector(i)
	return t.b[p : p+n]
}

func (t table) ints(i int) []int {
	p, n := t.vector(i)
	out := make([]int, n)
	for k := range out {
		out[k] = int(int32(binary.LittleEndian.Uint32(t.b[p+4*k:])))
	}
	return out
}

func (t table) tables(i int) []table {
	p, n := t.vector(i)
	out := make([]table, n)
	for k := range out {
		e := p + 4*k
		out[k] = table{t.b, e + u32(t.b, e)}
	}
	return out
}

// sub returns the table of field i, or false.
func (t table) sub(i int) (table, bool) {
	p := t.field(i)
	if p < 0 {
		return table{}, false
	}
	return table{t.b, p + u32(t.b, p)}, true
}
