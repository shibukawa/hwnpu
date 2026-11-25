package llm

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"slices"

	"github.com/shibukawa/bonsaiai/internal/safetensors"
	"github.com/shibukawa/bonsaiai/internal/vecops"
)

// HostTables names the tables whose rows the host looks up while it runs m (llm.TokenRow): the
// token embedding and the tables of the model's TokenInputs (Gemma 4's per-layer embeddings).
// With them written apart (WriteTables), running a converted model does not need the checkpoint.
func HostTables(m Model) []string {
	var names []string
	if e := m.Embedding(); e != "" {
		names = append(names, e)
	}
	if l, ok := m.(TokenInputs); ok {
		for _, lk := range l.TokenInputs() {
			names = append(names, lk.Table)
		}
	}
	slices.Sort(names)
	return slices.Compact(names)
}

// rawWeights is a checkpoint that gives a tensor's bytes as stored, for copying without
// converting (a bf16 table stays bf16, half the size of float32).
type rawWeights interface {
	section(name string) (*io.SectionReader, safetensors.Tensor, error)
}

func (w stWeights) section(name string) (*io.SectionReader, safetensors.Tensor, error) {
	return w.f.Section(name)
}

func (w *shardedWeights) section(name string) (*io.SectionReader, safetensors.Tensor, error) {
	f, ok := w.files[name]
	if !ok {
		return nil, safetensors.Tensor{}, fmt.Errorf("%s: %s has no tensor %s", w.arch.Name, w.path, name)
	}
	return f.Section(name)
}

// HostTableGroups is, by table, the Group of the model's TokenInputs that have one: the tables
// WriteTables stores in int8.
func HostTableGroups(m Model) map[string]int {
	l, ok := m.(TokenInputs)
	if !ok {
		return nil
	}
	var g map[string]int
	for _, lk := range l.TokenInputs() {
		if lk.Group > 0 {
			if g == nil {
				g = map[string]int{}
			}
			g[lk.Table] = lk.Group
		}
	}
	return g
}

// WriteTables writes the named tensors of the checkpoint to out, a safetensors file that
// OpenWeights opens like the checkpoint itself (the same names). A safetensors checkpoint's
// tensors are copied as stored; others' (GGUF) are written as float32. A matrix that groups
// names is stored in int8 with an fp16 scale every groups[name] values of a row (the scales a
// matrix of its own, safetensors.ScaleSuffix), which Rows expands.
func WriteTables(w Weights, names []string, groups map[string]int, out string) error {
	var ts []safetensors.Out
	for _, name := range names {
		if g := groups[name]; g > 0 {
			q, s, err := int8Table(w, name, g)
			if err != nil {
				return err
			}
			ts = append(ts, q, s)
			continue
		}
		if t, ok := w.(interface {
			Table(string) ([]float32, []int, error)
		}); ok {
			// a rotated checkpoint's table (Rotate): its values in fp16
			v, shape, err := t.Table(name)
			if err != nil {
				return err
			}
			b := make([]byte, 2*len(v))
			vecops.F32ToF16(b, v)
			ts = append(ts, safetensors.Out{Name: name, DType: "F16", Shape: shape, Size: int64(len(b)), Data: bytes.NewReader(b)})
			continue
		}
		if r, ok := w.(rawWeights); ok {
			sec, t, err := r.section(name)
			if err != nil {
				return err
			}
			ts = append(ts, safetensors.Out{Name: name, DType: t.DType, Shape: t.Shape, Size: sec.Size(), Data: sec})
			continue
		}
		v, shape, err := w.Get(name)
		if err != nil {
			return err
		}
		b := make([]byte, 4*len(v))
		for i, x := range v {
			binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
		}
		ts = append(ts, safetensors.Out{Name: name, DType: "F32", Shape: shape, Size: int64(len(b)), Data: bytes.NewReader(b)})
	}
	return safetensors.Write(out, ts, map[string]string{"format": "pt", "bonsaiai": "host tables", "source": w.Source()})
}

// int8Table is the tensors WriteTables stores matrix name in, a scale every group values: the
// int8 values and their scales, each computed from the checkpoint's rows as it is written.
func int8Table(w Weights, name string, group int) (q, s safetensors.Out, err error) {
	var shape []int
	if sw, ok := w.(interface{ Shape(string) ([]int, bool) }); ok {
		shape, _ = sw.Shape(name)
	} else if _, sh, err := w.Get(name); err == nil {
		shape = sh
	}
	if len(shape) != 2 || group <= 0 || shape[1]%group != 0 {
		return q, s, fmt.Errorf("llm: %s: %v is no matrix of rows in groups of %d", name, shape, group)
	}
	rows, cols := shape[0], shape[1]
	q = safetensors.Out{Name: name, DType: "I8", Shape: []int{rows, cols}, Size: int64(rows) * int64(cols),
		Data: &int8Reader{w: w, name: name, rows: rows, cols: cols, group: group}}
	s = safetensors.Out{Name: name + safetensors.ScaleSuffix, DType: "F16", Shape: []int{rows, cols / group},
		Size: int64(rows) * int64(cols/group) * 2,
		Data: &int8Reader{w: w, name: name, rows: rows, cols: cols, group: group, scales: true}}
	return q, s, nil
}

// int8Reader gives the int8 values (or, scales, their fp16 scales) of a matrix's rows, a chunk
// of rows at a time.
type int8Reader struct {
	w                 Weights
	name              string
	rows, cols, group int
	scales            bool
	next              int // the next row to read
	buf               []byte
}

func (r *int8Reader) Read(p []byte) (int, error) {
	for len(r.buf) == 0 {
		if r.next >= r.rows {
			return 0, io.EOF
		}
		end := min(r.next+1024, r.rows)
		x, err := r.w.Rows(r.name, r.next, end)
		if err != nil {
			return 0, err
		}
		q, s := QuantizeInt8(x, r.group)
		if r.scales {
			r.buf = make([]byte, 2*len(s))
			for i, h := range s {
				binary.LittleEndian.PutUint16(r.buf[2*i:], h)
			}
		} else {
			r.buf = make([]byte, len(q))
			for i, v := range q {
				r.buf[i] = byte(v)
			}
		}
		r.next = end
	}
	n := copy(p, r.buf)
	r.buf = r.buf[n:]
	return n, nil
}

// QuantizeInt8 rounds x in groups of group values to int8, symmetric, with an fp16 scale a
// group (its largest magnitude / 127), as tools/amd/iron_hostple.ple_int8 does.
func QuantizeInt8(x []float32, group int) ([]int8, []uint16) {
	q := make([]int8, len(x))
	s := make([]uint16, len(x)/group)
	for g := range s {
		part := x[g*group : (g+1)*group]
		m := float32(0)
		for _, v := range part {
			m = max(m, float32(math.Abs(float64(v))))
		}
		s[g] = vecops.F16Bits(m / 127)
		d := vecops.F16Value(s[g])
		if d == 0 {
			continue
		}
		for i, v := range part {
			q[g*group+i] = int8(max(-127, min(127, math.RoundToEven(float64(v/d)))))
		}
	}
	return q, s
}
