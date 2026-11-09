package llm

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"slices"

	"github.com/shibukawa/hwnpu/internal/safetensors"
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

// WriteTables writes the named tensors of the checkpoint to out, a safetensors file that
// OpenWeights opens like the checkpoint itself (the same names). A safetensors checkpoint's
// tensors are copied as stored; others' (GGUF) are written as float32.
func WriteTables(w Weights, names []string, out string) error {
	var ts []safetensors.Out
	for _, name := range names {
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
	return safetensors.Write(out, ts, map[string]string{"format": "pt", "hwnpu": "host tables", "source": w.Source()})
}
