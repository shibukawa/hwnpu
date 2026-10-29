// Package check compares a converted model's activations with a reference written by
// tools/ref/ref_logits.py from the Hugging Face implementation.
package check

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"math"
	"os"
)

// Ref is a reference run: the token ids, the hidden states and the logits, all fp32.
type Ref struct {
	IDs []int32
	// Hidden holds (layers + 1) x seq x hidden values: the token embeddings (the rows of the
	// embedding table, before anything else the model adds), then the output of each layer
	// (before a final norm).
	Hidden []float32
	// Logits holds the model's output, seq x width: logits, or an encoder's last hidden state.
	Logits []float32
	// Meta is the JSON sidecar, if there is one.
	Meta Meta
}

// Meta describes a reference run (PREFIX.json).
type Meta struct {
	ModelType  string `json:"model_type"`
	Seq        int    `json:"seq"`
	HiddenSize int    `json:"hidden_size"`
	NumLayers  int    `json:"num_layers"`
	VocabSize  int    `json:"vocab_size"`
	Text       string `json:"text"`
	DType      string `json:"dtype"`
	// Output is "logits" (causal language models) or "hidden" (encoders).
	Output string `json:"output"`
}

// LoadRef reads PREFIX.ids.bin, PREFIX.hidden.bin, PREFIX.logits.bin (little endian, row major)
// and PREFIX.json if present.
func LoadRef(prefix string) (*Ref, error) {
	r := &Ref{}
	raw, err := os.ReadFile(prefix + ".ids.bin")
	if err != nil {
		return nil, err
	}
	r.IDs = make([]int32, len(raw)/4)
	for i := range r.IDs {
		r.IDs[i] = int32(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	if r.Hidden, err = readF32(prefix + ".hidden.bin"); err != nil {
		return nil, err
	}
	if r.Logits, err = readF32(prefix + ".logits.bin"); err != nil {
		return nil, err
	}
	js, err := os.ReadFile(prefix + ".json")
	switch {
	case err == nil:
		if err := json.Unmarshal(js, &r.Meta); err != nil {
			return nil, fmt.Errorf("%s.json: %w", prefix, err)
		}
	case !errors.Is(err, fs.ErrNotExist):
		return nil, err
	}
	return r, nil
}

func readF32(path string) ([]float32, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	return out, nil
}

// Validate checks the sizes against a model's dimensions; width is the output's per token.
func (r *Ref) Validate(layers, hidden, width int) error {
	seq := len(r.IDs)
	if seq == 0 {
		return fmt.Errorf("reference has no tokens")
	}
	if len(r.Hidden) != (layers+1)*seq*hidden {
		return fmt.Errorf("reference hidden states hold %d values, expected (%d layers + 1) x %d tokens x %d", len(r.Hidden), layers, seq, hidden)
	}
	if len(r.Logits) != seq*width {
		return fmt.Errorf("reference outputs hold %d values, expected %d tokens x %d", len(r.Logits), seq, width)
	}
	return nil
}

// Layer returns the hidden state entering layer i (i = 0 is the token embeddings, i = layers the
// output of the last layer).
func (r *Ref) Layer(i, hidden int) []float32 {
	n := len(r.IDs) * hidden
	return r.Hidden[i*n : (i+1)*n]
}

// Stats says how close a result is to the reference.
type Stats struct {
	// RelRMS is the RMS of the difference over the RMS of the reference.
	RelRMS float64
	Cosine float64
	MaxAbs float64
	MaxRef float64
}

func (s Stats) String() string {
	return fmt.Sprintf("rel RMS err %.4f, cosine %.6f, max |diff| %.3g (max |ref| %.3g)", s.RelRMS, s.Cosine, s.MaxAbs, s.MaxRef)
}

// Compare compares got with ref element by element.
func Compare(got, ref []float32) (Stats, error) {
	if len(got) != len(ref) {
		return Stats{}, fmt.Errorf("%d values, reference %d", len(got), len(ref))
	}
	var se, gg, rr, dot, maxAbs, maxRef float64
	for i := range got {
		g, r := float64(got[i]), float64(ref[i])
		se += (g - r) * (g - r)
		gg += g * g
		rr += r * r
		dot += g * r
		maxAbs = math.Max(maxAbs, math.Abs(g-r))
		maxRef = math.Max(maxRef, math.Abs(r))
	}
	return Stats{RelRMS: math.Sqrt(se / rr), Cosine: dot / math.Sqrt(gg*rr), MaxAbs: maxAbs, MaxRef: maxRef}, nil
}

// LogitStats adds what matters for next token prediction.
type LogitStats struct {
	Stats
	// Agree counts the positions whose argmax matches the reference, out of N.
	Agree, N int
	// KL is the mean KL divergence of the reference distribution from the model's, in nats.
	KL float64
}

func (s LogitStats) String() string {
	return fmt.Sprintf("%v; argmax agreement %d/%d, mean KL(ref || model) %.4f nats", s.Stats, s.Agree, s.N, s.KL)
}

// CompareLogits compares seq x vocab logits.
func CompareLogits(got, ref []float32, vocab int) (LogitStats, error) {
	st, err := Compare(got, ref)
	if err != nil {
		return LogitStats{}, err
	}
	s := LogitStats{Stats: st, N: len(ref) / vocab}
	for i := 0; i < s.N; i++ {
		g, r := got[i*vocab:(i+1)*vocab], ref[i*vocab:(i+1)*vocab]
		if argmax(g) == argmax(r) {
			s.Agree++
		}
		s.KL += KL(r, g)
	}
	s.KL /= float64(s.N)
	return s, nil
}

func argmax(v []float32) int {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return best
}

// KL is KL(p || q) of the softmax distributions of two logit vectors.
func KL(p, q []float32) float64 {
	lp, lq := logSoftmax(p), logSoftmax(q)
	var kl float64
	for i := range lp {
		kl += math.Exp(lp[i]) * (lp[i] - lq[i])
	}
	return kl
}

func logSoftmax(v []float32) []float64 {
	m := float64(v[argmax(v)])
	var z float64
	for _, x := range v {
		z += math.Exp(float64(x) - m)
	}
	lz := math.Log(z) + m
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = float64(x) - lz
	}
	return out
}
