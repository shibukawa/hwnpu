package experts

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"runtime"
	"sort"
	"strconv"
	"sync"

	"github.com/shibukawa/bonsaiai/internal/safetensors"
	"github.com/shibukawa/bonsaiai/internal/vecops"
)

// Routing is how a layer's router picks and weighs its experts from its logits.
type Routing struct {
	// Score turns the logits into the experts' scores: "sigmoid" (LFM2-MoE) or "softmax".
	Score string `json:"score"`
	// Bias is that a bias a layer (Names.Bias) is added to the scores to choose the experts,
	// not to weigh them (LFM2-MoE's expert_bias).
	Bias bool `json:"bias,omitempty"`
	// Norm divides the chosen experts' scores by their sum (norm_topk_prob).
	Norm bool `json:"norm,omitempty"`
	// Scale multiplies the weights (routed_scaling_factor); 0 for 1.
	Scale float64 `json:"scale,omitempty"`
}

// Choose picks the k experts of the largest scores from a row's router logits, writing their
// numbers into idx and their weights into w (both k long). bias is the layer's, nil for none.
func (r Routing) Choose(logits, bias []float32, idx []int, w []float32) {
	n, k := len(logits), len(idx)
	score := make([]float32, n)
	switch r.Score {
	case "softmax":
		m := float32(math.Inf(-1))
		for _, v := range logits {
			m = max(m, v)
		}
		var sum float32
		for e, v := range logits {
			score[e] = float32(math.Exp(float64(v - m)))
			sum += score[e]
		}
		for e := range score {
			score[e] /= sum
		}
	default:
		for e, v := range logits {
			score[e] = 1 / (1 + float32(math.Exp(float64(-v))))
		}
	}
	order := make([]int, n)
	for e := range order {
		order[e] = e
	}
	key := score
	if bias != nil {
		key = make([]float32, n)
		for e := range key {
			key[e] = score[e] + bias[e]
		}
	}
	sort.SliceStable(order, func(a, b int) bool { return key[order[a]] > key[order[b]] })
	var sum float32
	for j := 0; j < k; j++ {
		idx[j] = order[j]
		w[j] = score[order[j]]
		sum += w[j]
	}
	for j := range w {
		if r.Norm {
			w[j] /= sum + 1e-6
		}
		if r.Scale != 0 {
			w[j] *= float32(r.Scale)
		}
	}
}

// Names are the checkpoint's tensors of a layer's experts.
type Names struct {
	// Gate, Up and Down name expert e's projections: Gate and Up (inter, hidden), Down (hidden,
	// inter), as Hugging Face's linear layers hold them.
	Gate, Up, Down func(e int) string
	// Bias is the layer's bias of the choice (Routing.Bias), "" for none.
	Bias string
}

// The tensors of a layer in the weights file (WriteFile).
func GateUpName(layer, e int) string { return fmt.Sprintf("%d.%d.gu", layer, e) }
func DownName(layer, e int) string   { return fmt.Sprintf("%d.%d.dn", layer, e) }
func ScalesName(layer int) string    { return fmt.Sprintf("%d.scales", layer) }
func BiasName(layer int) string      { return fmt.Sprintf("%d.bias", layer) }

// WriteFile writes the experts of layers into one safetensors file at path: for each layer and
// expert e, GateUpName (hidden, 2 inter) and DownName (hidden, inter) in int8 (I8), a scale a row
// of the projection (its absolute maximum over 127, in fp16), the scales of the layer in
// ScalesName (experts, ScaleRow) and with Routing.Bias the bias in BiasName (F32). get reads a
// checkpoint tensor as float32. A layer's experts are quantized in parallel as the file reaches
// it, so that one layer of them is in memory.
func WriteFile(path string, sh Shape, layers []int, names func(layer int) Names, get func(name string) ([]float32, error)) error {
	H, I, E := sh.Hidden, sh.Inter, sh.Experts
	S := sh.ScaleRow()
	type layerData struct {
		once   sync.Once
		gu, dn [][]byte
		scales []byte
		bias   []byte
		err    error
	}
	data := map[int]*layerData{}
	var current *layerData // the layer quantized last: one is held at a time
	load := func(l int) (*layerData, error) {
		d := data[l]
		d.once.Do(func() {
			if current != nil && current != d {
				current.gu, current.dn = nil, nil
			}
			current = d
			n := names(l)
			d.gu, d.dn = make([][]byte, E), make([][]byte, E)
			sc := make([]float32, E*S)
			var mu sync.Mutex
			var wg sync.WaitGroup
			sem := make(chan struct{}, runtime.NumCPU())
			for e := 0; e < E; e++ {
				wg.Add(1)
				sem <- struct{}{}
				go func(e int) {
					defer func() { <-sem; wg.Done() }()
					fail := func(err error) {
						mu.Lock()
						if d.err == nil {
							d.err = err
						}
						mu.Unlock()
					}
					gate, err := get(n.Gate(e))
					if err != nil {
						fail(err)
						return
					}
					up, err := get(n.Up(e))
					if err != nil {
						fail(err)
						return
					}
					down, err := get(n.Down(e))
					if err != nil {
						fail(err)
						return
					}
					if len(gate) != I*H || len(up) != I*H || len(down) != H*I {
						fail(fmt.Errorf("experts: layer %d expert %d: projections of %d, %d and %d values, want %d", l, e, len(gate), len(up), len(down), H*I))
						return
					}
					row := sc[e*S : (e+1)*S]
					q := make([]byte, 2*I*H) // the gate's rows then the up's, (2 inter, hidden)
					quantize(gate, I, H, q[:I*H], row[:I])
					quantize(up, I, H, q[I*H:], row[I:2*I])
					gu := make([]byte, H*2*I) // transposed: (hidden, 2 inter)
					for j := 0; j < 2*I; j++ {
						src := q[j*H : (j+1)*H]
						for h, v := range src {
							gu[h*2*I+j] = v
						}
					}
					dn := make([]byte, H*I)
					quantize(down, H, I, dn, row[2*I:])
					d.gu[e], d.dn[e] = gu, dn
				}(e)
			}
			wg.Wait()
			if d.err != nil {
				return
			}
			d.scales = make([]byte, 2*len(sc))
			vecops.F32ToF16(d.scales, sc)
			if n.Bias != "" {
				b, err := get(n.Bias)
				if err != nil {
					d.err = err
					return
				}
				if len(b) != E {
					d.err = fmt.Errorf("experts: layer %d: a bias of %d values for %d experts", l, len(b), E)
					return
				}
				d.bias = make([]byte, 4*E)
				for e, v := range b {
					binary.LittleEndian.PutUint32(d.bias[4*e:], math.Float32bits(v))
				}
			}
		})
		return d, d.err
	}
	lazy := func(l int, pick func(d *layerData) []byte) io.Reader {
		return &lazyReader{get: func() ([]byte, error) {
			d, err := load(l)
			if err != nil {
				return nil, err
			}
			return pick(d), nil
		}}
	}
	var outs []safetensors.Out
	for _, l := range layers {
		data[l] = &layerData{}
		for e := 0; e < E; e++ {
			outs = append(outs,
				safetensors.Out{Name: GateUpName(l, e), DType: "I8", Shape: []int{H, 2 * I}, Size: int64(H * 2 * I), Data: lazy(l, func(d *layerData) []byte { b := d.gu[e]; d.gu[e] = nil; return b })},
				safetensors.Out{Name: DownName(l, e), DType: "I8", Shape: []int{H, I}, Size: int64(H * I), Data: lazy(l, func(d *layerData) []byte { b := d.dn[e]; d.dn[e] = nil; return b })})
		}
		outs = append(outs, safetensors.Out{Name: ScalesName(l), DType: "F16", Shape: []int{E, S}, Size: int64(2 * E * S), Data: lazy(l, func(d *layerData) []byte { return d.scales })})
		if names(l).Bias != "" {
			outs = append(outs, safetensors.Out{Name: BiasName(l), DType: "F32", Shape: []int{E}, Size: int64(4 * E), Data: lazy(l, func(d *layerData) []byte { return d.bias })})
		}
	}
	meta := map[string]string{"hidden": strconv.Itoa(H), "inter": strconv.Itoa(I), "experts": strconv.Itoa(E), "format": "int8-row"}
	return safetensors.Write(path, outs, meta)
}

// quantize writes w (rows, cols) as int8 codes into q, a scale a row into scale: the row's
// absolute maximum over 127, rounded to fp16 first so that the codes round against the scale
// the device multiplies by.
func quantize(w []float32, rows, cols int, q []byte, scale []float32) {
	hb := make([]byte, 2)
	sv := make([]float32, 1)
	for r := 0; r < rows; r++ {
		row := w[r*cols : (r+1)*cols]
		var m float32
		for _, v := range row {
			m = max(m, float32(math.Abs(float64(v))))
		}
		s := m / 127
		if s == 0 {
			s = 1
		}
		vecops.F32ToF16(hb, []float32{s})
		vecops.F16ToF32(sv, hb)
		s = sv[0]
		scale[r] = s
		for c, v := range row {
			x := math.Round(float64(v / s))
			q[r*cols+c] = byte(int8(max(-127, min(127, x))))
		}
	}
}

// lazyReader reads the bytes get gives, calling it at the first Read.
type lazyReader struct {
	get func() ([]byte, error)
	r   *bytes.Reader
}

func (l *lazyReader) Read(p []byte) (int, error) {
	if l.r == nil {
		b, err := l.get()
		if err != nil {
			return 0, err
		}
		l.r = bytes.NewReader(b)
	}
	return l.r.Read(p)
}
