package supertonic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"path/filepath"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
	"github.com/shibukawa/hwnpu/internal/onnx"
)

// EstimatorProgram is vector_estimator as a program of a text of T characters and a latent of
// L frames: x is the noisy latent (1, Latent, L); text_emb (1, TextDim, T), style (1, 50, 256),
// latent_mask (1, 1, L), text_mask (1, 1, T), step and total (1) are its other inputs; y is
// the latent a step on. A shorter text or latent is padded with zeros and masked.
//
// The graph's sin and cos are given to it too (the Neural Engine has none: Core ML put them,
// and small operations around them, on the CPU, and fp16 holds the step's angles, up to 875,
// to 0.5): the step's embedding time (2, 64), sin and cos of step / total * scale * freqs, and
// the rotary embedding of the latent's positions, rope_sin and rope_cos (2, L, 32), and of the
// text's, text_sin and text_cos (2, T, 32): sin and cos of position / length * theta (hostSin).
func EstimatorProgram(g *onnx.Graph, T, L int) *onnx.Program {
	return &onnx.Program{G: g, Inputs: []onnx.ValueInfo{
		{Name: "noisy_latent", Shape: []int{1, Latent, L}},
		{Name: "text_emb", Shape: []int{1, TextDim, T}},
		{Name: "style_ttl", Shape: TTLShape},
		{Name: "latent_mask", Shape: []int{1, 1, L}},
		{Name: "text_mask", Shape: []int{1, 1, T}},
		{Name: "current_step", Shape: []int{1}},
		{Name: "total_step", Shape: []int{1}},
		{Name: timeNode + "Concat_output_0", Shape: []int{2, 2 * ropeDim}},
		{Name: ropeNode + "Sin_output_0", Shape: []int{2, L, ropeDim}},
		{Name: ropeNode + "Cos_output_0", Shape: []int{2, L, ropeDim}},
		{Name: ropeNode + "Sin_1_output_0", Shape: []int{2, T, ropeDim}},
		{Name: ropeNode + "Cos_1_output_0", Shape: []int{2, T, ropeDim}},
	}, Outputs: []string{"denoised_latent"}, Names: map[string]string{
		"text_emb": "text_emb", "style_ttl": "style", "latent_mask": "latent_mask", "text_mask": "text_mask",
		"current_step": "step", "total_step": "total",
		timeNode + "Concat_output_0": "time",
		ropeNode + "Sin_output_0":    "rope_sin", ropeNode + "Cos_output_0": "rope_cos",
		ropeNode + "Sin_1_output_0": "text_sin", ropeNode + "Cos_1_output_0": "text_cos",
	}}
}

// The values of vector_estimator (models.yaml's revision) whose sin and cos the host works
// out: the step's embedding, and the rotary embedding the attention of every block shares.
const (
	timeNode = "/vector_estimator/vector_field/time_encoder/sinusoidal/"
	ropeNode = "/vector_estimator/vector_field/main_blocks.3/attn/"
	ropeDim  = 32
)

// hostSin are the constants of the sin and cos the host works out, from the graph, as the
// programs' metadata keeps them.
type hostSin struct {
	Scale        float64
	Freqs, Theta []float64
}

const metaSin = "supertonic.sin"

func sinOf(g *onnx.Graph) (hostSin, error) {
	scale, freqs, theta := g.Init[timeNode+"Constant_2_output_0"], g.Init[timeNode+"Constant_3_output_0"], g.Init["vector_estimator.tts.ttl.vector_field.main_blocks.3.attn.theta"]
	if scale == nil || freqs == nil || theta == nil || len(scale.F) != 1 || len(freqs.F) != ropeDim || len(theta.F) != ropeDim {
		return hostSin{}, fmt.Errorf("supertonic: %s is not the vector_estimator of models.yaml's revision", Graphs.Estimator)
	}
	return hostSin{Scale: scale.F[0], Freqs: freqs.F, Theta: theta.F}, nil
}

// time is the step's embedding, (2, 64): sin then cos, twice.
func (h hostSin) time(step, total int) []float32 {
	out := make([]float32, 2*2*ropeDim)
	t := float64(step) / float64(total) * h.Scale
	for b := range 2 {
		for j, f := range h.Freqs {
			s, c := math.Sincos(t * f)
			out[b*2*ropeDim+j], out[b*2*ropeDim+ropeDim+j] = float32(s), float32(c)
		}
	}
	return out
}

// rope is the rotary embedding of m positions of a sequence of n, (2, m, 32): sin and cos.
func (h hostSin) rope(n, m int) (sin, cos []float32) {
	sin, cos = make([]float32, 2*m*ropeDim), make([]float32, 2*m*ropeDim)
	for i := range m {
		for j, th := range h.Theta {
			s, c := math.Sincos(float64(i) / float64(n) * th)
			for b := range 2 {
				k := (b*m+i)*ropeDim + j
				sin[k], cos[k] = float32(s), float32(c)
			}
		}
	}
	return sin, cos
}

// VocoderProgram is vocoder as a program of a latent of L frames: x (1, Latent, L), y the
// speech, L * Chunk samples in rows (Flat: an axis of the Neural Engine holds up to 16384, and
// the speech as one put its output's copy on the CPU).
func VocoderProgram(g *onnx.Graph, L int) *onnx.Program {
	return &onnx.Program{G: g, Inputs: []onnx.ValueInfo{{Name: "latent", Shape: []int{1, Latent, L}}}, Outputs: []string{"wav_tts"}, Flat: true}
}

// The text's graphs read their ids only in an embedding table (Gather), whose rows the host
// takes (fp16 counts exactly up to 2048, the table has 8322 rows): their programs are given
// the rows, (1, T, width).
const (
	dpEmbed  = "/sentence_encoder/text_embedder/char_embedder/Gather_output_0"
	encEmbed = "/text_encoder/text_embedder/char_embedder/Gather_output_0"
	dpTable  = "tts.dp.sentence_encoder.text_embedder.char_embedder.weight"
	encTable = "tts.ttl.text_encoder.text_embedder.char_embedder.weight"
	dpWidth  = 64
)

// DurationProgram is duration_predictor as a program of a text of T characters: x the rows of
// its ids (1, T, 64), style (1, 8, 16), text_mask (1, 1, T); y the speech's length in seconds.
func DurationProgram(g *onnx.Graph, T int) *onnx.Program {
	return &onnx.Program{G: g, Inputs: []onnx.ValueInfo{
		{Name: dpEmbed, Shape: []int{1, T, dpWidth}},
		{Name: "style_dp", Shape: DPShape},
		{Name: "text_mask", Shape: []int{1, 1, T}},
	}, Outputs: []string{"duration"}, Names: map[string]string{"style_dp": "style", "text_mask": "text_mask"}}
}

// EncoderProgram is text_encoder as a program of a text of T characters: x the rows of its
// ids (1, T, 256), style (1, 50, 256), text_mask (1, 1, T); y the text's embedding (1, 256, T).
func EncoderProgram(g *onnx.Graph, T int) *onnx.Program {
	return &onnx.Program{G: g, Inputs: []onnx.ValueInfo{
		{Name: encEmbed, Shape: []int{1, T, TextDim}},
		{Name: "style_ttl", Shape: TTLShape},
		{Name: "text_mask", Shape: []int{1, 1, T}},
	}, Outputs: []string{"text_emb"}, Names: map[string]string{"style_ttl": "style", "text_mask": "text_mask"}}
}

// ErrTooLong is a text, or its speech, longer than every program holds (and no CPU to run it
// on): it is to be spoken in pieces.
var ErrTooLong = errors.New("supertonic: the text is longer than the programs hold")

// npuStages run the four graphs as programs of fixed texts and latents (Sizes) on an NPU, the
// smallest that holds a text, the text and the latent padded with zeros and masked. A text
// longer than every program runs on cpu (ONNX Runtime), or is ErrTooLong without it.
type npuStages struct {
	cpu             Stages // nil: none
	progs           []npuSize
	dpRows, encRows []float32 // the embedding tables, (rows, width)
}

type npuSize struct {
	T, L              int
	dp, enc, est, voc runner.Program
	sin               hostSin
}

// OpenNPU opens the programs of c (of the same Sizes) on dev (nil: the CPU interpreter), the
// embedding tables from the model's graphs; cpu (nil for none) runs what no program holds. A
// Config of the text's programs alone (no Estimators and Vocoders) opens those (OpenAMD's).
func OpenNPU(ctx context.Context, dev *hwnpu.Device, cpu Stages, c Config) (Stages, error) {
	n := len(c.Durations)
	flow := len(c.Estimators) > 0 || len(c.Vocoders) > 0
	if n == 0 || len(c.Encoders) != n || flow && (len(c.Estimators) != n || len(c.Vocoders) != n) {
		return nil, fmt.Errorf("supertonic: %d, %d, %d and %d programs of the stages: convert them again", len(c.Durations), len(c.Encoders), len(c.Estimators), len(c.Vocoders))
	}
	s := &npuStages{cpu: cpu}
	var err error
	if s.dpRows, err = table(filepath.Join(c.Dir, "onnx", Graphs.Duration), dpTable, dpWidth); err != nil {
		return nil, err
	}
	if s.encRows, err = table(filepath.Join(c.Dir, "onnx", Graphs.Encoder), encTable, TextDim); err != nil {
		return nil, err
	}
	fail := func(err error) (Stages, error) {
		s.Close()
		return nil, err
	}
	for i := range n {
		var p npuSize
		progs := []struct {
			path string
			out  *runner.Program
		}{{c.Durations[i], &p.dp}, {c.Encoders[i], &p.enc}}
		if flow {
			progs = append(progs, struct {
				path string
				out  *runner.Program
			}{c.Estimators[i], &p.est}, struct {
				path string
				out  *runner.Program
			}{c.Vocoders[i], &p.voc})
		}
		for _, x := range progs {
			if *x.out, err = runner.Open(ctx, dev, x.path); err != nil {
				for _, q := range []runner.Program{p.dp, p.enc, p.est} {
					if q != nil {
						q.Close()
					}
				}
				return fail(err)
			}
		}
		s.progs = append(s.progs, p) // closed with s from here
		if !flow {
			s.progs[i].T = p.dp.Inputs()["x"][1]
			if et := p.enc.Inputs()["x"][1]; et != s.progs[i].T {
				return fail(fmt.Errorf("supertonic: %s is of %d characters, %s of %d", c.Durations[i], s.progs[i].T, c.Encoders[i], et))
			}
			continue
		}
		in := p.est.Inputs()
		s.progs[i].T, s.progs[i].L = in["text_emb"][2], in["x"][2]
		if err := json.Unmarshal([]byte(p.est.Metadata()[metaSin]), &s.progs[i].sin); err != nil || len(s.progs[i].sin.Theta) != ropeDim {
			return fail(fmt.Errorf("supertonic: %s has no %s: convert it again", c.Estimators[i], metaSin))
		}
		if vl := p.voc.Inputs()["x"][2]; vl != s.progs[i].L {
			return fail(fmt.Errorf("supertonic: %s is of %d frames, %s of %d", c.Estimators[i], s.progs[i].L, c.Vocoders[i], vl))
		}
		if dt, et := p.dp.Inputs()["x"][1], p.enc.Inputs()["x"][1]; dt != s.progs[i].T || et != s.progs[i].T {
			return fail(fmt.Errorf("supertonic: %s is of %d characters, %s of %d, %s of %d", c.Estimators[i], s.progs[i].T, c.Durations[i], dt, c.Encoders[i], et))
		}
	}
	return s, nil
}

// table is the embedding table name of an ONNX graph, rows of width.
func table(path, name string, width int) ([]float32, error) {
	g, err := onnx.Load(path)
	if err != nil {
		return nil, err
	}
	t := g.Init[name]
	if t == nil || len(t.Shape) != 2 || t.Shape[1] != width {
		return nil, fmt.Errorf("supertonic: %s has no table %s of %d columns: not the graph of models.yaml's revision", path, name, width)
	}
	out := make([]float32, len(t.F))
	for i, v := range t.F {
		out[i] = float32(v)
	}
	return out, nil
}

// OnNPU reports whether every program runs all on the NPU.
func (s *npuStages) OnNPU() bool {
	for _, p := range s.progs {
		for _, q := range []runner.Program{p.dp, p.enc, p.est, p.voc} {
			if q != nil && !q.OnNPU() {
				return false
			}
		}
	}
	return true
}

// size is the smallest program that holds a text of T and a latent of L, nil for none.
func (s *npuStages) size(T, L int) *npuSize {
	var best *npuSize
	for i := range s.progs {
		p := &s.progs[i]
		if T <= p.T && L <= p.L && (best == nil || p.L < best.L) {
			best = p
		}
	}
	return best
}

// rows are the table's rows of ids, padded with zeros to m rows.
func rows(tab []float32, width int, ids []int64, m int) []float32 {
	out := make([]float32, m*width)
	for i, id := range ids {
		copy(out[i*width:(i+1)*width], tab[int(id)*width:(int(id)+1)*width])
	}
	return out
}

func (s *npuStages) Duration(ctx context.Context, ids []int64, style []float32) (float32, error) {
	p := s.size(len(ids), 0)
	if p == nil {
		if s.cpu == nil {
			return 0, ErrTooLong
		}
		return s.cpu.Duration(ctx, ids, style)
	}
	y, err := p.dp.Run(ctx, map[string][]float32{
		"x": rows(s.dpRows, dpWidth, ids, p.T), "style": style, "text_mask": mask(len(ids), p.T),
	})
	if err != nil {
		return 0, err
	}
	return y[0], nil
}

func (s *npuStages) Encode(ctx context.Context, ids []int64, style []float32) ([]float32, error) {
	T := len(ids)
	p := s.size(T, 0)
	if p == nil {
		if s.cpu == nil {
			return nil, ErrTooLong
		}
		return s.cpu.Encode(ctx, ids, style)
	}
	y, err := p.enc.Run(ctx, map[string][]float32{
		"x": rows(s.encRows, TextDim, ids, p.T), "style": style, "text_mask": mask(T, p.T),
	})
	if err != nil {
		return nil, err
	}
	out := make([]float32, TextDim*T)
	for r := range TextDim {
		copy(out[r*T:(r+1)*T], y[r*p.T:r*p.T+T])
	}
	return out, nil
}

// pad lays rows of n values into rows of m, zeros after.
func pad(x []float32, rows, n, m int) []float32 {
	out := make([]float32, rows*m)
	for r := range rows {
		copy(out[r*m:r*m+n], x[r*n:(r+1)*n])
	}
	return out
}

func mask(n, m int) []float32 {
	out := make([]float32, m)
	for i := range n {
		out[i] = 1
	}
	return out
}

func (s *npuStages) Estimate(ctx context.Context, x []float32, L int, emb []float32, T int, style []float32, step, total int) ([]float32, error) {
	p := s.size(T, L)
	if p == nil {
		if s.cpu == nil {
			return nil, ErrTooLong
		}
		return s.cpu.Estimate(ctx, x, L, emb, T, style, step, total)
	}
	rs, rc := p.sin.rope(L, p.L)
	ts, tc := p.sin.rope(T, p.T)
	y, err := p.est.Run(ctx, map[string][]float32{
		"x": pad(x, Latent, L, p.L), "text_emb": pad(emb, TextDim, T, p.T), "style": style,
		"latent_mask": mask(L, p.L), "text_mask": mask(T, p.T),
		"step": {float32(step)}, "total": {float32(total)}, "time": p.sin.time(step, total),
		"rope_sin": rs, "rope_cos": rc, "text_sin": ts, "text_cos": tc,
	})
	if err != nil {
		return nil, err
	}
	out := make([]float32, Latent*L)
	for r := range Latent {
		copy(out[r*L:(r+1)*L], y[r*p.L:r*p.L+L])
	}
	return out, nil
}

func (s *npuStages) Vocode(ctx context.Context, x []float32, L int) ([]float32, error) {
	p := s.size(0, L)
	if p == nil {
		if s.cpu == nil {
			return nil, ErrTooLong
		}
		return s.cpu.Vocode(ctx, x, L)
	}
	y, err := p.voc.Run(ctx, map[string][]float32{"x": pad(x, Latent, L, p.L)})
	if err != nil {
		return nil, err
	}
	return y[:L*Chunk], nil
}

func (s *npuStages) Close() error {
	for _, p := range s.progs {
		for _, q := range []runner.Program{p.dp, p.enc, p.est, p.voc} {
			if q != nil {
				q.Close()
			}
		}
	}
	if s.cpu != nil {
		return s.cpu.Close()
	}
	return nil
}
