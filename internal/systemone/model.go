package systemone

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
	_ "github.com/shibukawa/hwnpu/internal/llm/archs"
	"github.com/shibukawa/hwnpu/internal/llm/chain"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3"
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/pyjson"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

// Config says where the model is.
type Config struct {
	// Packages are the d1 model converted by hwnpu-convert, each an .mlpackage or an
	// out.parts.json, with different -seq: a prompt runs on the shortest that holds it.
	Packages []string
	// ModelConfig is config.json, Weights the checkpoint the token embeddings and the output
	// layer's rows of the labels are read from, Tokenizer tokenizer.json and Calibration
	// calibration.json (default: next to config.json).
	ModelConfig, Weights, Tokenizer, Calibration string
	// CacheDir is the compile cache ("" for none).
	CacheDir string
	// CPU runs the programs with the CPU interpreter (internal/milcpu) instead of the NPU: slow,
	// for checking the answers on a machine without one.
	CPU bool
	// Log receives what loading does (nil for none).
	Log io.Writer
	// Name is the model's name in the API (default: the first package's).
	Name string
}

// Model answers System One requests with a d1 model, one question at a time.
type Model struct {
	Name string

	tok         encoder
	cfg         *lfm3.Config
	w           llm.Weights
	dev         *hwnpu.Device
	progs       []program // by Seq
	temperature float64
	escape      func(string) string

	mu         sync.Mutex
	emb, heads map[int32][]float32
}

// encoder is the tokenizer: Encode reads the spellings of added tokens in the text as them.
type encoder interface {
	Encode(text string) []int32
}

// program is one converted package: Seq tokens per call, x and mask in, the hidden states out.
type program interface {
	Seq() int
	// Run reads the n tokens of x (Seq rows of the hidden size, padded with zeros) and returns
	// the final hidden state at the last of them.
	Run(ctx context.Context, x []float32, n int) ([]float32, error)
	Close() error
}

// Load loads the model.
func Load(ctx context.Context, c Config) (*Model, error) {
	if len(c.Packages) == 0 {
		return nil, errors.New("systemone: no package")
	}
	logf := func(format string, args ...any) {
		if c.Log != nil {
			fmt.Fprintf(c.Log, format+"\n", args...)
		}
	}
	a, model, err := llm.LoadConfig(c.ModelConfig, "")
	if err != nil {
		return nil, err
	}
	cfg, ok := model.(*lfm3.Config)
	if !ok || !cfg.D1 {
		return nil, fmt.Errorf("systemone: %s is not a d1 model (LFM3 with architectures D1Model)", c.ModelConfig)
	}
	dir := filepath.Dir(c.ModelConfig)
	if c.Tokenizer == "" {
		c.Tokenizer = filepath.Join(dir, "tokenizer.json")
	}
	if c.Calibration == "" {
		c.Calibration = filepath.Join(dir, "calibration.json")
	}
	m := &Model{Name: c.Name, cfg: cfg, emb: map[int32][]float32{}, heads: map[int32][]float32{}}
	if m.Name == "" {
		m.Name = strings.TrimSuffix(strings.TrimSuffix(filepath.Base(c.Packages[0]), ".parts.json"), ".mlpackage")
	}
	if m.temperature, err = LoadCalibration(c.Calibration); err != nil {
		return nil, err
	}
	tok, err := tokenizer.Load(c.Tokenizer)
	if err != nil {
		return nil, err
	}
	m.tok, m.escape = tok, Escaper(tok.Control())
	if m.w, err = llm.OpenWeights(c.Weights, a); err != nil {
		return nil, err
	}
	if !c.CPU {
		opts := []hwnpu.Option{hwnpu.WithBackend("coreml"), hwnpu.WithPlacement(hwnpu.PlacementPrefer)}
		if c.CacheDir != "" {
			opts = append(opts, hwnpu.WithCacheDir(c.CacheDir))
		}
		if m.dev, err = hwnpu.Open(ctx, opts...); err != nil {
			m.Close()
			return nil, err
		}
	}
	for _, pkg := range c.Packages {
		t0 := time.Now()
		var p program
		if c.CPU {
			p, err = openCPU(pkg, cfg.HiddenSize)
		} else {
			p, err = openDevice(ctx, m.dev, pkg, cfg.HiddenSize, logf)
		}
		if err != nil {
			m.Close()
			return nil, fmt.Errorf("systemone: %s: %w", pkg, err)
		}
		m.progs = append(m.progs, p)
		logf("%s: %d tokens per call, loaded in %.1fs", filepath.Base(pkg), p.Seq(), time.Since(t0).Seconds())
	}
	slices.SortFunc(m.progs, func(x, y program) int { return x.Seq() - y.Seq() })
	return m, nil
}

// LoadCalibration reads the temperature of calibration.json: one for all the answers ("default").
// Temperatures by question type and option count are rejected, as llama.cpp does.
func LoadCalibration(path string) (float64, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return 0, fmt.Errorf("systemone: %w", err)
	}
	var c struct {
		Default      float64            `json:"default"`
		Temperatures map[string]float64 `json:"temperatures"`
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		return 0, fmt.Errorf("systemone: %s: %w", path, err)
	}
	if len(c.Temperatures) > 0 {
		return 0, fmt.Errorf("systemone: %s: temperatures by bucket are not supported", path)
	}
	if c.Default <= 0 {
		return 0, fmt.Errorf("systemone: %s: no positive default temperature", path)
	}
	return c.Default, nil
}

// Close releases the model.
func (m *Model) Close() error {
	for _, p := range m.progs {
		p.Close()
	}
	m.progs = nil
	if m.dev != nil {
		m.dev.Close()
		m.dev = nil
	}
	if m.w != nil {
		m.w.Close()
		m.w = nil
	}
	return nil
}

// MaxTokens is the longest prompt the model takes.
func (m *Model) MaxTokens() int { return m.progs[len(m.progs)-1].Seq() }

// Response is the answer to a request.
type Response struct {
	Model       string
	IDs         []string // the questions' ids, in the request's order
	Answers     []Answer
	InputTokens int
}

// MarshalJSON writes the response as llama.cpp does.
func (r *Response) MarshalJSON() ([]byte, error) {
	var b bytes.Buffer
	name, _ := json.Marshal(r.Model)
	b.WriteString(`{"model":` + string(name) + `,"answers":{`)
	for i, id := range r.IDs {
		if i > 0 {
			b.WriteByte(',')
		}
		k, _ := json.Marshal(id)
		a, err := r.Answers[i].MarshalJSON()
		if err != nil {
			return nil, err
		}
		b.Write(k)
		b.WriteByte(':')
		b.Write(a)
	}
	b.WriteString(`},"usage":{"input_tokens":` + strconv.Itoa(r.InputTokens) + `,"output_tokens":0}}`)
	return b.Bytes(), nil
}

// Decide answers the questions of a request in turn.
func (m *Model) Decide(ctx context.Context, r *Request) (*Response, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	res := &Response{Model: m.Name}
	for _, q := range r.Questions {
		a, n, err := m.answer(ctx, r.State, q)
		if err != nil {
			return nil, err
		}
		res.IDs = append(res.IDs, q.ID)
		res.Answers = append(res.Answers, a)
		res.InputTokens += n
	}
	return res, nil
}

// Tokens returns the prompt of a question as tokens, with the label groups of its options.
func (m *Model) Tokens(state *pyjson.Value, q Question) (ids []int32, groups [][]int32, err error) {
	codes, groups, err := Labels(q, m.tok.Encode)
	if err != nil {
		return nil, nil, err
	}
	return m.tok.Encode(Prompt(state, q, codes, m.escape)), groups, nil
}

func (m *Model) answer(ctx context.Context, state *pyjson.Value, q Question) (Answer, int, error) {
	ids, groups, err := m.Tokens(state, q)
	if err != nil {
		return Answer{}, 0, err
	}
	n := len(ids)
	i := slices.IndexFunc(m.progs, func(p program) bool { return p.Seq() >= n })
	if i < 0 {
		return Answer{}, 0, bad("questions.%s: the prompt is %d tokens, the model takes %d (convert with a larger -seq)", q.ID, n, m.MaxTokens())
	}
	p := m.progs[i]
	d := m.cfg.HiddenSize
	x := make([]float32, p.Seq()*d)
	for j, t := range ids {
		e, err := m.row(m.emb, m.cfg.Embedding(), t)
		if err != nil {
			return Answer{}, 0, err
		}
		copy(x[j*d:], e)
	}
	h, err := p.Run(ctx, x, n)
	if err != nil {
		return Answer{}, 0, err
	}
	logits := make([]float64, len(groups))
	for gi, g := range groups {
		best := -1e300
		for _, t := range g {
			row, err := m.row(m.heads, m.cfg.Head(), t)
			if err != nil {
				return Answer{}, 0, err
			}
			var dot float64
			for j, v := range row {
				dot += float64(v) * float64(h[j])
			}
			best = max(best, dot)
		}
		logits[gi] = best
	}
	return Decide(q, logits, m.temperature), n, nil
}

// row reads row t of the named matrix, once.
func (m *Model) row(cache map[int32][]float32, name string, t int32) ([]float32, error) {
	if r, ok := cache[t]; ok {
		return r, nil
	}
	r, err := m.w.Rows(name, int(t), int(t)+1)
	if err != nil {
		return nil, err
	}
	cache[t] = r
	return r, nil
}

// devProgram runs a package on the device.
type devProgram struct {
	c          *chain.Chain
	x, mask, y *hwnpu.Buffer
	seq, d     int
}

func openDevice(ctx context.Context, dev *hwnpu.Device, path string, d int, logf func(string, ...any)) (*devProgram, error) {
	c, err := chain.Open(ctx, dev, path)
	if err != nil {
		return nil, err
	}
	p := &devProgram{c: c, x: c.Input("x"), mask: c.Input("mask"), y: c.Output(), d: d}
	if p.x == nil || p.mask == nil {
		c.Close()
		return nil, errors.New("not a d1 program: it needs the inputs x and mask")
	}
	p.seq = p.x.Size() / (2 * d)
	if p.y.Size() != 2*p.seq*d {
		c.Close()
		return nil, fmt.Errorf("output of %d bytes is not the hidden state (%d, %d) in fp16", p.y.Size(), p.seq, d)
	}
	if !c.OnNPU() {
		logf("%s: NOT all on the NPU", filepath.Base(path))
	}
	return p, nil
}

func (p *devProgram) Seq() int { return p.seq }

func (p *devProgram) Run(ctx context.Context, x []float32, n int) ([]float32, error) {
	hwnpu.EncodeF16(p.x.Bytes(), x)
	mask := make([]float32, p.seq)
	for i := range n {
		mask[i] = 1
	}
	hwnpu.EncodeF16(p.mask.Bytes(), mask)
	p.x.SyncToDevice(0, p.x.Size())
	p.mask.SyncToDevice(0, p.mask.Size())
	if err := p.c.Run(ctx); err != nil {
		return nil, err
	}
	off := 2 * (n - 1) * p.d
	p.y.SyncFromDevice(off, 2*p.d)
	h := make([]float32, p.d)
	hwnpu.DecodeF16(h, p.y.Bytes()[off:off+2*p.d])
	return h, nil
}

func (p *devProgram) Close() error { return p.c.Close() }

// cpuProgram runs a package's parts with the CPU interpreter.
type cpuProgram struct {
	parts   []*milcpu.Program
	closers []io.Closer
	seq, d  int
}

func openCPU(path string, d int) (*cpuProgram, error) {
	man, err := llm.ReadManifest(path)
	if err != nil {
		return nil, err
	}
	p := &cpuProgram{d: d}
	for _, part := range man.Tiered()[0].Parts {
		prog, f, err := milcpu.OpenPackage(part.Path)
		if err != nil {
			p.Close()
			return nil, err
		}
		p.parts, p.closers = append(p.parts, prog), append(p.closers, f)
	}
	x := p.parts[0].Inputs()["x"]
	if len(x) != 3 || x[2] != d {
		p.Close()
		return nil, fmt.Errorf("input x %v is not (1, seq, %d)", x, d)
	}
	p.seq = x[1]
	return p, nil
}

func (p *cpuProgram) Seq() int { return p.seq }

func (p *cpuProgram) Run(_ context.Context, x []float32, n int) ([]float32, error) {
	mask := make([]float32, p.seq)
	for i := range n {
		mask[i] = 1
	}
	h := x
	for _, part := range p.parts {
		in := map[string]milcpu.Tensor{}
		for name, shape := range part.Inputs() {
			switch name {
			case "x":
				in[name] = milcpu.Tensor{Shape: shape, Data: h}
			case "mask":
				in[name] = milcpu.Tensor{Shape: shape, Data: mask}
			default:
				return nil, fmt.Errorf("unexpected input %s", name)
			}
		}
		out, err := part.Run(in, milcpu.Options{})
		if err != nil {
			return nil, err
		}
		h = out["y"].Data
	}
	return h[(n-1)*p.d : n*p.d], nil
}

func (p *cpuProgram) Close() error {
	for _, c := range p.closers {
		c.Close()
	}
	return nil
}
