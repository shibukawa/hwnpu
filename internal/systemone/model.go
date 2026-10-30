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
	"github.com/shibukawa/hwnpu/internal/audio"
	"github.com/shibukawa/hwnpu/internal/llm"
	_ "github.com/shibukawa/hwnpu/internal/llm/archs"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
	"github.com/shibukawa/hwnpu/internal/pyjson"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
	"github.com/shibukawa/hwnpu/internal/vision"
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
	// Vision are the LFM3-VL vision encoder converted by hwnpu-convert -arch lfm3-vision, a
	// package per image size (internal/vision); none to take no images. Its config.json and
	// image processor settings are the ones next to ModelConfig.
	Vision []string
	// Audio is the LFM3-Omni audio encoder converted by hwnpu-convert -arch lfm3-audio-encoder,
	// a package per clip length (internal/audio), for d1-Omni; none to take no audio.
	Audio []string
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
	progs       []*program // by Seq
	vision      *vision.Encoder
	audio       *audio.Encoder
	temperature float64
	escape      func(string) string

	mu         sync.Mutex
	emb, heads map[int32][]float32
}

// encoder is the tokenizer: Encode reads the spellings of added tokens in the text as them.
type encoder interface {
	Encode(text string) []int32
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
		p, err := openProgram(ctx, m.dev, pkg, cfg.HiddenSize)
		if err != nil {
			m.Close()
			return nil, fmt.Errorf("systemone: %s: %w", pkg, err)
		}
		m.progs = append(m.progs, p)
		where := "all on the NPU"
		if !p.OnNPU() {
			where = "NOT all on the NPU"
		}
		logf("%s: %d tokens per call, %s, loaded in %.1fs", filepath.Base(pkg), p.Seq(), where, time.Since(t0).Seconds())
	}
	slices.SortFunc(m.progs, func(x, y *program) int { return x.Seq() - y.Seq() })
	if len(c.Vision) > 0 {
		t0 := time.Now()
		if m.vision, err = vision.Load(ctx, m.dev, dir, c.Vision); err != nil {
			m.Close()
			return nil, err
		}
		where := "all on the NPU"
		if !m.vision.OnNPU() {
			where = "NOT all on the NPU"
		}
		logf("vision encoder for images of %v, %s, loaded in %.1fs", m.vision.Sizes(), where, time.Since(t0).Seconds())
	}
	if len(c.Audio) > 0 {
		t0 := time.Now()
		if m.audio, err = audio.Load(ctx, m.dev, dir, c.Audio); err != nil {
			m.Close()
			return nil, err
		}
		where := "all on the NPU"
		if !m.audio.OnNPU() {
			where = "NOT all on the NPU"
		}
		logf("audio encoder for clips of up to %.0f s, %s, loaded in %.1fs", m.audio.MaxSeconds(), where, time.Since(t0).Seconds())
	}
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
	if m.vision != nil {
		m.vision.Close()
		m.vision = nil
	}
	if m.audio != nil {
		m.audio.Close()
		m.audio = nil
	}
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

// Decide answers the questions of a request in turn. The images and clips are encoded once for
// all the questions.
func (m *Model) Decide(ctx context.Context, r *Request) (*Response, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	var media [][]float32
	for _, md := range r.Media {
		rows, err := m.encode(ctx, md)
		if err != nil {
			return nil, err
		}
		media = append(media, rows)
	}
	res := &Response{Model: m.Name}
	for _, q := range r.Questions {
		all := media
		if q.Spoken != nil {
			rows, err := m.encode(ctx, Media{Audio: q.Spoken})
			if err != nil {
				return nil, err
			}
			all = append(slices.Clip(media), rows)
		}
		a, n, err := m.answer(ctx, r.State, q, all)
		if err != nil {
			return nil, err
		}
		res.IDs = append(res.IDs, q.ID)
		res.Answers = append(res.Answers, a)
		res.InputTokens += n
	}
	return res, nil
}

// encode returns the embeddings of an image or a clip.
func (m *Model) encode(ctx context.Context, md Media) ([]float32, error) {
	switch {
	case md.Image != nil && m.vision == nil:
		return nil, &Unsupported{"this server does not take images: start it with the vision encoder"}
	case md.Image != nil:
		rows, _, err := m.vision.Encode(ctx, md.Image)
		return rows, err
	case m.audio == nil:
		return nil, &Unsupported{"this server does not take audio: start it with the audio encoder (d1-Omni)"}
	}
	rows, _, err := m.audio.Encode(ctx, md.Audio)
	if errors.Is(err, audio.ErrTooLong) {
		return nil, bad("%v", err)
	}
	return rows, err
}

// imageToken stands in Tokens' result for the positions of the images' and clips' embeddings.
const imageToken = -1

// Tokens returns the prompt of a question as tokens, with the label groups of its options. Each
// image or clip (in d1-Omni a clip is wrapped like an image) is <|vision_start|>, an imageToken
// per row of its embeddings (counts: the request's media, then the spoken question's),
// <|vision_end|>.
func (m *Model) Tokens(state *pyjson.Value, q Question, counts []int) (ids []int32, groups [][]int32, err error) {
	codes, groups, err := Labels(q, m.tok.Encode)
	if err != nil {
		return nil, nil, err
	}
	media := len(counts)
	if q.Spoken != nil {
		media--
	}
	pieces := strings.Split(Prompt(state, q, codes, m.escape, media), MediaMarker)
	if len(pieces)-1 != len(counts) {
		return nil, nil, fmt.Errorf("systemone: %d media for %d places in the prompt", len(counts), len(pieces)-1)
	}
	for i, piece := range pieces {
		ids = append(ids, m.tok.Encode(piece)...)
		if i+1 < len(pieces) {
			start, end := m.tok.Encode(vision.Start), m.tok.Encode(vision.End)
			if len(start) != 1 || len(end) != 1 {
				return nil, nil, fmt.Errorf("systemone: the tokenizer has no %s and %s tokens", vision.Start, vision.End)
			}
			ids = append(ids, start...)
			for range counts[i] {
				ids = append(ids, imageToken)
			}
			ids = append(ids, end...)
		}
	}
	return ids, groups, nil
}

// answer answers one question. media are the embeddings of the request's images and clips, then
// of the spoken question, in prompt order.
func (m *Model) answer(ctx context.Context, state *pyjson.Value, q Question, media [][]float32) (Answer, int, error) {
	d := m.cfg.HiddenSize
	var counts []int
	for _, rows := range media {
		counts = append(counts, len(rows)/d)
	}
	ids, groups, err := m.Tokens(state, q, counts)
	if err != nil {
		return Answer{}, 0, err
	}
	n := len(ids)
	i := slices.IndexFunc(m.progs, func(p *program) bool { return p.Seq() >= n })
	if i < 0 {
		return Answer{}, 0, bad("questions.%s: the prompt is %d tokens, the model takes %d (convert with a larger -seq)", q.ID, n, m.MaxTokens())
	}
	p := m.progs[i]
	x := make([]float32, p.Seq()*d)
	img := slices.Concat(media...)
	for j, t := range ids {
		if t == imageToken {
			copy(x[j*d:(j+1)*d], img[:d])
			img = img[d:]
			continue
		}
		e, err := m.row(m.emb, m.cfg.Embedding(), t)
		if err != nil {
			return Answer{}, 0, err
		}
		copy(x[j*d:], e)
	}
	h, err := p.last(ctx, x, n)
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

// program is one converted package: Seq tokens per call, x and mask in, the hidden states out.
type program struct {
	runner.Program
	seq, d int
}

func openProgram(ctx context.Context, dev *hwnpu.Device, path string, d int) (*program, error) {
	p, err := runner.Open(ctx, dev, path)
	if err != nil {
		return nil, err
	}
	x, mask, y := p.Inputs()["x"], p.Inputs()["mask"], p.Output()
	if len(x) != 3 || x[2] != d || len(mask) != 2 || mask[1] != x[1] || len(p.Inputs()) != 2 {
		p.Close()
		return nil, fmt.Errorf("not a d1 program: it takes %v, not x (1, seq, %d) and mask (1, seq)", p.Inputs(), d)
	}
	if len(y) != 3 || y[1] != x[1] || y[2] != d {
		p.Close()
		return nil, fmt.Errorf("output %v is not the hidden state (1, %d, %d)", y, x[1], d)
	}
	return &program{Program: p, seq: x[1], d: d}, nil
}

// Seq is the tokens per call.
func (p *program) Seq() int { return p.seq }

// last reads the n tokens of x (Seq rows of the hidden size, padded with zeros) and returns the
// final hidden state at the last of them.
func (p *program) last(ctx context.Context, x []float32, n int) ([]float32, error) {
	mask := make([]float32, p.seq)
	for i := range n {
		mask[i] = 1
	}
	y, err := p.Run(ctx, map[string][]float32{"x": x, "mask": mask})
	if err != nil {
		return nil, err
	}
	return y[(n-1)*p.d : n*p.d], nil
}
