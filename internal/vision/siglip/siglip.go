// Package siglip runs SigLIP 2 (google/siglip2-*, Apache-2.0) on AMD's NPU: embeddings of
// pictures and texts in one space, and zero-shot labels (the sigmoid of their scaled cosine).
// The graphs are tools/amd/iron_siglip2.py's: the vision tower and its attention pooling head,
// and the text tower on Labels texts a call. The host resizes and scales the picture, cuts its
// patches, lowercases and tokenizes the texts (Gemma's tokenizer, <eos>, padded to 64), looks
// up the token embedding rows, normalizes the embeddings and takes the logits.
package siglip

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3vl"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

// Export is model.json of an export of tools/amd/iron_siglip2.py.
type Export struct {
	Name       string    `json:"name"`
	Kind       string    `json:"kind"`
	LogitScale float64   `json:"logit_scale"` // exp of the checkpoint's logit_scale
	LogitBias  float64   `json:"logit_bias"`
	Mean       []float32 `json:"mean"`
	Std        []float32 `json:"std"`
	EOS        int32     `json:"eos"`
	Pad        int32     `json:"pad"`
	Lowercase  bool      `json:"lowercase"`
	PatchIn    int       `json:"patch_in"` // a patch's values, padded
	Tokens     int       `json:"tokens"`   // the vision graph's rows
	NPatches   int       `json:"patches"`  // the picture's patches: the rows that are tokens
	Labels     int       `json:"labels"`   // the texts a call of the text graph takes
	TextLen    int       `json:"text_len"` // a text's tokens (64)
	Config     struct {
		Vision struct {
			Hidden int `json:"hidden"`
		} `json:"vision"`
		Text struct {
			Hidden int `json:"hidden"`
		} `json:"text"`
		Image      int `json:"image"`
		Patch      int `json:"patch"`
		Vocab      int `json:"vocab"`
		Projection int `json:"projection"`
	} `json:"config"`
}

// ReadExport reads dir's model.json.
func ReadExport(dir string) (*Export, error) {
	b, err := os.ReadFile(filepath.Join(dir, "model.json"))
	if err != nil {
		return nil, err
	}
	var x Export
	if err := json.Unmarshal(b, &x); err != nil {
		return nil, fmt.Errorf("siglip: %w", err)
	}
	if x.Kind != "siglip" || x.Labels == 0 || x.TextLen == 0 || x.Config.Patch == 0 {
		return nil, fmt.Errorf("siglip: %s is not an export of tools/amd/iron_siglip2.py", dir)
	}
	return &x, nil
}

// IsExport reports whether dir is an export of tools/amd/iron_siglip2.py.
func IsExport(dir string) bool {
	_, err := ReadExport(dir)
	return err == nil
}

// Model is a loaded export. Each tower's graph is loaded when first used.
type Model struct {
	Export
	dev          *hwnpu.Device
	dir          string
	vision, text *iron.Graph
	tok          *tokenizer.Tokenizer
	embed        *os.File
	mu           sync.Mutex
	// Times are the last call's: preparing the input on the host, the NPU, the host after.
	Times [3]time.Duration
	// Logf, when set, is told what is loaded.
	Logf func(format string, args ...any)
}

// Open reads the export in dir.
func Open(ctx context.Context, dev *hwnpu.Device, dir string) (*Model, error) {
	x, err := ReadExport(dir)
	if err != nil {
		return nil, err
	}
	return &Model{Export: *x, dev: dev, dir: dir}, nil
}

// Close releases the device memory and the files.
func (m *Model) Close() error {
	var err error
	for _, g := range []**iron.Graph{&m.vision, &m.text} {
		if *g != nil {
			if e := (*g).Close(); err == nil {
				err = e
			}
			*g = nil
		}
	}
	if m.embed != nil {
		m.embed.Close()
		m.embed = nil
	}
	return err
}

// Dim is the embeddings' length.
func (m *Model) Dim() int { return m.Config.Vision.Hidden }

func (m *Model) load(ctx context.Context, g **iron.Graph, name string) error {
	if *g != nil {
		return nil
	}
	t0 := time.Now()
	x, err := iron.OpenGraph(ctx, m.dev, filepath.Join(m.dir, name))
	if err != nil {
		return fmt.Errorf("siglip: %s: %w", name, err)
	}
	*g = x
	if m.Logf != nil {
		m.Logf("siglip: loaded %s in %.2f s", name, time.Since(t0).Seconds())
	}
	return nil
}

// PatchRows is the vision graph's input of img: stretched to the model's size (PIL's
// antialiased bilinear, as the processor), scaled to [-1, 1], each patch's values (channel,
// row, column) a row, zeros after.
func (m *Model) PatchRows(img image.Image) []float32 {
	S, P := m.Config.Image, m.Config.Patch
	px := lfm3vl.Resize(img, S, S, "bilinear")
	g := S / P
	out := make([]float32, m.Tokens*m.PatchIn)
	for py := 0; py < g; py++ {
		for pxi := 0; pxi < g; pxi++ {
			row := out[(py*g+pxi)*m.PatchIn:]
			for c := 0; c < 3; c++ {
				for ky := 0; ky < P; ky++ {
					for kx := 0; kx < P; kx++ {
						v := float32(px[((py*P+ky)*S+pxi*P+kx)*3+c]) / 255
						row[(c*P+ky)*P+kx] = (v - m.Mean[c]) / m.Std[c]
					}
				}
			}
		}
	}
	return out
}

// Image is img's embedding, normalized.
func (m *Model) Image(ctx context.Context, img image.Image) ([]float32, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if err := m.load(ctx, &m.vision, "vision"); err != nil {
		return nil, err
	}
	t0 := time.Now()
	if err := m.vision.SetInput("patches", toBF16(m.PatchRows(img))); err != nil {
		return nil, err
	}
	t1 := time.Now()
	if err := m.vision.Run(ctx, map[string]int64{"nk": int64(m.NPatches)}); err != nil {
		return nil, fmt.Errorf("siglip: vision: %w", err)
	}
	out, err := m.vision.Output(m.vision.Outputs()[0].Name)
	if err != nil {
		return nil, err
	}
	t2 := time.Now()
	e := normalize(bf16Row(out, 0, m.Dim()))
	m.Times = [3]time.Duration{t1.Sub(t0), t2.Sub(t1), time.Since(t2)}
	return e, nil
}

// TokenIDs is a text's token ids as the text tower takes them: lowercased (SigLIP 2 was trained
// on lowercased text), tokenized, cut to 63 tokens, <eos>, then <pad> to 64.
func (m *Model) TokenIDs(text string) ([]int32, error) {
	if err := m.openText(); err != nil {
		return nil, err
	}
	if m.Lowercase {
		text = strings.ToLower(text)
	}
	ids := m.tok.Encode(text)
	if len(ids) > m.TextLen-1 {
		ids = ids[:m.TextLen-1]
	}
	ids = append(ids, m.EOS)
	for len(ids) < m.TextLen {
		ids = append(ids, m.Pad)
	}
	return ids, nil
}

func (m *Model) openText() error {
	if m.tok != nil {
		return nil
	}
	tok, err := tokenizer.Load(filepath.Join(m.dir, "text", "tokenizer.json"))
	if err != nil {
		return fmt.Errorf("siglip: %w", err)
	}
	f, err := os.Open(filepath.Join(m.dir, "text", "embed.bin"))
	if err != nil {
		return fmt.Errorf("siglip: %w", err)
	}
	m.tok, m.embed = tok, f
	return nil
}

// Texts are the embeddings of texts, normalized, Labels texts a call of the text graph.
func (m *Model) Texts(ctx context.Context, texts []string) ([][]float32, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if err := m.load(ctx, &m.text, "text"); err != nil {
		return nil, err
	}
	if err := m.openText(); err != nil {
		return nil, err
	}
	E, L, N := m.Config.Text.Hidden, m.Labels, m.TextLen
	out := make([][]float32, 0, len(texts))
	in := make([]byte, N*L*E*2)
	row := make([]byte, 2*E)
	m.Times = [3]time.Duration{}
	for s := 0; s < len(texts); s += L {
		t0 := time.Now()
		clear(in)
		n := min(L, len(texts)-s)
		for i := 0; i < n; i++ {
			ids, err := m.TokenIDs(texts[s+i])
			if err != nil {
				return nil, err
			}
			for t, id := range ids {
				if id < 0 || int(id) >= m.Config.Vocab {
					return nil, fmt.Errorf("siglip: token %d outside the vocabulary", id)
				}
				if _, err := m.embed.ReadAt(row, int64(id)*int64(2*E)); err != nil {
					return nil, fmt.Errorf("siglip: embed.bin: %w", err)
				}
				copy(in[(t*L+i)*2*E:], row) // token-major: row t L + i
			}
		}
		if err := m.text.SetInput("x", in); err != nil {
			return nil, err
		}
		t1 := time.Now()
		if err := m.text.Run(ctx, nil); err != nil {
			return nil, fmt.Errorf("siglip: text: %w", err)
		}
		y, err := m.text.Output(m.text.Outputs()[0].Name)
		if err != nil {
			return nil, err
		}
		t2 := time.Now()
		P := m.Config.Projection
		for i := 0; i < n; i++ {
			out = append(out, normalize(bf16Row(y, (N-1)*L+i, P)))
		}
		m.Times[0] += t1.Sub(t0)
		m.Times[1] += t2.Sub(t1)
		m.Times[2] += time.Since(t2)
	}
	return out, nil
}

// Probability is the model's probability that a text of embedding txt describes a picture of
// embedding img (both normalized): the sigmoid of their cosine times the scale plus the bias.
func (m *Model) Probability(img, txt []float32) float64 {
	return 1 / (1 + math.Exp(-m.Logit(img, txt)))
}

// Logit is the cosine of img and txt times the model's scale, plus its bias.
func (m *Model) Logit(img, txt []float32) float64 {
	return Dot(img, txt)*m.LogitScale + m.LogitBias
}

// Dot is a · b in float64.
func Dot(a, b []float32) float64 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

func normalize(v []float32) []float32 {
	n := math.Sqrt(Dot(v, v))
	if n == 0 {
		return v
	}
	for i := range v {
		v[i] = float32(float64(v[i]) / n)
	}
	return v
}

// bf16Row is row r of a (rows, width) bf16 matrix, as float32.
func bf16Row(b []byte, r, width int) []float32 {
	out := make([]float32, width)
	for j := range out {
		out[j] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(b[2*(r*width+j):])) << 16)
	}
	return out
}

func toBF16(xs []float32) []byte {
	b := make([]byte, 2*len(xs))
	for i, x := range xs {
		bits := math.Float32bits(x)
		var h uint16
		if bits&0x7fffffff > 0x7f800000 {
			h = uint16(bits>>16) | 0x40
		} else {
			h = uint16((bits + 0x7fff + (bits>>16)&1) >> 16)
		}
		binary.LittleEndian.PutUint16(b[2*i:], h)
	}
	return b
}
