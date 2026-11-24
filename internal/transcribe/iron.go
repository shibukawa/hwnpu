package transcribe

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	"github.com/shibukawa/hwnpu/internal/llm/whisper"
	"github.com/shibukawa/hwnpu/internal/mapped"
	"github.com/shibukawa/hwnpu/internal/vecops"
)

// AMD's NPU runs the two graphs of tools/amd/iron_whisper.py (an export NAME.amd-npu2: encoder/,
// decoder/, model.json, and the checkpoint's config.json, generation_config.json and
// tokenizer.json): the encoder of a window, which gives the decoder's cross attention keys and
// values, and the decoder of one token, which holds its KV cache in the graph's states and
// gives the logits.

// IronMeta is model.json of an export of tools/amd/iron_whisper.py.
type IronMeta struct {
	Kind   string `json:"kind"` // whisper
	Rows   int    `json:"rows"` // the encoder's rows: its frames padded
	ConvIn int    `json:"conv_in"`
	Vocab  int    `json:"vocab_pad"`
	Frames int    `json:"frames"`
	Config struct {
		Hidden    int `json:"hidden"`
		Mels      int `json:"mels"`
		DecLayers int `json:"dec_layers"`
		Positions int `json:"positions"`
		Targets   int `json:"targets"`
		Vocab     int `json:"vocab"`
	} `json:"config"`
}

// IsIronExport reports whether dir is an export of tools/amd/iron_whisper.py.
func IsIronExport(dir string) bool {
	m, err := readIronMeta(dir)
	return err == nil && m.Kind == "whisper" && fileExists(filepath.Join(dir, "decoder", "manifest.json")) &&
		fileExists(filepath.Join(dir, "encoder", "manifest.json"))
}

func readIronMeta(dir string) (*IronMeta, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "model.json"))
	if err != nil {
		return nil, err
	}
	var m IronMeta
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, fmt.Errorf("transcribe: %s: model.json: %w", dir, err)
	}
	return &m, nil
}

// LoadIron loads the export of tools/amd/iron_whisper.py in dir on dev, AMD's NPU.
func LoadIron(ctx context.Context, dev *hwnpu.Device, dir string) (*Transcriber, error) {
	m, err := readIronMeta(dir)
	if err != nil {
		return nil, err
	}
	if m.Kind != "whisper" {
		return nil, fmt.Errorf("transcribe: %s is no export of tools/amd/iron_whisper.py", dir)
	}
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	cfg, err := whisper.ParseConfig(raw)
	if err != nil {
		return nil, err
	}
	t, err := newTranscriber(dir, cfg)
	if err != nil {
		return nil, err
	}
	p := &ironPrograms{m: m}
	ok := false
	defer func() {
		if !ok {
			p.Close()
		}
	}()
	H := m.Config.Hidden
	// mapped, not read: a token takes one row of the 133 MB
	if p.embedFile, err = mapped.Open(filepath.Join(dir, "decoder", "embed.bin")); err != nil {
		return nil, err
	}
	p.embed = p.embedFile.Bytes()
	if p.positionRows, err = os.ReadFile(filepath.Join(dir, "decoder", "positions.bin")); err != nil {
		return nil, err
	}
	if len(p.embed) != 2*H*m.Config.Vocab || len(p.positionRows) != 2*H*m.Config.Targets {
		return nil, fmt.Errorf("transcribe: %s: the decoder's tables are not %d and %d rows of %d", dir, m.Config.Vocab, m.Config.Targets, H)
	}
	if p.enc, err = iron.OpenGraph(ctx, dev, filepath.Join(dir, "encoder")); err != nil {
		return nil, fmt.Errorf("transcribe: encoder: %w", err)
	}
	if p.dec, err = iron.OpenGraph(ctx, dev, filepath.Join(dir, "decoder")); err != nil {
		return nil, fmt.Errorf("transcribe: decoder: %w", err)
	}
	p.row = make([]byte, 2*H)
	p.logits = make([]float32, m.Config.Vocab)
	ok = true
	t.p = p
	return t, nil
}

// ironPrograms are the graphs of an export of tools/amd/iron_whisper.py.
type ironPrograms struct {
	m                   *IronMeta
	enc, dec            *iron.Graph
	embed, positionRows []byte // bf16 rows (embed of embedFile)
	embedFile           *mapped.File
	row                 []byte
	logits              []float32
	at                  int
}

func (p *ironPrograms) onNPU() bool    { return true }
func (p *ironPrograms) reset()         { p.at = 0 }
func (p *ironPrograms) positions() int { return p.m.Config.Targets }
func (p *ironPrograms) pos() int       { return p.at }

func (p *ironPrograms) Close() error {
	for _, g := range []*iron.Graph{p.enc, p.dec} {
		if g != nil {
			g.Close()
		}
	}
	p.embed = nil
	p.embedFile.Close()
	return nil
}

// ConvRows is the encoder's input of a window's log mel frames (frame-major, frames x mels):
// conv1's im2col (each mel bin's three taps, padding 1), the even frames from row 0 and the odd
// from row rows (tools/amd/iron_whisper.py's conv_rows).
func ConvRows(mel []float32, mels, frames, rows, convIn int) []float32 {
	out := make([]float32, 2*rows*convIn)
	for f := 0; f < frames; f++ {
		r := f / 2
		if f%2 == 1 {
			r += rows
		}
		row := out[r*convIn:]
		for t := 0; t < 3; t++ {
			src := f + t - 1
			if src < 0 || src >= frames {
				continue
			}
			for c := 0; c < mels; c++ {
				row[c*3+t] = mel[src*mels+c]
			}
		}
	}
	return out
}

func (p *ironPrograms) encode(ctx context.Context, mel []float32) error {
	m := p.m
	x := ConvRows(mel, m.Config.Mels, m.Frames, m.Rows, m.ConvIn)
	if err := p.enc.SetInput(p.enc.Inputs()[0].Name, toBF16(x)); err != nil {
		return err
	}
	if err := p.enc.Run(ctx, map[string]int64{"nk": int64(m.Config.Positions)}); err != nil {
		return fmt.Errorf("transcribe: encoder: %w", err)
	}
	// layer i's keys are output 2i, its values 2i + 1: into the decoder's states
	for i := 0; i < m.Config.DecLayers; i++ {
		for j, name := range []string{"xk", "xv"} {
			out, err := p.enc.Output(fmt.Sprintf("out%d", 2*i+j))
			if err != nil {
				return err
			}
			b, _, toDevice, err := p.dec.Named(fmt.Sprintf("%s%d", name, i))
			if err != nil {
				return err
			}
			if len(out) != len(b) {
				return fmt.Errorf("transcribe: the encoder's %s of layer %d is %d bytes, the decoder's %d", name, i, len(out), len(b))
			}
			copy(b, out)
			if err := toDevice(); err != nil {
				return err
			}
		}
	}
	return nil
}

func (p *ironPrograms) step(ctx context.Context, id int32) ([]float32, error) {
	m := p.m
	H := m.Config.Hidden
	if int(id) >= m.Config.Vocab || id < 0 {
		return nil, fmt.Errorf("transcribe: token %d of a vocabulary of %d", id, m.Config.Vocab)
	}
	if p.at >= m.Config.Targets {
		return nil, fmt.Errorf("transcribe: the decoder holds %d positions", m.Config.Targets)
	}
	e, q := p.embed[2*H*int(id):], p.positionRows[2*H*p.at:]
	for i := 0; i < H; i++ {
		v := bf16Value(binary.LittleEndian.Uint16(e[2*i:])) + bf16Value(binary.LittleEndian.Uint16(q[2*i:]))
		binary.LittleEndian.PutUint16(p.row[2*i:], bf16Bits(v))
	}
	if err := p.dec.SetInput(p.dec.Inputs()[0].Name, p.row); err != nil {
		return nil, err
	}
	if err := p.dec.Run(ctx, map[string]int64{"pos": int64(p.at), "n": int64(p.at + 1), "nk": int64(m.Config.Positions)}); err != nil {
		return nil, fmt.Errorf("transcribe: decoder: %w", err)
	}
	p.at++
	out, err := p.dec.Output(p.dec.Outputs()[0].Name)
	if err != nil {
		return nil, err
	}
	vecops.BF16ToF32(p.logits, out)
	return p.logits, nil
}

func bf16Value(h uint16) float32 { return math.Float32frombits(uint32(h) << 16) }

// bf16Bits rounds x to bf16, to the nearest even.
func bf16Bits(x float32) uint16 {
	bits := math.Float32bits(x)
	if bits&0x7fffffff > 0x7f800000 {
		return uint16(bits>>16) | 0x40
	}
	return uint16((bits + 0x7fff + (bits>>16)&1) >> 16)
}

func toBF16(xs []float32) []byte {
	b := make([]byte, 2*len(xs))
	for i, x := range xs {
		binary.LittleEndian.PutUint16(b[2*i:], bf16Bits(x))
	}
	return b
}
