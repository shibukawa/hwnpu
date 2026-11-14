package siglip

import (
	"cmp"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/vision/siglip/mil"
)

// ExportVersion counts the changes to the programs of Prepare that call for converting again
// (the export's directory is NAME-vN.coreml).
const ExportVersion = 1

// ProgramLabels is the texts a call of a converted text program takes.
const ProgramLabels = 8

// FormatCoreML and FormatCoreAI are Export.Format of an export of Prepare: Core ML packages
// (vision.mlpackage, text.mlpackage) or Core AI programs (vision.aimodel, text.aimodel), with
// text/embed.bin in fp16. An export of tools/amd/iron_siglip2.py has none.
const (
	FormatCoreML = llm.BackendCoreML
	FormatCoreAI = llm.BackendCoreAI
)

// IsProgramFormat reports whether an export's Format is Prepare's (programs, not IRON's graphs).
func IsProgramFormat(format string) bool { return format == FormatCoreML || format == FormatCoreAI }

// Setup is what Prepare makes ready.
type Setup struct {
	Dir  string // the checkpoints (Dir/NAME) and the exports
	Repo string // the Hugging Face repository (google/siglip2-base-patch16-224)
	Name string
	// Format is the program format: FormatCoreAI, or FormatCoreML ("", for the CPU interpreter).
	Format string
	Log    io.Writer
}

// Prepare makes a SigLIP 2 ready for Apple's Neural Engine in s.Dir, as hwnpu-chat makes its
// models ready: it downloads the checkpoint when missing and converts it, in Go, into the
// export NAME-vN.FORMAT (the two towers in fp16, the tokenizer and the token embedding table the
// host looks up), whose directory it returns. The first load compiles the programs.
func Prepare(ctx context.Context, s Setup) (string, error) {
	log := s.Log
	if log == nil {
		log = io.Discard
	}
	root := filepath.Join(s.Dir, s.Name)
	format := cmp.Or(s.Format, FormatCoreML)
	out := filepath.Join(s.Dir, fmt.Sprintf("%s-v%d.%s", strings.ToLower(s.Name), ExportVersion, format))
	if _, err := os.Stat(out + ".done"); err == nil {
		return out, nil
	}
	if err := os.MkdirAll(root, 0o755); err != nil {
		return "", err
	}
	for _, f := range []string{"config.json", "tokenizer.json", "model.safetensors"} {
		if err := chat.Fetch(ctx, s.Repo, f, filepath.Join(root, f), log); err != nil {
			return "", err
		}
	}
	bar := console.NewBar(log, "converting "+filepath.Base(out))
	if err := Convert(root, out, format); err != nil {
		bar.Done("failed")
		return "", err
	}
	bar.Done(console.Duration(bar.Elapsed()))
	return out, nil
}

// Convert writes the export of the SigLIP 2 checkpoint root (config.json, model.safetensors,
// tokenizer.json) into out in format (FormatCoreML or FormatCoreAI): the vision and text
// programs, text/tokenizer.json, text/embed.bin (the token embedding rows in fp16) and
// model.json; out.done, written last, says it is whole.
func Convert(root, out, format string) error {
	c, err := mil.ReadConfig(root, ProgramLabels)
	if err != nil {
		return err
	}
	wt, err := mil.OpenWeights(root)
	if err != nil {
		return err
	}
	defer wt.Close()
	if err := os.RemoveAll(out); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Join(out, "text"), 0o755); err != nil {
		return err
	}
	for _, p := range []*mil.Part{mil.Vision(c), mil.Text(c)} {
		if _, err := llm.WritePart(filepath.Join(out, p.Name+llm.ProgramExt(format)), p, wt, llm.Options{Backend: format, Format: "fp16"}, func(pkg *mlpackage.Package) {
			pkg.Description = fmt.Sprintf("SigLIP 2's %s tower of %s, fp16 (hwnpu)", p.Name, filepath.Base(root))
		}); err != nil {
			return fmt.Errorf("siglip: %s: %w", p.Name, err)
		}
	}
	tok, err := os.ReadFile(filepath.Join(root, "tokenizer.json"))
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(out, "text", "tokenizer.json"), tok, 0o644); err != nil {
		return err
	}
	if err := writeEmbed(filepath.Join(out, "text", "embed.bin"), wt, c); err != nil {
		return err
	}
	scale, _, err := wt.Get("logit_scale")
	if err != nil {
		return err
	}
	bias, _, err := wt.Get("logit_bias")
	if err != nil {
		return err
	}
	if len(scale) != 1 || len(bias) != 1 {
		return fmt.Errorf("siglip: %s: logit_scale and logit_bias of %d and %d values", root, len(scale), len(bias))
	}
	x := Export{Name: filepath.Base(root), Kind: "siglip", Format: format, Version: ExportVersion,
		LogitScale: math.Exp(float64(scale[0])), LogitBias: float64(bias[0]),
		Mean: []float32{0.5, 0.5, 0.5}, Std: []float32{0.5, 0.5, 0.5}, EOS: 1, Pad: 0, Lowercase: true,
		PatchIn: c.PatchIn(), Tokens: c.Patches(), NPatches: c.Patches(), Labels: c.Labels, TextLen: mil.TextLen}
	x.Config.Vision.Hidden, x.Config.Text.Hidden = c.Vision.Hidden, c.Text.Hidden
	x.Config.Image, x.Config.Patch, x.Config.Vocab, x.Config.Projection = c.Image, c.Patch, c.Vocab, c.Projection
	b, err := json.MarshalIndent(x, "", " ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(out, "model.json"), append(b, '\n'), 0o644); err != nil {
		return err
	}
	return os.WriteFile(out+".done", nil, 0o644)
}

// writeEmbed writes the token embedding table as fp16 rows.
func writeEmbed(path string, wt *mil.Weights, c mil.Config) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	const step = 4096
	E := c.Text.Hidden
	buf := make([]byte, 2*step*E)
	for r := 0; r < c.Vocab; r += step {
		n := min(step, c.Vocab-r)
		rows, err := wt.Rows(mil.EmbedName, r, r+n)
		if err != nil {
			return err
		}
		if len(rows) != n*E {
			return fmt.Errorf("siglip: %s: %d values in %d rows of %d", mil.EmbedName, len(rows), n, E)
		}
		for i, v := range rows {
			binary.LittleEndian.PutUint16(buf[2*i:], mlpackage.Float16Bits(v))
		}
		if _, err := f.Write(buf[:2*n*E]); err != nil {
			return err
		}
	}
	return f.Close()
}

// programs are the towers of an export of Prepare, each loaded when first used.
type programs struct {
	vision, text runner.Program
}

func (m *Model) program(ctx context.Context, p *runner.Program, name string) error {
	if *p != nil {
		return nil
	}
	t0 := time.Now()
	x, err := runner.Open(ctx, m.dev, filepath.Join(m.dir, name+llm.ProgramExt(m.Format)))
	if err != nil {
		return fmt.Errorf("siglip: %s: %w", name, err)
	}
	if m.dev != nil && !x.OnNPU() {
		x.Close()
		return fmt.Errorf("siglip: %s does not run on the NPU alone", name)
	}
	*p = x
	if m.Logf != nil {
		m.Logf("siglip: loaded %s in %.2f s, %s", name, time.Since(t0).Seconds(), where(m.dev))
	}
	return nil
}

func where(dev *hwnpu.Device) string {
	if dev == nil {
		return "on the CPU interpreter"
	}
	return "all on the NPU"
}

// imageProgram is Image with the vision program.
func (m *Model) imageProgram(ctx context.Context, img image.Image) ([]float32, error) {
	if err := m.program(ctx, &m.prog.vision, "vision"); err != nil {
		return nil, err
	}
	t0 := time.Now()
	in := m.PatchRows(img)
	t1 := time.Now()
	y, err := m.prog.vision.Run(ctx, map[string][]float32{"patches": in})
	if err != nil {
		return nil, fmt.Errorf("siglip: vision: %w", err)
	}
	t2 := time.Now()
	if len(y) < m.Dim() {
		return nil, fmt.Errorf("siglip: vision: %d values for an embedding of %d", len(y), m.Dim())
	}
	e := normalize(append([]float32(nil), y[:m.Dim()]...)) // row 0 of the head's rows
	m.Times = [3]time.Duration{t1.Sub(t0), t2.Sub(t1), time.Since(t2)}
	return e, nil
}

// textsPrograms is Texts with the text program: Labels texts a call, batch-major (text i's
// token t at row i 64 + t). A call's unused texts are copies of its first, not zeros (a row of
// zeros through a layer norm in fp16 is 0 / 0).
func (m *Model) textsProgram(ctx context.Context, texts []string) ([][]float32, error) {
	if err := m.program(ctx, &m.prog.text, "text"); err != nil {
		return nil, err
	}
	if err := m.openText(); err != nil {
		return nil, err
	}
	E, L, N, P := m.Config.Text.Hidden, m.Labels, m.TextLen, m.Config.Projection
	out := make([][]float32, 0, len(texts))
	in := make([]float32, L*N*E)
	row := make([]byte, 2*E)
	m.Times = [3]time.Duration{}
	for s := 0; s < len(texts); s += L {
		t0 := time.Now()
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
				dst := in[(i*N+t)*E:][:E]
				for j := range dst {
					dst[j] = mlpackage.Float16Value(binary.LittleEndian.Uint16(row[2*j:]))
				}
			}
		}
		for i := n; i < L; i++ {
			copy(in[i*N*E:(i+1)*N*E], in[:N*E])
		}
		t1 := time.Now()
		y, err := m.prog.text.Run(ctx, map[string][]float32{"x": in})
		if err != nil {
			return nil, fmt.Errorf("siglip: text: %w", err)
		}
		t2 := time.Now()
		if len(y) != L*P {
			return nil, fmt.Errorf("siglip: text: %d values for %d embeddings of %d", len(y), L, P)
		}
		for i := 0; i < n; i++ {
			out = append(out, normalize(append([]float32(nil), y[i*P:(i+1)*P]...)))
		}
		m.Times[0] += t1.Sub(t0)
		m.Times[1] += t2.Sub(t1)
		m.Times[2] += time.Since(t2)
	}
	return out, nil
}
