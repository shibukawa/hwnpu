package supertonic

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	"github.com/shibukawa/hwnpu/internal/safetensors"
	"github.com/shibukawa/hwnpu/internal/vecops"
)

// On AMD's NPU the flow and the vocoder run as the graphs of tools/amd/iron_supertonic.py (an
// export NAME.amd-npu2 of each of Sizes), the text's two graphs as Core ML packages on the CPU
// interpreter (Prepare's of them alone): no ONNX Runtime either.

// AMDTool is the IRON exporter of the flow and the vocoder (tools/amd, built into hwnpu).
const AMDTool = "iron_supertonic.py"

// AMDExportVersion is the version of PrepareAMD's exports (a key of the shared layers:
// chat.PinnedRecipe), raised by hand when what PrepareAMD passes AMDTool changes.
const AMDExportVersion = 1

// AMDFiles are the repository's files AMDTool reads.
var AMDFiles = []string{"onnx/" + Graphs.Estimator, "onnx/" + Graphs.Vocoder}

// AMDExportName is the name of the export for AMD's NPU: NAME.amd-npu2 in lower case.
func AMDExportName(name string) string { return strings.ToLower(name) + ".amd-npu2" }

// amdModel is model.json of an export of AMDTool.
type amdModel struct {
	Kind  string `json:"kind"`
	Sizes []Size `json:"sizes"`
}

func readAMDModel(dir string) (*amdModel, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "model.json"))
	if err != nil {
		return nil, err
	}
	var m amdModel
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, fmt.Errorf("supertonic: %s: model.json: %w", dir, err)
	}
	return &m, nil
}

func estimatorDir(dir string, s Size) string {
	return filepath.Join(dir, fmt.Sprintf("estimator-t%d-l%d", s.T, s.L))
}

func vocoderDir(dir string, s Size) string {
	return filepath.Join(dir, fmt.Sprintf("vocoder-l%d", s.L))
}

// IsIronExport reports whether dir is a whole export of AMDTool of every size of Sizes.
func IsIronExport(dir string) bool {
	m, err := readAMDModel(dir)
	if err != nil || m.Kind != Arch {
		return false
	}
	have := map[Size]bool{}
	for _, s := range m.Sizes {
		have[s] = true
	}
	for _, s := range Sizes {
		if !have[s] {
			return false
		}
		for _, f := range []string{filepath.Join(estimatorDir(dir, s), "manifest.json"), filepath.Join(vocoderDir(dir, s), "manifest.json")} {
			if _, err := os.Stat(f); err != nil {
				return false
			}
		}
	}
	_, err = os.Stat(filepath.Join(dir, "host.safetensors"))
	return err == nil
}

// PrepareAMD makes the export of the flow and the vocoder of m for AMD's NPU ready and returns
// its directory: here (cache.ExportRoots; one made by hand goes in the sources written back
// to), else from the shared layers, else it downloads the graphs and exports them with IRON in
// IRON's environment (chat.IRONEnv) into the cache's entries/amd-xdna2, and puts it in the
// sources written back to.
func PrepareAMD(ctx context.Context, m chat.Model, log io.Writer) (string, error) {
	if log == nil {
		log = io.Discard
	}
	name := AMDExportName(m.Name)
	if d := cache.FindExport(name, IsIronExport); d != "" {
		chat.ShareMadeByHand(m, d, log)
		return d, nil
	}
	out := filepath.Join(chat.AMDRoot(), name)
	imported := []chat.SharedRecipe{{Kind: chat.ImportKind, Recipe: chat.ImportedRecipe(m, cache.AMDTarget, name)}}
	err := chat.SharedExportOr(ctx, chat.ExportKind(AMDTool), chat.PinnedRecipe(m, AMDTool, AMDExportVersion), imported, out, log, func() error {
		python, err := chat.IRONPython(ctx)
		if err != nil {
			return fmt.Errorf("%s has no export for AMD's NPU here (%s) or in the sources, and making one needs IRON (amd/IRON with mlir-aie and Peano, Python 3.12): %w. Install it as INSTALL_AMD.md says and set %s to its virtual environment or to a script that sets it up, or put an export made elsewhere in a source (hwnpu-dev sources import)",
				m.Name, out, err, chat.IRONEnv)
		}
		fmt.Fprintf(log, "%s: no export of the flow and the vocoder for AMD's NPU here or in the sources; exporting them with IRON\n", m.Name)
		want := make([]chat.File, len(AMDFiles))
		for i, f := range AMDFiles {
			want[i] = chat.File{Repo: m.Repo, Name: f}
		}
		if err := chat.FetchFiles(ctx, want, log); err != nil {
			return err
		}
		args := []string{filepath.Join(chat.RepoDir(m.Repo), "onnx"), chat.ExportOut, "--sizes"}
		for _, s := range Sizes {
			args = append(args, fmt.Sprintf("%dx%d", s.T, s.L))
		}
		return chat.RunIRON(ctx, python, m.Name, AMDTool, args, out, log)
	})
	if err != nil {
		return "", err
	}
	if !IsIronExport(out) {
		return "", fmt.Errorf("supertonic: %s is no whole export of %s", out, AMDTool)
	}
	return out, nil
}

// amdStages run the text's graphs as text (npuStages of the CPU interpreter's programs) and
// the flow and the vocoder as the export's graphs on AMD's NPU.
type amdStages struct {
	text  *npuStages
	sizes []*amdSize
	host  amdHost
}

// amdSize is an estimator graph and a vocoder graph of one Size, and how the estimator lays its
// stream out (its manifest's meta).
type amdSize struct {
	Size
	est, voc *iron.Graph
	meta     struct {
		P       int `json:"P"`
		Segment int `json:"segment"`
		Rows    int `json:"rows"`
		Cols    int `json:"latent_cols"`
		Style   int `json:"style_rows"`
		Q       int `json:"Q"`
	}
}

// amdHost is host.safetensors: what the host works out of each step (the time embedding of the
// time blocks) and the rotary embedding's frequencies.
type amdHost struct {
	freqs, w0, b0, w2, b2, rowsW, rowsB, theta []float32
	blocks                                     int
}

func readAMDHost(path string) (amdHost, error) {
	f, err := safetensors.Open(path)
	if err != nil {
		return amdHost{}, err
	}
	defer f.Close()
	var h amdHost
	for _, t := range []struct {
		name string
		to   *[]float32
	}{{"time.freqs", &h.freqs}, {"time.w0", &h.w0}, {"time.b0", &h.b0}, {"time.w2", &h.w2}, {"time.b2", &h.b2},
		{"time.rows.w", &h.rowsW}, {"time.rows.b", &h.rowsB}, {"rope.theta", &h.theta}} {
		if *t.to, err = f.Float32(t.name); err != nil {
			return h, fmt.Errorf("supertonic: %s: %w", path, err)
		}
	}
	if len(h.theta) != ropeDim || len(h.rowsB) == 0 || len(h.rowsB)%512 != 0 || len(h.rowsW) != len(h.rowsB)*len(h.b2) {
		return h, fmt.Errorf("supertonic: %s is not of this hwnpu's export", path)
	}
	h.blocks = len(h.rowsB) / 512
	return h, nil
}

// timeRows are the time blocks' rows (blocks, 512) of a step: the sinusoid of step / total
// through the time encoder's MLP (Mish), then each block's Linear.
func (h amdHost) timeRows(step, total int) []float32 {
	n := len(h.freqs)
	s := make([]float64, 2*n)
	t := float64(step) / float64(total) * 1000
	for j, f := range h.freqs {
		s[j], s[n+j] = math.Sincos(t * float64(f))
	}
	lin := func(w, b []float32, x []float64) []float64 {
		out := make([]float64, len(b))
		for o := range out {
			acc := float64(b[o])
			for i, v := range x {
				acc += float64(w[o*len(x)+i]) * v
			}
			out[o] = acc
		}
		return out
	}
	u := lin(h.w0, h.b0, s)
	for i, v := range u {
		u[i] = v * math.Tanh(math.Log1p(math.Exp(v)))
	}
	e := lin(h.w2, h.b2, u)
	out := make([]float32, h.blocks*512)
	for k := range h.blocks {
		r := lin(h.rowsW[k*512*len(e):(k+1)*512*len(e)], h.rowsB[k*512:(k+1)*512], e)
		for i, v := range r {
			out[k*512+i] = float32(v)
		}
	}
	return out
}

// angles are RoPE's table of m positions of a sequence of n, (m, 64): the cosine and the sine
// of position / n * theta, interleaved.
func (h amdHost) angles(n, m int) []float32 {
	out := make([]float32, m*2*ropeDim)
	for i := range m {
		for j, th := range h.theta {
			s, c := math.Sincos(float64(i) / float64(n) * float64(th))
			out[i*2*ropeDim+2*j], out[i*2*ropeDim+2*j+1] = float32(c), float32(s)
		}
	}
	return out
}

// OpenAMD opens the text's programs of c (Durations and Encoders, Core ML packages) on the CPU
// interpreter and the export dir's graphs on amd, AMD's NPU.
func OpenAMD(ctx context.Context, amd *hwnpu.Device, c Config, dir string) (Stages, error) {
	text, err := OpenNPU(ctx, nil, nil, Config{Dir: c.Dir, Durations: c.Durations, Encoders: c.Encoders})
	if err != nil {
		return nil, err
	}
	s := &amdStages{text: text.(*npuStages)}
	fail := func(err error) (Stages, error) {
		s.Close()
		return nil, err
	}
	if s.host, err = readAMDHost(filepath.Join(dir, "host.safetensors")); err != nil {
		return fail(err)
	}
	for _, size := range Sizes {
		p := &amdSize{Size: size}
		s.sizes = append(s.sizes, p) // closed with s from here
		if p.est, err = iron.OpenGraph(ctx, amd, estimatorDir(dir, size)); err != nil {
			return fail(fmt.Errorf("supertonic: the flow of %d x %d: %w", size.T, size.L, err))
		}
		if err := json.Unmarshal(p.est.Manifest.Meta, &p.meta); err != nil || p.meta.Rows == 0 || p.meta.Q == 0 {
			return fail(fmt.Errorf("supertonic: %s has no layout in its meta: export it again", estimatorDir(dir, size)))
		}
		if p.voc, err = iron.OpenGraph(ctx, amd, vocoderDir(dir, size)); err != nil {
			return fail(fmt.Errorf("supertonic: the vocoder of %d frames: %w", size.L, err))
		}
	}
	return s, nil
}

// OnNPU reports that the flow and the vocoder run on AMD's NPU (the text's graphs on the CPU).
func (s *amdStages) OnNPU() bool { return true }

func (s *amdStages) Duration(ctx context.Context, ids []int64, style []float32) (float32, error) {
	return s.text.Duration(ctx, ids, style)
}

func (s *amdStages) Encode(ctx context.Context, ids []int64, style []float32) ([]float32, error) {
	return s.text.Encode(ctx, ids, style)
}

// size is the smallest size that holds a text of T and a latent of L, nil for none.
func (s *amdStages) size(T, L int) *amdSize {
	for _, p := range s.sizes {
		if T <= p.T && L <= p.L {
			return p
		}
	}
	return nil
}

func setBF16(g *iron.Graph, name string, xs []float32) error {
	b := make([]byte, 2*len(xs))
	putBF16(b, xs)
	return g.SetInput(name, b)
}

func output(g *iron.Graph) ([]float32, error) {
	raw, err := g.Output(g.Outputs()[0].Name)
	if err != nil {
		return nil, err
	}
	out := make([]float32, len(raw)/2)
	vecops.BF16ToF32(out, raw)
	return out, nil
}

// Estimate runs both branches of the guidance in one call of the graph and takes the step on
// the host: x + (4 v_cond - 3 v_uncond) / total (the ONNX graph's).
func (s *amdStages) Estimate(ctx context.Context, x []float32, L int, emb []float32, T int, style []float32, step, total int) ([]float32, error) {
	p := s.size(T, L)
	if p == nil {
		return nil, ErrTooLong
	}
	m := p.meta
	S, P, seg, XP := m.Rows, m.P, m.Segment, m.Cols
	xs := make([]float32, S*XP)
	mask := make([]float32, S*512)
	for b := range 2 {
		for i := range L {
			r := b*seg + P + i
			for c := range Latent {
				xs[r*XP+c] = x[c*L+i]
			}
			for c := range 512 {
				mask[r*512+c] = 1
			}
		}
	}
	text := make([]float32, p.T*TextDim)
	for j := range T {
		for c := range TextDim {
			text[j*TextDim+c] = emb[c*T+j]
		}
	}
	st := make([]float32, m.Style*256)
	copy(st, style)
	g := p.est
	for _, in := range []struct {
		name string
		v    []float32
	}{{"x", xs}, {"text", text}, {"style", st}, {"mask", mask}, {"tb", s.host.timeRows(step, total)},
		{"qang", s.host.angles(L, m.Q)}, {"kang", s.host.angles(T, p.T)}} {
		if err := setBF16(g, in.name, in.v); err != nil {
			return nil, err
		}
	}
	if err := g.Run(ctx, map[string]int64{"nt": int64(T), "ns": 50}); err != nil {
		return nil, fmt.Errorf("supertonic: the flow: %w", err)
	}
	v, err := output(g)
	if err != nil {
		return nil, err
	}
	out := make([]float32, Latent*L)
	for i := range L {
		rc, ru := (P+i)*XP, (seg+P+i)*XP
		for c := range Latent {
			out[c*L+i] = x[c*L+i] + (4*v[rc+c]-3*v[ru+c])/float32(total)
		}
	}
	return out, nil
}

// Vocode gives the graph the latent's 6 L frames of 24 channels divided by the normalizer's
// scale, after 6 copies of the first (the causal edge padding).
func (s *amdStages) Vocode(ctx context.Context, x []float32, L int) ([]float32, error) {
	p := s.size(0, L)
	if p == nil {
		return nil, ErrTooLong
	}
	const sub, ch = 6, 24
	in := make([]float32, (sub*p.L+8)*ch)
	for l := range L {
		for k := range sub {
			r := sub + l*sub + k
			for c := range ch {
				in[r*ch+c] = x[(c*sub+k)*L+l] / 0.25
			}
		}
	}
	for r := range sub {
		copy(in[r*ch:(r+1)*ch], in[sub*ch:(sub+1)*ch])
	}
	if err := setBF16(p.voc, p.voc.Inputs()[0].Name, in); err != nil {
		return nil, err
	}
	if err := p.voc.Run(ctx, nil); err != nil {
		return nil, fmt.Errorf("supertonic: the vocoder: %w", err)
	}
	y, err := output(p.voc)
	if err != nil {
		return nil, err
	}
	return y[:L*Chunk], nil
}

func (s *amdStages) Close() error {
	for _, p := range s.sizes {
		for _, g := range []*iron.Graph{p.est, p.voc} {
			if g != nil {
				g.Close()
			}
		}
	}
	return s.text.Close()
}

// putBF16 writes xs as bf16 (to the nearest even) into b.
func putBF16(b []byte, xs []float32) {
	for i, v := range xs {
		bits := math.Float32bits(v)
		var h uint16
		if bits&0x7fffffff > 0x7f800000 {
			h = uint16(bits>>16) | 0x40
		} else {
			h = uint16((bits + 0x7fff + (bits>>16)&1) >> 16)
		}
		b[2*i], b[2*i+1] = byte(h), byte(h>>8)
	}
}
