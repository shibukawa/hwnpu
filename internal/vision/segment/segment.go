// Package segment cuts out the thing a point or a box picks in a picture with EfficientSAM
// (Meta's, Apache-2.0): the authors' ONNX exports of its image encoder and of its prompt's mask
// decoder (yunyangx/EfficientSAM, efficientsam_{ti,s}_{encoder,decoder}.onnx).
//
// The encoder (a ViT at 1024 x 1024) turns a picture into an embedding once: on AMD's NPU from an
// export of tools/amd/iron_efficientsam.py, on Intel's NPU as OpenVINO IR converted from its ONNX
// file (OpenProgram), else on the CPU with ONNX Runtime when its library is there, else with
// hwnpu's evaluator (internal/onnx; slow). The decoder turns the embedding and
// a prompt (points in or out of the thing, or a box) into three masks at the picture's size and
// the IoU it predicts for each, on the host with hwnpu's evaluator, per prompt.
package segment

import (
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"image"
	"io"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
	"github.com/shibukawa/hwnpu/internal/onnx"
	"github.com/shibukawa/hwnpu/internal/onnxrt"
)

// Size is the encoder's picture: every picture is stretched to Size x Size.
const Size = 1024

// Grid is the embedding's grid (16-pixel patches).
const Grid = Size / 16

var (
	mean = [3]float32{0.485, 0.456, 0.406}
	std  = [3]float32{0.229, 0.224, 0.225}
)

// Setup says what Fetch downloads: the encoder's and the decoder's files of a repository.
type Setup struct {
	Dir, Repo        string
	Encoder, Decoder string // file names in the repository
	// SHA256 are the files', by name, when the definitions pin them.
	SHA256 map[string]string
	Log    io.Writer
}

// Fetch downloads the two ONNX files into s.Dir unless they are there, checks their sha256 and
// returns their paths.
func Fetch(ctx context.Context, s Setup) (encoder, decoder string, err error) {
	if err := os.MkdirAll(s.Dir, 0o755); err != nil {
		return "", "", err
	}
	var paths [2]string
	for i, f := range []string{s.Encoder, s.Decoder} {
		p := filepath.Join(s.Dir, filepath.FromSlash(f))
		if err := chat.Fetch(ctx, s.Repo, f, p, cmp.Or[io.Writer](s.Log, io.Discard)); err != nil {
			return "", "", err
		}
		if want := s.SHA256[f]; want != "" {
			if err := checkSHA(p, want); err != nil {
				return "", "", err
			}
		}
		paths[i] = p
	}
	return paths[0], paths[1], nil
}

func checkSHA(path, want string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return err
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != want {
		return fmt.Errorf("segment: %s has sha256 %s, not %s", path, got, want)
	}
	return nil
}

// ExportVersion is the version of ExportNPU's exports (a key of the shared layers: chat.PinnedRecipe),
// bumped by hand when what it passes iron_efficientsam.py changes.
const ExportVersion = 1

// ExportNPU makes the export of tools/amd/iron_efficientsam.py of the encoder's ONNX file in out
// with IRON's python (chat.IRONPython).
func ExportNPU(ctx context.Context, python []string, name, encoder, out string, log io.Writer) error {
	return chat.RunIRON(ctx, python, name, "iron_efficientsam.py", []string{encoder, chat.ExportOut}, out, log)
}

// Model is an encoder and a decoder.
type Model struct {
	// Times are the last picture's resize and the encoder's run.
	Times [2]time.Duration
	dec   *onnx.Graph
	npu   *iron.Graph    // the encoder on AMD's NPU
	prog  runner.Program // or a program of OpenProgram (Apple's Neural Engine, Intel's NPU)
	ort   *ort           // or on the CPU with ONNX Runtime
	enc   *onnx.Graph    // or with hwnpu's evaluator
	x     *exportConfig  // the NPU export's model.json
	// progName is the program's file
	progName string
}

type exportConfig struct {
	Kind   string `json:"kind"`
	Size   int    `json:"size"`
	Tokens int    `json:"tokens"`
	Embed  int    `json:"embed"`
}

type ort struct {
	s *onnxrt.Session
}

// OpenNPU loads an export of ExportNPU (dir) onto dev, and the decoder's ONNX file.
func OpenNPU(ctx context.Context, dev *hwnpu.Device, dir, decoder string) (*Model, error) {
	b, err := os.ReadFile(filepath.Join(dir, "model.json"))
	if err != nil {
		return nil, err
	}
	var x exportConfig
	if err := json.Unmarshal(b, &x); err != nil {
		return nil, fmt.Errorf("segment: %s: %w", dir, err)
	}
	if x.Kind != "efficientsam" || x.Size != Size || x.Tokens != Grid*Grid || x.Embed != 256 {
		return nil, fmt.Errorf("segment: %s is not an export of tools/amd/iron_efficientsam.py", dir)
	}
	dec, err := onnx.Load(decoder)
	if err != nil {
		return nil, err
	}
	g, err := iron.OpenGraph(ctx, dev, dir)
	if err != nil {
		return nil, err
	}
	return &Model{dec: dec, npu: g, x: &x}, nil
}

// OpenCPU loads the encoder into ONNX Runtime on threads threads (0 or 1: the calling thread)
// when its library is there (onnxrt.Available), else into hwnpu's evaluator, and the decoder.
func OpenCPU(encoder, decoder string, threads int) (*Model, error) {
	dec, err := onnx.Load(decoder)
	if err != nil {
		return nil, err
	}
	m := &Model{dec: dec}
	if onnxrt.Available() {
		r, err := onnxrt.Load()
		if err != nil {
			return nil, err
		}
		raw, err := onnxrt.ReadModel(encoder)
		if err != nil {
			return nil, err
		}
		s, err := r.Open(raw, onnxrt.Options{Threads: threads})
		if err != nil {
			return nil, err
		}
		m.ort = &ort{s: s}
		return m, nil
	}
	if m.enc, err = onnx.Load(encoder); err != nil {
		return nil, err
	}
	return m, nil
}

// Where is where the encoder runs: "AMD's NPU", "the CPU (ONNX Runtime)" or "the CPU (hwnpu's
// evaluator, slow)".
func (m *Model) Where() string {
	switch {
	case m.npu != nil:
		return "AMD's NPU"
	case m.prog != nil:
		if m.prog.OnNPU() {
			return "the NPU (" + m.progName + ")"
		}
		return m.progName + ", NOT all on the NPU"
	case m.ort != nil:
		return "the CPU (ONNX Runtime)"
	}
	return "the CPU (hwnpu's evaluator, slow)"
}

func (m *Model) Close() error {
	if m.npu != nil {
		return m.npu.Close()
	}
	if m.prog != nil {
		return m.prog.Close()
	}
	if m.ort != nil {
		return m.ort.s.Close()
	}
	return nil
}

// Embedding is a picture's embedding (256, 64, 64) and the picture's size.
type Embedding struct {
	T    *onnx.Tensor
	W, H int
}

// Pixels are a picture's RGB in [0, 1], channels first (3, H, W).
func Pixels(img image.Image) []float32 {
	b := img.Bounds()
	W, H := b.Dx(), b.Dy()
	out := make([]float32, 3*W*H)
	for y := range H {
		for x := range W {
			r, g, bl, _ := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
			i := y*W + x
			out[i], out[W*H+i], out[2*W*H+i] = float32(r>>8)/255, float32(g>>8)/255, float32(bl>>8)/255
		}
	}
	return out
}

// Resize is px (3, h, w) stretched to (3, H, W) bilinearly as ONNX's Resize (linear,
// half_pixel, no antialiasing: torch's interpolate) does: the encoder's own first step.
func Resize(px []float32, w, h, W, H int) []float32 {
	type tap struct {
		i0, i1 int
		f      float32
	}
	taps := func(n, in int) []tap {
		t := make([]tap, n)
		s := float64(in) / float64(n)
		for o := range n {
			f := min(max((float64(o)+0.5)*s-0.5, 0), float64(in-1))
			i0 := int(f)
			t[o] = tap{i0, min(i0+1, in-1), float32(f - float64(i0))}
		}
		return t
	}
	ty, tx := taps(H, h), taps(W, w)
	out := make([]float32, 3*W*H)
	for c := range 3 {
		p, o := px[c*w*h:][:w*h], out[c*W*H:][:W*H]
		for y, a := range ty {
			r0, r1 := p[a.i0*w:][:w], p[a.i1*w:][:w]
			for x, b := range tx {
				top := r0[b.i0]*(1-b.f) + r0[b.i1]*b.f
				bot := r1[b.i0]*(1-b.f) + r1[b.i1]*b.f
				o[y*W+x] = top*(1-a.f) + bot*a.f
			}
		}
	}
	return out
}

// PatchRows are the NPU graph's input of a picture of Size x Size (3, Size, Size, in [0, 1]):
// normalized, each 16 x 16 patch's 3 x 16 x 16 values (channel, row, column) a row, in bf16.
func PatchRows(px []float32) []byte {
	out := make([]byte, 2*Grid*Grid*768)
	for gy := range Grid {
		for gx := range Grid {
			row := out[2*(gy*Grid+gx)*768:]
			k := 0
			for c := range 3 {
				for y := range 16 {
					for x := range 16 {
						v := (px[(c*Size+gy*16+y)*Size+gx*16+x] - mean[c]) / std[c]
						binary.LittleEndian.PutUint16(row[2*k:], bf16(v))
						k++
					}
				}
			}
		}
	}
	return out
}

// Embed runs the encoder on a picture.
func (m *Model) Embed(ctx context.Context, img image.Image) (*Embedding, error) {
	b := img.Bounds()
	W, H := b.Dx(), b.Dy()
	t0 := time.Now()
	px := Pixels(img)
	e := &Embedding{W: W, H: H}
	switch {
	case m.npu != nil:
		rows := PatchRows(Resize(px, W, H, Size, Size))
		t1 := time.Now()
		if err := m.npu.SetInput(m.npu.Inputs()[0].Name, rows); err != nil {
			return nil, err
		}
		if err := m.npu.Run(ctx, nil); err != nil {
			return nil, fmt.Errorf("segment: %w", err)
		}
		out, err := m.npu.Output(m.npu.Outputs()[0].Name)
		if err != nil {
			return nil, err
		}
		// (4096, 256) token-major, bf16 -> (1, 256, 64, 64)
		t := &onnx.Tensor{Shape: []int{1, 256, Grid, Grid}, F: make([]float64, 256*Grid*Grid), Type: onnx.Float}
		for p := range Grid * Grid {
			for c := range 256 {
				t.F[c*Grid*Grid+p] = float64(math.Float32frombits(uint32(binary.LittleEndian.Uint16(out[2*(p*256+c):])) << 16))
			}
		}
		e.T = t
		m.Times = [2]time.Duration{t1.Sub(t0), time.Since(t1)}
	case m.prog != nil:
		x := Resize(px, W, H, Size, Size)
		t1 := time.Now()
		y, err := m.prog.Run(ctx, map[string][]float32{"x": x})
		if err != nil {
			return nil, fmt.Errorf("segment: %w", err)
		}
		if len(y) != 256*Grid*Grid {
			return nil, fmt.Errorf("segment: the encoder's program gives %d values, not (1, 256, %d, %d)", len(y), Grid, Grid)
		}
		t := &onnx.Tensor{Shape: []int{1, 256, Grid, Grid}, F: make([]float64, len(y)), Type: onnx.Float}
		for i, v := range y {
			t.F[i] = float64(v)
		}
		e.T = t
		m.Times = [2]time.Duration{t1.Sub(t0), time.Since(t1)}
	case m.ort != nil:
		t1 := time.Now()
		out, err := m.ort.s.Run(map[string]onnxrt.Tensor{"batched_images": {Shape: []int64{1, 3, int64(H), int64(W)}, Data: px}})
		if err != nil {
			return nil, err
		}
		y, ok := out["image_embeddings"]
		if !ok || len(y.Data) != 256*Grid*Grid {
			return nil, fmt.Errorf("segment: the encoder gives no image_embeddings of (1, 256, %d, %d)", Grid, Grid)
		}
		t := &onnx.Tensor{Shape: []int{1, 256, Grid, Grid}, F: make([]float64, len(y.Data)), Type: onnx.Float}
		for i, v := range y.Data {
			t.F[i] = float64(v)
		}
		e.T = t
		m.Times = [2]time.Duration{t1.Sub(t0), time.Since(t1)}
	default:
		t1 := time.Now()
		in := &onnx.Tensor{Shape: []int{1, 3, H, W}, F: make([]float64, len(px)), Type: onnx.Float}
		for i, v := range px {
			in.F[i] = float64(v)
		}
		out, err := m.enc.Eval(map[string]*onnx.Tensor{"batched_images": in}, []string{"image_embeddings"})
		if err != nil {
			return nil, err
		}
		e.T = out["image_embeddings"]
		m.Times = [2]time.Duration{t1.Sub(t0), time.Since(t1)}
	}
	return e, nil
}

// Point is a prompt's point in the picture's pixels: Label 1 in the thing, 0 out of it, 2 and 3
// a box's top left and bottom right corners.
type Point struct {
	X, Y  float64
	Label int
}

// Mask is one of a prompt's masks: true where the thing is (W x H, row major) and the IoU the
// decoder predicts for it.
type Mask struct {
	W, H  int
	In    []bool
	Score float64
}

// Area is the mask's pixels; Box its bounding box (x1, y1, x2, y2; zero for an empty mask).
func (k *Mask) Area() int {
	n := 0
	for _, v := range k.In {
		if v {
			n++
		}
	}
	return n
}

func (k *Mask) Box() [4]int {
	x1, y1, x2, y2 := k.W, k.H, -1, -1
	for y := range k.H {
		for x := range k.W {
			if k.In[y*k.W+x] {
				x1, y1, x2, y2 = min(x1, x), min(y1, y), max(x2, x), max(y2, y)
			}
		}
	}
	if x2 < 0 {
		return [4]int{}
	}
	return [4]int{x1, y1, x2 + 1, y2 + 1}
}

// Segment gives the three masks of a prompt on an embedding, by the decoder's order (its
// candidates from the whole thing to a part).
func (m *Model) Segment(e *Embedding, prompt []Point) ([]Mask, error) {
	n := len(prompt)
	if n == 0 {
		return nil, fmt.Errorf("segment: a prompt of no points")
	}
	pts := &onnx.Tensor{Shape: []int{1, 1, n, 2}, F: make([]float64, 2*n), Type: onnx.Float}
	lab := &onnx.Tensor{Shape: []int{1, 1, n}, F: make([]float64, n), Type: onnx.Float}
	for i, p := range prompt {
		pts.F[2*i], pts.F[2*i+1], lab.F[i] = p.X, p.Y, float64(p.Label)
	}
	out, err := m.dec.Eval(map[string]*onnx.Tensor{
		"image_embeddings":     e.T,
		"batched_point_coords": pts,
		"batched_point_labels": lab,
		"orig_im_size":         {Shape: []int{2}, F: []float64{float64(e.H), float64(e.W)}, Type: onnx.Int64},
	}, []string{"output_masks", "iou_predictions"})
	if err != nil {
		return nil, err
	}
	masks, iou := out["output_masks"], out["iou_predictions"]
	if len(masks.Shape) != 5 || masks.Shape[3] != e.H || masks.Shape[4] != e.W {
		return nil, fmt.Errorf("segment: the decoder gives masks of %v for a picture of %dx%d", masks.Shape, e.W, e.H)
	}
	k := masks.Shape[2]
	res := make([]Mask, k)
	for j := range k {
		mk := Mask{W: e.W, H: e.H, In: make([]bool, e.W*e.H), Score: iou.F[j]}
		for i, v := range masks.F[j*e.W*e.H:][:e.W*e.H] {
			mk.In[i] = v > 0
		}
		res[j] = mk
	}
	return res, nil
}

func bf16(x float32) uint16 {
	bits := math.Float32bits(x)
	if bits&0x7fffffff > 0x7f800000 {
		return uint16(bits>>16) | 0x40
	}
	return uint16((bits + 0x7fff + (bits>>16)&1) >> 16)
}
