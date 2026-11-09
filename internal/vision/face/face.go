// Package face finds faces in a picture with YuNet (OpenCV Zoo's face_detection_yunet_2023mar,
// MIT): each face's box, score and five landmarks (the eyes, the nose tip and the mouth's
// corners). It runs on AMD's NPU (the graph tools/amd/iron_yunet.py exports, Open) or on the
// CPU (the ONNX file with ONNX Runtime, OpenCPU), where models.yaml's device puts it.
//
// The host scales the picture to fit one of the sizes (without enlarging it), places it at the
// top left (zeros past it, as OpenCV pads to multiples of 32), runs the network, then decodes
// the three strides' cells and suppresses the overlaps as OpenCV's FaceDetectorYN does.
package face

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image"
	"math"
	"os"
	"path/filepath"
	"slices"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3vl"
	"github.com/shibukawa/hwnpu/internal/onnxrt"
)

// Export is model.json of an export: its sizes, each a directory WxH.
type Export struct {
	Name     string   `json:"name"`
	Kind     string   `json:"kind"`
	Sizes    [][2]int `json:"sizes"`
	Strides  []int    `json:"strides"`
	Channels int      `json:"channels"` // an output cell's columns (64)
	Head     struct {
		Cls  int `json:"cls"`
		Obj  int `json:"obj"`
		BBox int `json:"bbox"`
		Kps  int `json:"kps"`
	} `json:"head"`
}

func readExport(dir string) (*Export, error) {
	b, err := os.ReadFile(filepath.Join(dir, "model.json"))
	if err != nil {
		return nil, err
	}
	var x Export
	if err := json.Unmarshal(b, &x); err != nil {
		return nil, fmt.Errorf("face: %w", err)
	}
	if x.Kind != "face-detect" || len(x.Sizes) == 0 || len(x.Strides) == 0 || x.Channels == 0 {
		return nil, fmt.Errorf("face: %s is not an export of tools/amd/iron_yunet.py", dir)
	}
	return &x, nil
}

// IsExport reports whether dir is an export of tools/amd/iron_yunet.py.
func IsExport(dir string) bool {
	_, err := readExport(dir)
	return err == nil
}

// Face is a face found: its box and landmarks in the picture's pixels.
type Face struct {
	X, Y, W, H float32 // the box: its top left corner and size
	Score      float32 // sqrt(class x objectness), 0..1
	// Landmarks in OpenCV's order: the right eye, the left eye (the person's: the right eye
	// is on the picture's left), the nose tip, the right and the left corner of the mouth.
	Landmarks [5][2]float32
}

// Detector is a loaded model: an export on the NPU (one size's graph on the device at a time)
// or the ONNX file on the CPU.
type Detector struct {
	Export
	// Device is where it runs: "npu" or "cpu".
	Device string
	run    runner
	// Score is the least score a face is kept with (OpenCV's default 0.9), NMS the overlap
	// (intersection over union) above which the lower of two faces goes (0.3), TopK the most
	// faces kept before the suppression (5000).
	Score, NMS float32
	TopK       int
	// Size, when set, is the size every picture runs at (else Choose's).
	Size [2]int
	// Logf, when set, is told what is done.
	Logf func(format string, args ...any)
	// OnClose, when set, is called by Close after the model is released (the caller's device).
	OnClose func()
}

// cells are one stride's outputs: per cell the class's and objectness' probabilities, the
// box's dx, dy, log w, log h (4 a cell) and the landmarks' x, y (10 a cell), in strides from
// the cell.
type cells struct {
	stride    int
	cls, obj  []float32
	bbox, kps []float32
}

// runner runs the network on a W x H picture's RGB pixels px (pw x ph at the top left).
type runner interface {
	run(ctx context.Context, px []uint8, pw, ph, W, H int) ([]cells, error)
	close() error
}

// Open reads the export (tools/amd/iron_yunet.py) in dir; a size's graph is loaded onto dev for
// the first picture that takes it.
func Open(ctx context.Context, dev *hwnpu.Device, dir string) (*Detector, error) {
	x, err := readExport(dir)
	if err != nil {
		return nil, err
	}
	return newDetector(*x, "npu", &npuRunner{x: *x, dev: dev, dir: dir}), nil
}

// CPUSizes are the sizes OpenCPU runs at by default: the exports' (iron_yunet.py's), for the
// same results. Detector.Size takes any multiple of 32.
var CPUSizes = [][2]int{{320, 256}, {640, 480}}

// OpenCPU loads the ONNX file (face_detection_yunet_2023mar.onnx; its fixed 640 x 640 shapes
// are freed) into ONNX Runtime on threads threads of the CPU (0: the calling thread).
func OpenCPU(path string, threads int) (*Detector, error) {
	rt, err := onnxrt.Load()
	if err != nil {
		return nil, err
	}
	b, err := onnxrt.ReadModel(path)
	if err != nil {
		return nil, err
	}
	if b, err = onnxrt.FreeShapes(b); err != nil {
		return nil, err
	}
	s, err := rt.Open(b, onnxrt.Options{Threads: threads})
	if err != nil {
		return nil, err
	}
	x := Export{Name: "YuNet", Kind: "face-detect", Sizes: CPUSizes, Strides: []int{8, 16, 32}}
	return newDetector(x, "cpu", &cpuRunner{s: s, strides: x.Strides}), nil
}

func newDetector(x Export, device string, r runner) *Detector {
	return &Detector{Export: x, Device: device, run: r, Score: 0.9, NMS: 0.3, TopK: 5000}
}

// Close releases the model (the device memory, or the session).
func (d *Detector) Close() error {
	err := d.run.close()
	if d.OnClose != nil {
		d.OnClose()
		d.OnClose = nil
	}
	return err
}

// Choose is the size a w x h picture runs at: the one it is scaled down least to fit
// (not enlarged: a smaller picture sits at the top left), the smallest of those.
func (d *Detector) Choose(w, h int) (int, int) {
	if d.Size != [2]int{} {
		return d.Size[0], d.Size[1]
	}
	best, bw, bh := -1.0, 0, 0
	for _, s := range d.Sizes {
		sc := fit(w, h, s[0], s[1])
		if sc > best+1e-9 || (math.Abs(sc-best) <= 1e-9 && s[0]*s[1] < bw*bh) {
			best, bw, bh = sc, s[0], s[1]
		}
	}
	return bw, bh
}

// fit is the scale that fits a w x h picture into W x H, at most 1.
func fit(w, h, W, H int) float64 {
	return min(1, float64(W)/float64(w), float64(H)/float64(h))
}

// Input is the graph's input of the RGB pixels px (pw x ph, raster order) for a W x H graph:
// (W H, 4) bfloat16, B, G, R over 255 and a one, the picture at the top left and zeros
// (with the one) past it.
func Input(px []uint8, pw, ph, W, H int) []byte {
	out := make([]byte, W*H*4*2)
	one := toBF16(1)
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			o := (y*W + x) * 8
			if x < pw && y < ph {
				i := (y*pw + x) * 3
				binary.LittleEndian.PutUint16(out[o:], toBF16(float32(px[i+2])/255))
				binary.LittleEndian.PutUint16(out[o+2:], toBF16(float32(px[i+1])/255))
				binary.LittleEndian.PutUint16(out[o+4:], toBF16(float32(px[i])/255))
			}
			binary.LittleEndian.PutUint16(out[o+6:], one)
		}
	}
	return out
}

// Detect finds the faces in img, the highest score first.
func (d *Detector) Detect(ctx context.Context, img image.Image) ([]Face, error) {
	b := img.Bounds()
	W, H := d.Choose(b.Dx(), b.Dy())
	if W%32 != 0 || H%32 != 0 || W <= 0 || H <= 0 {
		return nil, fmt.Errorf("face: %dx%d is not of multiples of 32", W, H)
	}
	sc := fit(b.Dx(), b.Dy(), W, H)
	pw, ph := max(1, int(math.Round(float64(b.Dx())*sc))), max(1, int(math.Round(float64(b.Dy())*sc)))
	pw, ph = min(pw, W), min(ph, H)
	t0 := time.Now()
	px := lfm3vl.Resize(img, pw, ph, "bilinear")
	t1 := time.Now()
	out, err := d.run.run(ctx, px, pw, ph, W, H)
	if err != nil {
		return nil, err
	}
	t2 := time.Now()
	var faces []Face
	for _, c := range out {
		faces = append(faces, d.decode(c, W, H)...)
	}
	faces = d.suppress(faces)
	inv := float32(1 / sc)
	for i := range faces {
		f := &faces[i]
		f.X, f.Y, f.W, f.H = f.X*inv, f.Y*inv, f.W*inv, f.H*inv
		for j := range f.Landmarks {
			f.Landmarks[j][0] *= inv
			f.Landmarks[j][1] *= inv
		}
	}
	if d.Logf != nil {
		d.Logf("face: %dx%d at %dx%d on the %s, resize %.1f ms, network %.1f ms, decode %.1f ms", pw, ph, W, H, d.Device,
			t1.Sub(t0).Seconds()*1000, t2.Sub(t1).Seconds()*1000, time.Since(t2).Seconds()*1000)
	}
	return faces, nil
}

// decode is the faces of one stride's cells (W/s x H/s, row major) at least Score, as OpenCV's
// FaceDetectorYN::postProcess.
func (d *Detector) decode(c cells, W, H int) []Face {
	cols, rows := W/c.stride, H/c.stride
	st := float32(c.stride)
	var faces []Face
	for r := 0; r < rows; r++ {
		for col := 0; col < cols; col++ {
			i := r*cols + col
			score := float32(math.Sqrt(float64(clamp01(c.cls[i]) * clamp01(c.obj[i]))))
			if score < d.Score {
				continue
			}
			bb := c.bbox[4*i : 4*i+4]
			cx := (float32(col) + bb[0]) * st
			cy := (float32(r) + bb[1]) * st
			w := float32(math.Exp(float64(bb[2]))) * st
			h := float32(math.Exp(float64(bb[3]))) * st
			f := Face{X: cx - w/2, Y: cy - h/2, W: w, H: h, Score: score}
			for n := range f.Landmarks {
				f.Landmarks[n][0] = (c.kps[10*i+2*n] + float32(col)) * st
				f.Landmarks[n][1] = (c.kps[10*i+2*n+1] + float32(r)) * st
			}
			faces = append(faces, f)
		}
	}
	return faces
}

// suppress keeps the faces in order of score, at most TopK considered, dropping any that
// overlaps a kept one by more than NMS (OpenCV's NMSBoxes).
func (d *Detector) suppress(faces []Face) []Face {
	slices.SortStableFunc(faces, func(a, b Face) int {
		switch {
		case a.Score > b.Score:
			return -1
		case a.Score < b.Score:
			return 1
		}
		return 0
	})
	if d.TopK > 0 && len(faces) > d.TopK {
		faces = faces[:d.TopK]
	}
	var kept []Face
	for _, f := range faces {
		ok := true
		for _, k := range kept {
			if IoU(f, k) > d.NMS {
				ok = false
				break
			}
		}
		if ok {
			kept = append(kept, f)
		}
	}
	return kept
}

// IoU is the intersection over union of two faces' boxes.
func IoU(a, b Face) float32 {
	x0, y0 := max(a.X, b.X), max(a.Y, b.Y)
	x1, y1 := min(a.X+a.W, b.X+b.W), min(a.Y+a.H, b.Y+b.H)
	if x1 <= x0 || y1 <= y0 {
		return 0
	}
	in := (x1 - x0) * (y1 - y0)
	return in / (a.W*a.H + b.W*b.H - in)
}

// npuRunner runs the export's graph of the size asked (one on the device at a time).
type npuRunner struct {
	x    Export
	dev  *hwnpu.Device
	dir  string
	w, h int
	g    *iron.Graph
}

func (r *npuRunner) close() error {
	if r.g == nil {
		return nil
	}
	err := r.g.Close()
	r.g = nil
	return err
}

func (r *npuRunner) run(ctx context.Context, px []uint8, pw, ph, W, H int) ([]cells, error) {
	if r.g == nil || r.w != W || r.h != H {
		if !slices.Contains(r.x.Sizes, [2]int{W, H}) {
			return nil, fmt.Errorf("face: the export has no %dx%d (it has %v)", W, H, r.x.Sizes)
		}
		if err := r.close(); err != nil {
			return nil, err
		}
		// A small graph called for each frame: its commands' runs are kept (made once).
		g, err := iron.OpenGraphWith(ctx, r.dev, filepath.Join(r.dir, fmt.Sprintf("%dx%d", W, H)), iron.GraphOptions{KeepRuns: true})
		if err != nil {
			return nil, err
		}
		r.g, r.w, r.h = g, W, H
	}
	if err := r.g.SetInput(r.g.Inputs()[0].Name, Input(px, pw, ph, W, H)); err != nil {
		return nil, err
	}
	if err := r.g.Run(ctx, nil); err != nil {
		return nil, fmt.Errorf("face: %w", err)
	}
	var out []cells
	for i, h := range r.g.Outputs() {
		b, err := r.g.Output(h.Name)
		if err != nil {
			return nil, err
		}
		out = append(out, r.x.cells(b, W, H, r.x.Strides[i]))
	}
	return out, nil
}

// cells reads one stride's graph output (cells x Channels bfloat16: the logits, box and
// landmarks at Head's columns).
func (x Export) cells(out []byte, W, H, stride int) cells {
	n := (W / stride) * (H / stride)
	C := x.Channels
	at := func(cell, ch int) float32 { return fromBF16(binary.LittleEndian.Uint16(out[2*(cell*C+ch):])) }
	c := cells{stride: stride, cls: make([]float32, n), obj: make([]float32, n), bbox: make([]float32, 4*n), kps: make([]float32, 10*n)}
	for i := 0; i < n; i++ {
		c.cls[i] = sigmoid(at(i, x.Head.Cls))
		c.obj[i] = sigmoid(at(i, x.Head.Obj))
		for k := 0; k < 4; k++ {
			c.bbox[4*i+k] = at(i, x.Head.BBox+k)
		}
		for k := 0; k < 10; k++ {
			c.kps[10*i+k] = at(i, x.Head.Kps+k)
		}
	}
	return c
}

// cpuRunner runs the ONNX model with ONNX Runtime.
type cpuRunner struct {
	s       *onnxrt.Session
	strides []int
}

func (r *cpuRunner) close() error { return r.s.Close() }

func (r *cpuRunner) run(ctx context.Context, px []uint8, pw, ph, W, H int) ([]cells, error) {
	// (1, 3, H, W): B, G, R planes in 0..255 (blobFromImage), zeros past the picture
	x := make([]float32, 3*W*H)
	for y := 0; y < ph; y++ {
		for i := 0; i < pw; i++ {
			p := (y*pw + i) * 3
			o := y*W + i
			x[o], x[W*H+o], x[2*W*H+o] = float32(px[p+2]), float32(px[p+1]), float32(px[p])
		}
	}
	out, err := r.s.Run(map[string]onnxrt.Tensor{r.s.Inputs[0]: {Shape: []int64{1, 3, int64(H), int64(W)}, Data: x}})
	if err != nil {
		return nil, fmt.Errorf("face: %w", err)
	}
	var cs []cells
	for _, st := range r.strides {
		get := func(k string) ([]float32, error) {
			t, ok := out[fmt.Sprintf("%s_%d", k, st)]
			if !ok {
				return nil, fmt.Errorf("face: the model has no output %s_%d", k, st)
			}
			return t.Data, nil
		}
		c := cells{stride: st}
		for _, f := range []struct {
			k string
			p *[]float32
		}{{"cls", &c.cls}, {"obj", &c.obj}, {"bbox", &c.bbox}, {"kps", &c.kps}} {
			if *f.p, err = get(f.k); err != nil {
				return nil, err
			}
		}
		cs = append(cs, c)
	}
	return cs, nil
}

func sigmoid(x float32) float32 { return float32(1 / (1 + math.Exp(-float64(x)))) }

func clamp01(x float32) float32 { return min(1, max(0, x)) }

func toBF16(x float32) uint16 {
	bits := math.Float32bits(x)
	if bits&0x7fffffff > 0x7f800000 {
		return uint16(bits>>16) | 0x40
	}
	return uint16((bits + 0x7fff + (bits>>16)&1) >> 16)
}

func fromBF16(h uint16) float32 { return math.Float32frombits(uint32(h) << 16) }
