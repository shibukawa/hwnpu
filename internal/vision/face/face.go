// Package face finds faces in a picture on AMD's NPU with YuNet (OpenCV Zoo's
// face_detection_yunet_2023mar, MIT), the graph tools/amd/iron_yunet.py exports: each face's
// box, score and five landmarks (the eyes, the nose tip and the mouth's corners).
//
// The host scales the picture to fit one of the export's sizes (without enlarging it), places
// it at the top left (zeros past it, as OpenCV pads to multiples of 32), runs the graph, then
// decodes the three strides' cells and suppresses the overlaps as OpenCV's FaceDetectorYN does.
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

// Detector is a loaded export. One size's graph is on the device at a time.
type Detector struct {
	Export
	dev *hwnpu.Device
	dir string
	cur *sized
	// Score is the least score a face is kept with (OpenCV's default 0.9), NMS the overlap
	// (intersection over union) above which the lower of two faces goes (0.3), TopK the most
	// faces kept before the suppression (5000).
	Score, NMS float32
	TopK       int
	// Size, when set, is the export's size every picture runs at (else Choose's).
	Size [2]int
	// Logf, when set, is told what is done.
	Logf func(format string, args ...any)
}

type sized struct {
	w, h int
	g    *iron.Graph
}

// Open reads the export in dir; a size's graph is loaded onto dev for the first picture
// that takes it.
func Open(ctx context.Context, dev *hwnpu.Device, dir string) (*Detector, error) {
	x, err := readExport(dir)
	if err != nil {
		return nil, err
	}
	return &Detector{Export: *x, dev: dev, dir: dir, Score: 0.9, NMS: 0.3, TopK: 5000}, nil
}

// Close releases the device memory.
func (d *Detector) Close() error {
	if d.cur == nil {
		return nil
	}
	err := d.cur.g.Close()
	d.cur = nil
	return err
}

// Choose is the export's size a w x h picture runs at: the one it is scaled down least to fit
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

func (d *Detector) use(ctx context.Context, w, h int) (*sized, error) {
	if d.cur != nil && d.cur.w == w && d.cur.h == h {
		return d.cur, nil
	}
	if !slices.Contains(d.Sizes, [2]int{w, h}) {
		return nil, fmt.Errorf("face: the export has no %dx%d (it has %v)", w, h, d.Sizes)
	}
	if err := d.Close(); err != nil {
		return nil, err
	}
	// A small graph called for each frame: its commands' runs are kept (made once).
	g, err := iron.OpenGraphWith(ctx, d.dev, filepath.Join(d.dir, fmt.Sprintf("%dx%d", w, h)), iron.GraphOptions{KeepRuns: true})
	if err != nil {
		return nil, err
	}
	d.cur = &sized{w: w, h: h, g: g}
	return d.cur, nil
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
	t0 := time.Now()
	W, H := d.Choose(b.Dx(), b.Dy())
	s, err := d.use(ctx, W, H)
	if err != nil {
		return nil, err
	}
	sc := fit(b.Dx(), b.Dy(), W, H)
	pw, ph := max(1, int(math.Round(float64(b.Dx())*sc))), max(1, int(math.Round(float64(b.Dy())*sc)))
	pw, ph = min(pw, W), min(ph, H)
	tl := time.Now()
	in := Input(lfm3vl.Resize(img, pw, ph, "bilinear"), pw, ph, W, H)
	if err := s.g.SetInput(s.g.Inputs()[0].Name, in); err != nil {
		return nil, err
	}
	t1 := time.Now()
	if err := s.g.Run(ctx, nil); err != nil {
		return nil, fmt.Errorf("face: %w", err)
	}
	t2 := time.Now()
	var faces []Face
	for i, h := range s.g.Outputs() {
		out, err := s.g.Output(h.Name)
		if err != nil {
			return nil, err
		}
		faces = append(faces, d.decode(out, W, H, d.Strides[i])...)
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
		d.Logf("face: %dx%d at %dx%d, load %.1f ms, input %.1f ms, graph %.1f ms, decode %.1f ms", pw, ph, W, H,
			tl.Sub(t0).Seconds()*1000, t1.Sub(tl).Seconds()*1000, t2.Sub(t1).Seconds()*1000, time.Since(t2).Seconds()*1000)
	}
	return faces, nil
}

// decode is the faces of one stride's cells (W/s x H/s, row major, Channels columns each:
// the class's and objectness' logits, the box's dx, dy, log w, log h, then the landmarks' x,
// y, all in strides from the cell) at least Score, as OpenCV's FaceDetectorYN::postProcess.
func (d *Detector) decode(out []byte, W, H, stride int) []Face {
	cols, rows := W/stride, H/stride
	C := d.Channels
	at := func(cell, ch int) float32 { return fromBF16(binary.LittleEndian.Uint16(out[2*(cell*C+ch):])) }
	st := float32(stride)
	var faces []Face
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			cell := r*cols + c
			cls := clamp01(sigmoid(at(cell, d.Head.Cls)))
			obj := clamp01(sigmoid(at(cell, d.Head.Obj)))
			score := float32(math.Sqrt(float64(cls * obj)))
			if score < d.Score {
				continue
			}
			bb := d.Head.BBox
			cx := (float32(c) + at(cell, bb)) * st
			cy := (float32(r) + at(cell, bb+1)) * st
			w := float32(math.Exp(float64(at(cell, bb+2)))) * st
			h := float32(math.Exp(float64(at(cell, bb+3)))) * st
			f := Face{X: cx - w/2, Y: cy - h/2, W: w, H: h, Score: score}
			for n := range f.Landmarks {
				f.Landmarks[n][0] = (at(cell, d.Head.Kps+2*n) + float32(c)) * st
				f.Landmarks[n][1] = (at(cell, d.Head.Kps+2*n+1) + float32(r)) * st
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
