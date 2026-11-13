package ocr

import (
	"bytes"
	"context"
	"encoding/binary"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
	"github.com/shibukawa/hwnpu/internal/onnx"
	"github.com/shibukawa/hwnpu/internal/vecops"
)

func readF32(t *testing.T, path string) []float32 {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(raw)/4)
	if err := binary.Read(bytes.NewReader(raw), binary.LittleEndian, out); err != nil {
		t.Fatal(err)
	}
	return out
}

func TestSortBoxes(t *testing.T) {
	b := [][4][2]float32{{{100, 52}}, {{10, 50}}, {{5, 100}}, {{50, 10}}}
	SortBoxes(b)
	got := [][2]float32{b[0][0], b[1][0], b[2][0], b[3][0]}
	want := [][2]float32{{50, 10}, {10, 50}, {100, 52}, {5, 100}}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%v, want %v", got, want)
		}
	}
}

// A line drawn as a block of probability 1 is found as its rectangle grown by the unclip.
func TestBoxes(t *testing.T) {
	r := &Reader{Thresh: 0.2, BoxThresh: 0.45, Unclip: 1.4}
	W, H := 64, 32
	prob := make([]float32, W*H)
	for y := 10; y < 20; y++ {
		for x := 5; x < 55; x++ {
			prob[y*W+x] = 1
		}
	}
	boxes := r.Boxes(prob, W, H, W, H)
	if len(boxes) != 1 {
		t.Fatalf("%d boxes", len(boxes))
	}
	// the pixels' centers span 49 x 9; d = 49 9 1.4 / (2 58) = 5.32
	b := boxes[0]
	if math.Abs(float64(b[0][0]-(5-5.32))) > 0.6 && b[0][0] != 0 || math.Abs(float64(b[0][1]-(10-5.32))) > 0.1 || math.Abs(float64(b[2][0]-(54+5.32))) > 0.1 || math.Abs(float64(b[2][1]-(19+5.32))) > 0.1 {
		t.Errorf("box %v", b)
	}
}

// A crop of a box twice as high as wide is turned a quarter counterclockwise.
func TestCropTurns(t *testing.T) {
	w, h := 20, 40
	rgb := make([]uint8, w*h*3)
	for y := range h {
		for x := range w {
			if y < h/2 {
				rgb[(y*w+x)*3] = 255 // the top half red
			}
		}
	}
	px, cw := Crop(rgb, w, h, [4][2]float32{{0, 0}, {20, 0}, {20, 40}, {0, 40}}, 320)
	if cw != 96 {
		t.Fatalf("width %d, want 96", cw)
	}
	// turned counterclockwise, the top half is on the left
	if px[(24*cw+5)*3] != 255 || px[(24*cw+90)*3] != 0 {
		t.Errorf("left %d, right %d", px[(24*cw+5)*3], px[(24*cw+90)*3])
	}
}

func run(t *testing.T, model string, shape []int, x []float32) ([]float32, []int) {
	t.Helper()
	g, err := onnx.Load(model)
	if err != nil {
		t.Fatal(err)
	}
	p := NewProgram(g, shape)
	dir := filepath.Join(t.TempDir(), "p.mlpackage")
	if _, err := llm.WritePart(dir, p, nil, llm.Options{Format: "fp16"}, nil); err != nil {
		t.Fatal(err)
	}
	var dev *hwnpu.Device
	if os.Getenv("HWNPU_OCR_NPU") == "1" {
		d, _, err := llm.OpenDevice(context.Background(), "", t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		defer d.Close()
		dev = d
	}
	prog, err := runner.Open(context.Background(), dev, dir)
	if err != nil {
		t.Fatal(err)
	}
	defer prog.Close()
	t.Logf("on the NPU: %v", prog.OnNPU())
	if pl, ok := prog.(interface {
		Placement() (bool, []string, string)
	}); ok && !prog.OnNPU() {
		known, off, why := pl.Placement()
		t.Logf("placement known %v: %v %s", known, off, why)
	}
	y, err := prog.Run(context.Background(), map[string][]float32{"x": x})
	if err != nil {
		t.Fatal(err)
	}
	return y, prog.Output()
}

// The detector's and the recognizer's programs (the CPU interpreter, fp16 weights) against ONNX
// Runtime (or on the NPU with HWNPU_OCR_NPU=1), from the reference's inputs
// (tools/ref/ref_detect_ocr.py ocrdet / ocrrec):
//
//	HWNPU_OCR_DET=det/inference.onnx HWNPU_OCR_DET_REF=ref/ocrdet-sample HWNPU_OCR_REC=rec/inference.onnx HWNPU_OCR_REC_REF=ref/ocrrec-line go test ./internal/vision/ocr -run Programs -v
func TestPrograms(t *testing.T) {
	if m, ref := os.Getenv("HWNPU_OCR_DET"), os.Getenv("HWNPU_OCR_DET_REF"); m != "" && ref != "" {
		want := readF32(t, ref+".prob.bin")
		x := readF32(t, ref+".x.bin")
		side := int(math.Sqrt(float64(len(want))))
		y, _ := run(t, m, []int{1, 3, side, side}, x)
		var worst float64
		var over, both int
		for i := range want {
			worst = math.Max(worst, math.Abs(float64(y[i]-want[i])))
			if want[i] > 0.2 || y[i] > 0.2 {
				over++
				if want[i] > 0.2 && y[i] > 0.2 {
					both++
				}
			}
		}
		t.Logf("detector %dx%d: largest difference %.4f, pixels over 0.2 in both %d of %d", side, side, worst, both, over)
		if float64(both) < 0.99*float64(over) {
			t.Errorf("the text pixels differ: %d of %d", both, over)
		}
	}
	if m, ref := os.Getenv("HWNPU_OCR_REC"), os.Getenv("HWNPU_OCR_REC_REF"); m != "" && ref != "" {
		want := readF32(t, ref+".prob.bin")
		x := readF32(t, ref+".x.bin")
		W := len(x) / (3 * Height)
		y, shape := run(t, m, []int{1, 3, Height, W}, x) // the logits before the softmax
		steps, n := shape[1], onnx.Len(shape[2:])
		t.Logf("output %v", shape)
		for s := range steps {
			row := y[s*n : (s+1)*n]
			mx := slices.Max(row)
			var z float64
			for _, v := range row {
				z += math.Exp(float64(v - mx))
			}
			for c := range row {
				row[c] = float32(math.Exp(float64(row[c]-mx)) / z)
			}
		}
		same := 0
		var worst float64
		for s := range steps {
			a, b := 0, 0
			for c := range n {
				if y[s*n+c] > y[s*n+a] {
					a = c
				}
				if want[s*n+c] > want[s*n+b] {
					b = c
				}
				worst = math.Max(worst, math.Abs(float64(y[s*n+c]-want[s*n+c])))
			}
			if a == b {
				same++
			}
		}
		t.Logf("recognizer width %d: %d of %d steps the same class, largest difference %.4f", W, same, steps, worst)
		if float64(same) < 0.99*float64(steps) || worst > 0.05 {
			t.Errorf("%d of %d steps differ, the largest difference %g", steps-same, steps, worst)
		}
	}
}

// TestRead reads a picture through Open (downloading into HWNPU_OCR_DIR) on the CPU
// interpreter and prints its lines; HWNPU_OCR_WANT, lines separated by |, must each be read.
// For a whole page against PaddleOCR's pipeline, compare hwnpu-image ocr's output with
// tools/ref/ref_ppocr.py's (tools/ref/compare_ocr.py).
func TestRead(t *testing.T) {
	dir, pic := os.Getenv("HWNPU_OCR_DIR"), os.Getenv("HWNPU_OCR_PICTURE")
	if dir == "" || pic == "" {
		t.Skip("HWNPU_OCR_DIR and HWNPU_OCR_PICTURE are not set")
	}
	r, err := Open(context.Background(), nil, Setup{Dir: dir,
		Det: Source{Repo: "PaddlePaddle/PP-OCRv6_small_det_onnx"}, Rec: Source{Repo: "PaddlePaddle/PP-OCRv6_small_rec_onnx"}})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	r.Logf = t.Logf
	f, err := os.Open(pic)
	if err != nil {
		t.Fatal(err)
	}
	img, _, err := image.Decode(f)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	lines, err := r.Read(context.Background(), img)
	if err != nil {
		t.Fatal(err)
	}
	var all []string
	for _, l := range lines {
		t.Logf("%.3f %q %v", l.Score, l.Text, l.Box)
		all = append(all, l.Text)
	}
	t.Logf("times %+v, %d recognizer calls", r.Times, r.Calls)
	if want := os.Getenv("HWNPU_OCR_WANT"); want != "" {
		for _, w := range strings.Split(want, "|") {
			found := false
			for _, l := range all {
				found = found || l == w
			}
			if !found {
				t.Errorf("no line %q", w)
			}
		}
	}
}

// sortKey orders fp16 values as their values.
func TestSortKey(t *testing.T) {
	vals := []float32{-65504, -3.5, -1, -0.25, 0, 0.125, 1, 2.5, 1000, 65504}
	for i := 1; i < len(vals); i++ {
		a, b := sortKey(vecops.F16Bits(vals[i-1])), sortKey(vecops.F16Bits(vals[i]))
		if a >= b {
			t.Errorf("%g (%#x) is not below %g (%#x)", vals[i-1], a, vals[i], b)
		}
	}
}
