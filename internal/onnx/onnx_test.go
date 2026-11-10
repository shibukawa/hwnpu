package onnx

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/milcpu"
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

func relErr(got []float32, want []float32) float64 {
	var se, sw float64
	for i := range want {
		d := float64(got[i]) - float64(want[i])
		se += d * d
		sw += float64(want[i]) * float64(want[i])
	}
	return math.Sqrt(se / sw)
}

// The host evaluator follows numpy's broadcasting and ONNX's slicing.
func TestEvalOps(t *testing.T) {
	a := &Tensor{Shape: []int{2, 3}, F: []float64{1, 2, 3, 4, 5, 6}, Type: Float}
	b := &Tensor{Shape: []int{3}, F: []float64{10, 20, 30}, Type: Float}
	out, err := EvalNode(&Node{Op: "Add"}, []*Tensor{a, b})
	if err != nil || len(out[0].F) != 6 || out[0].F[4] != 25 {
		t.Fatalf("Add: %v %v", out, err)
	}
	s, err := EvalNode(&Node{Op: "Slice"}, []*Tensor{a, {Shape: []int{1}, F: []float64{-2}}, {Shape: []int{1}, F: []float64{9.2e18}}, {Shape: []int{1}, F: []float64{1}}})
	if err != nil || len(s[0].F) != 4 || s[0].F[0] != 2 || s[0].F[3] != 6 {
		t.Fatalf("Slice: %v %v", s, err)
	}
	m, err := EvalNode(&Node{Op: "MatMul"}, []*Tensor{a, {Shape: []int{3, 1}, F: []float64{1, 0, -1}}})
	if err != nil || m[0].F[0] != -2 || m[0].F[1] != -2 {
		t.Fatalf("MatMul: %v %v", m, err)
	}
	k, err := EvalNode(&Node{Op: "TopK", Attrs: map[string]Attr{}}, []*Tensor{{Shape: []int{1, 4}, F: []float64{3, 9, 1, 9}}, {Shape: []int{1}, F: []float64{2}}})
	if err != nil || k[1].F[0] != 1 || k[1].F[1] != 3 {
		t.Fatalf("TopK: %v %v", k, err)
	}
	n, err := EvalNode(&Node{Op: "NonMaxSuppression", Attrs: map[string]Attr{}}, []*Tensor{
		{Shape: []int{1, 3, 4}, F: []float64{0, 0, 10, 10, 1, 1, 11, 11, 20, 20, 30, 30}},
		{Shape: []int{1, 1, 3}, F: []float64{0.9, 0.8, 0.7}},
		{Shape: []int{1}, F: []float64{10}}, {Shape: []int{1}, F: []float64{0.5}}, {Shape: []int{1}, F: []float64{0}},
	})
	if err != nil || n[0].Shape[0] != 2 || n[0].F[2] != 0 || n[0].F[5] != 2 {
		t.Fatalf("NonMaxSuppression: %v %v", n, err)
	}
}

// With RTMO-s (HWNPU_RTMO = its model.onnx) and a reference of rtmo_ref.py (HWNPU_RTMO_REF), the
// program of the graph up to the head's outputs gives the reference's on the CPU interpreter in
// fp16, and the graph's tail evaluated on the host from them gives its people and keypoints.
func TestRTMO(t *testing.T) {
	path, ref := os.Getenv("HWNPU_RTMO"), os.Getenv("HWNPU_RTMO_REF")
	if path == "" || ref == "" {
		t.Skip("HWNPU_RTMO and HWNPU_RTMO_REF are not set")
	}
	g, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	var meta struct {
		Cut    []string         `json:"cut"`
		Shapes map[string][]int `json:"shapes"`
	}
	raw, err := os.ReadFile(ref + ".json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &meta); err != nil {
		t.Fatal(err)
	}
	p := &Program{G: g, Inputs: []ValueInfo{{Name: "input", Shape: []int{1, 3, 640, 640}}}, Outputs: meta.Cut, Axis: -1}
	dir := filepath.Join(t.TempDir(), "rtmo.mlpackage")
	if _, err := llm.WritePart(dir, p, nil, llm.Options{Format: "fp16"}, nil); err != nil {
		t.Fatal(err)
	}
	prog, f, err := milcpu.OpenPackage(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	x := readF32(t, ref+".input.bin")
	res, err := prog.Run(map[string]milcpu.Tensor{"x": {Shape: []int{1, 3, 640, 640}, Data: x}}, milcpu.Options{})
	if err != nil {
		t.Fatal(err)
	}
	y := res["y"]
	given := p.Folded()
	at := 0
	width := y.Shape[len(y.Shape)-1]
	rows := len(y.Data) / width
	for i, name := range meta.Cut {
		w := p.Widths()[i]
		want := readF32(t, ref+".cut"+string(rune('0'+i))+".bin")
		got := make([]float32, rows*w)
		for r := range rows {
			copy(got[r*w:(r+1)*w], y.Data[r*width+at:r*width+at+w])
		}
		e := relErr(got, want)
		t.Logf("%s %v: relative error %.3g", name, meta.Shapes[name], e)
		if e > 0.02 {
			t.Errorf("%s: relative error %g", name, e)
		}
		tt := &Tensor{Shape: meta.Shapes[name], Type: Float}
		for _, v := range got {
			tt.F = append(tt.F, float64(v))
		}
		given[name] = tt
		at += w
	}
	out, err := g.EvalShapes(given, p.Shapes(), []string{"dets", "keypoints"})
	if err != nil {
		t.Fatal(err)
	}
	dets, kpts := out["dets"], out["keypoints"]
	wantD, wantK := readF32(t, ref+".dets.bin"), readF32(t, ref+".keypoints.bin")
	t.Logf("dets %v (want %v), keypoints %v", dets.Shape, meta.Shapes["dets"], kpts.Shape)
	if len(dets.F) != len(wantD) || len(kpts.F) != len(wantK) {
		t.Fatalf("%d people, want %d", dets.Shape[1], meta.Shapes["dets"][1])
	}
	var box, pt, score float64
	for i := range dets.F {
		d := math.Abs(dets.F[i] - float64(wantD[i]))
		if i%5 == 4 {
			score = math.Max(score, d)
		} else {
			box = math.Max(box, d)
		}
	}
	// the visible keypoints (score 0.3 and over, as the model card's example keeps them)
	var hidden float64
	for i := 0; i < len(kpts.F); i += 3 {
		d := math.Hypot(kpts.F[i]-float64(wantK[i]), kpts.F[i+1]-float64(wantK[i+1]))
		if wantK[i+2] >= 0.3 {
			pt = math.Max(pt, d)
		} else {
			hidden = math.Max(hidden, d)
		}
	}
	t.Logf("largest differences: box %.3g px, score %.3g, visible keypoint %.3g px (hidden %.3g)", box, score, pt, hidden)
	// the graph with its weights rounded to fp16 and computed in fp32 (onnx's reference
	// evaluator) is as far from the reference: box 1.71 px, visible keypoint 3.37 px
	if box > 2.5 || score > 0.02 || pt > 4 {
		t.Errorf("box %g px, score %g, keypoint %g px", box, score, pt)
	}
}

// The cut of RTMO at its top-k and non-maximum suppression is the head's outputs.
func TestRTMOCut(t *testing.T) {
	path := os.Getenv("HWNPU_RTMO")
	if path == "" {
		t.Skip("HWNPU_RTMO is not set")
	}
	g, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	cut := g.Cut("input", map[string]bool{"TopK": true, "NonMaxSuppression": true}, []string{"dets", "keypoints"})
	t.Logf("cut: %q", cut)
}
