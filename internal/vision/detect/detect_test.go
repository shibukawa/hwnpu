package detect

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
	"github.com/shibukawa/hwnpu/internal/onnx"
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

func TestSuppress(t *testing.T) {
	objs := Suppress([]Object{
		{Box: [4]float32{0, 0, 10, 10}, Class: 0, Score: 0.5},
		{Box: [4]float32{1, 1, 11, 11}, Class: 0, Score: 0.9},
		{Box: [4]float32{1, 1, 11, 11}, Class: 1, Score: 0.4}, // another class: kept
		{Box: [4]float32{50, 50, 60, 60}, Class: 0, Score: 0.6},
	}, 0.65, 0)
	if len(objs) != 3 || objs[0].Score != 0.9 || objs[1].Score != 0.6 || objs[2].Class != 1 {
		t.Errorf("%+v", objs)
	}
	if got := Suppress([]Object{{Score: 0.9, Box: [4]float32{0, 0, 1, 1}}, {Score: 0.8, Box: [4]float32{5, 5, 6, 6}}}, 0.5, 1); len(got) != 1 {
		t.Errorf("max 1: %d", len(got))
	}
}

// TestRTMDet checks the program (on the CPU interpreter, or on the NPU with HWNPU_DETECT_NPU=1)
// against ONNX Runtime (tools/ref/ref_detect_ocr.py rtmdet): the scores and boxes of every
// anchor the graph lays out ($HWNPU_DETECT_REF.scores.bin, .boxes.bin from the reference's
// input .x.bin), and the objects.
//
//	HWNPU_DETECT=rtmdet-s-coco.onnx HWNPU_DETECT_REF=ref/rtmdet-football go test ./internal/vision/detect -run RTMDet -v
func TestRTMDet(t *testing.T) {
	path, ref := os.Getenv("HWNPU_DETECT"), os.Getenv("HWNPU_DETECT_REF")
	if path == "" || ref == "" {
		t.Skip("HWNPU_DETECT and HWNPU_DETECT_REF are not set")
	}
	g, err := onnx.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	p, err := NewProgram(g)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("heads %v", p.Outputs)
	dir := filepath.Join(t.TempDir(), "rtmdet.mlpackage")
	if _, err := llm.WritePart(dir, p, nil, llm.Options{Format: "fp16"}, nil); err != nil {
		t.Fatal(err)
	}
	var dev *hwnpu.Device
	if os.Getenv("HWNPU_DETECT_NPU") == "1" {
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
	y, err := prog.Run(context.Background(), map[string][]float32{"x": readF32(t, ref+".x.bin")})
	if err != nil {
		t.Fatal(err)
	}
	heads := p.Split(y)
	if hs := HeadShapes(len(COCO)); fmt.Sprint(hs) != fmt.Sprint(p.OutShapes()) {
		t.Fatalf("HeadShapes %v, the program's %v", hs, p.OutShapes())
	}
	// the anchors in the graph's order: stride by stride, cell by cell
	wantS, wantB := readF32(t, ref+".scores.bin"), readF32(t, ref+".boxes.bin")
	var maxS, maxB, maxBConf float64
	at := 0
	for i, s := range Strides {
		cls, box := heads[p.Outputs[2*i]], heads[p.Outputs[2*i+1]]
		C, H, W := cls.Shape[1], cls.Shape[2], cls.Shape[3]
		for cell := range H * W {
			for c := range C {
				got := 1 / (1 + math.Exp(-cls.F[c*H*W+cell]))
				maxS = math.Max(maxS, math.Abs(got-float64(wantS[(at+cell)*C+c])))
			}
			px, py := float64(cell%W*s), float64(cell/W*s)
			b := [4]float64{px - box.F[cell], py - box.F[H*W+cell], px + box.F[2*H*W+cell], py + box.F[3*H*W+cell]}
			best := 0.0
			for c := range C {
				best = math.Max(best, float64(wantS[(at+cell)*C+c]))
			}
			for k := range 4 {
				e := math.Abs(b[k] - float64(wantB[(at+cell)*4+k]))
				maxB = math.Max(maxB, e)
				if best >= 0.1 {
					maxBConf = math.Max(maxBConf, e)
				}
			}
		}
		at += H * W
	}
	t.Logf("largest differences over %d anchors: score %.4f, box %.2f px (%.2f px where a score is 0.1 or more)", at, maxS, maxB, maxBConf)
	// the graph with its weights rounded to fp16 and computed in fp32 (ONNX Runtime) is as far:
	// football-match.jpg, score 0.052, box 33.8 px (7.5 px where a score is 0.1 or more); the
	// NPU 0.063 and 5.5 px, the CPU interpreter in fp32 0.052 and 7.5 px
	if maxS > 0.08 || maxBConf > 8 {
		t.Errorf("score %g, box %g px", maxS, maxBConf)
	}
	d := &Detector{Score: 0.3, NMS: 0.65, Max: 300, Classes: COCO}
	objs := d.Decode(onnx.SplitFlat(y, HeadShapes(len(COCO))))
	dets, labels := readF32(t, ref+".dets.bin"), readF32(t, ref+".labels.bin")
	var want []Object
	for i := 0; i < len(labels); i++ {
		if dets[i*5+4] >= 0.3 {
			want = append(want, Object{Box: [4]float32(dets[i*5 : i*5+4]), Class: int(labels[i]), Score: dets[i*5+4]})
		}
	}
	for _, o := range objs {
		t.Logf("%-12s %.3f %v", o.Label, o.Score, o.Box)
	}
	// an object near the least score may be on either side of it
	if math.Abs(float64(len(objs)-len(want))) > 1 {
		t.Fatalf("%d objects, ONNX Runtime %d", len(objs), len(want))
	}
	// the same objects, in the order of scores that may swap where they are close
	for _, w := range want {
		found := false
		for _, o := range objs {
			if o.Class == w.Class && math.Abs(float64(o.Score-w.Score)) <= 0.03 && IoU(o.Box, w.Box) >= 0.9 {
				found = true
			}
		}
		if !found && w.Score >= 0.33 {
			t.Errorf("no object like ONNX Runtime's %+v", w)
		}
	}
}
