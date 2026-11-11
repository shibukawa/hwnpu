package detect

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/shibukawa/hwnpu/internal/onnx"
)

// TestCNNLayout checks the grids' layout: the input's pixel (y, x) at band + y wp + P + x with
// its ones channel, and an output grid's planes read back channels first.
func TestCNNLayout(t *testing.T) {
	x := &CNNExport{Width: 3, Height: 2, P: 2}
	x.Input.Wp, x.Input.Cp = 16, 4
	x.Input.Rows = 256
	px := make([]float32, 3*2*3)
	for i := range px {
		px[i] = float32(i)
	}
	g := x.InputGrid(px)
	at := func(b []byte, r, c, cp int) float32 {
		return math.Float32frombits(uint32(binary.LittleEndian.Uint16(b[2*(r*cp+c):])) << 16)
	}
	band := 2*16 + 2
	r := band + 1*16 + 2 + 2 // pixel (1, 2)
	if at(g, r, 0, 4) != px[(0*2+1)*3+2] || at(g, r, 2, 4) != px[(2*2+1)*3+2] || at(g, r, 3, 4) != 1 || at(g, r-1-2, 3, 4) != 0 {
		t.Errorf("pixel (1, 2): %v %v %v; the pad before row 1 %v", at(g, r, 0, 4), at(g, r, 2, 4), at(g, r, 3, 4), at(g, r-3, 3, 4))
	}
	o := CNNOutput{C: 2, Cp: 64, H: 2, W: 3, Wp: 16, Band: band}
	b := make([]byte, 2*256*64)
	for y := range 2 {
		for i := range 3 {
			for c := range 2 {
				binary.LittleEndian.PutUint16(b[2*((band+y*16+2+i)*64+c):], bf16(float32(100*c+10*y+i)))
			}
		}
	}
	p := o.Planes(b, 2)
	if p[0*6+1*3+2] != 12 || p[1*6+0*3+1] != 101 {
		t.Errorf("planes %v", p)
	}
}

// TestNPUOutputs checks the heads of RTMDet-m's and -s's graphs (when downloaded): the classes'
// logits as they are, the boxes before exp and the stride (m) or before the stride (s).
func TestNPUOutputs(t *testing.T) {
	dir := os.Getenv("HWNPU_RTMDET_DIR")
	if dir == "" {
		d, _ := os.UserCacheDir()
		dir = filepath.Join(d, "hwnpu", "models", "RTMDet-ONNX")
	}
	for file, want := range map[string][]float64{"rtmdet-m-coco.onnx": {0, -8, 0, -16, 0, -32}, "rtmdet-s-coco.onnx": {0, 8, 0, 16, 0, 32}} {
		path := filepath.Join(dir, file)
		if _, err := os.Stat(path); err != nil {
			t.Logf("%s: not downloaded", file)
			continue
		}
		g, err := onnx.Load(path)
		if err != nil {
			t.Fatal(err)
		}
		names, scales, err := NPUOutputs(g)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(scales, want) || len(names) != 6 {
			t.Errorf("%s: %v %v, want scales %v", file, names, scales, want)
		}
	}
}
