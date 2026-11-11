package detect

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/shibukawa/hwnpu/internal/onnx"
)

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
