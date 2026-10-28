package lfm3

import (
	"context"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/backend/coreml"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/types"
)

// Runs each of the first HWNPU_LFM3_PERLAYER layers (default 8) alone on the Neural Engine, fed
// with the reference hidden state entering it, and reports that layer's own fp16 error, so that
// errors do not accumulate across layers and the layer types can be compared.
func TestPerLayer(t *testing.T) {
	dir := os.Getenv("HWNPU_LFM3_DIR")
	if dir == "" {
		t.Skip("HWNPU_LFM3_DIR not set")
	}
	ctx := context.Background()
	infos, err := coreml.Provider{}.Enumerate(ctx)
	if err != nil || len(infos) == 0 {
		t.Skip("no Neural Engine")
	}
	cfg, err := LoadConfig(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	w, err := OpenWeights(filepath.Join(dir, "LFM3-1.7B.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	ids := readI32(t, filepath.Join(dir, "ref_bf16.ids.bin"))
	seq, d := len(ids), cfg.HiddenSize
	hidden := readF32(t, filepath.Join(dir, "ref_bf16.hidden.bin"))
	n := envInt("HWNPU_LFM3_PERLAYER", 8)
	format := os.Getenv("HWNPU_LFM3_FORMAT")
	if format == "" {
		format = "fp16"
	}
	dev, err := coreml.Provider{}.Open(ctx, infos[0], backend.Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	for i := 0; i < n; i++ {
		in := hidden[i*seq*d : (i+1)*seq*d]
		ref := hidden[(i+1)*seq*d : (i+2)*seq*d]
		xb := make([]byte, 2*len(in))
		for j, v := range in {
			binary.LittleEndian.PutUint16(xb[2*j:], mlpackage.Float16Bits(v))
		}
		out := filepath.Join(t.TempDir(), "l.mlpackage")
		pkg, err := mlpackage.Create(out)
		if err != nil {
			t.Fatal(err)
		}
		b := mlpackage.NewBuilder("CoreML8", pkg.Blobs)
		inF, y, err := Build(b, cfg, w, Options{Seq: seq, Start: i, Layers: 1, Format: format, Group: envInt("HWNPU_LFM3_GROUP", 16), ColBlock: envInt("HWNPU_LFM3_COLBLOCK", 0), ConvMode: os.Getenv("HWNPU_LFM3_CONV"), Quant: os.Getenv("HWNPU_LFM3_QUANT"), PostScale: float32(envInt("HWNPU_LFM3_POSTSCALE", 1))})
		if err != nil {
			t.Fatal(err)
		}
		if err := pkg.Finish(b.Finish(&mlpackage.Var{Name: y.Name}), 9, []mlpackage.Feature{inF}, []mlpackage.Feature{y}); err != nil {
			t.Fatal(err)
		}
		g, err := dev.Compile(ctx, types.GraphSource{Path: out})
		if err != nil {
			t.Fatalf("layer %d: %v", i, err)
		}
		p := g.Placement()
		got := run(t, dev, g, xb)
		g.Close()
		report(cfg.LayerTypes[i]+" layer "+itoa(i)+" alone (npu ops "+itoa(p.NPUOps)+", elsewhere "+itoa(len(p.Ops))+")", got, ref, t)
	}
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	s := ""
	for ; i > 0; i /= 10 {
		s = string(rune('0'+i%10)) + s
	}
	return s
}
