package vision

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3vl"
	"github.com/shibukawa/hwnpu/internal/milcpu"
)

const tinyVL = `{"architectures": ["Lfm3VlForConditionalGeneration"], "model_type": "lfm3_vl",
	"projector_hidden_act": "gelu", "projector_bias": true, "projector_layer_norm_eps": 1e-6, "projector_hidden_size": 32,
	"downsample_factor": 2,
	"text_config": {"model_type": "lfm3", "hidden_size": 64},
	"vision_config": {"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2,
		"num_key_value_heads": 1, "head_dim": 8, "patch_size": 4, "temporal_patch_size": 2, "in_channels": 3,
		"conv_kernel_size": 3, "norm_eps": 1e-5, "layer_types": ["conv", "full_attention"],
		"rope_parameters": {"rope_theta": 100}}}`

const tinyProcessor = `{"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "rescale_factor": 0.00392156862745098,
	"resample": 2, "size": {"shortest_edge": 256, "longest_edge": 1024}}`

// writeTiny writes a tiny LFM3-VL checkpoint's config files to a directory and converts its
// vision encoder for the given image sizes, as hwnpu-convert does; it returns the directory and
// the packages.
func writeTiny(t *testing.T, sizes ...string) (string, []string) {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(tinyVL), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "preprocessor_config.json"), []byte(tinyProcessor), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := lfm3vl.ParseConfig([]byte(tinyVL))
	if err != nil {
		t.Fatal(err)
	}
	var pkgs []string
	for _, s := range sizes {
		out := filepath.Join(dir, "vision-"+s+".mlpackage")
		opts := llm.Options{Format: "fp16", Params: map[string]string{"image": s}}
		path, err := llm.Convert(lfm3vl.Arch, cfg, llm.NewSynthetic(cfg), out, "none", 0, opts, io.Discard)
		if err != nil {
			t.Fatal(err)
		}
		pkgs = append(pkgs, path)
	}
	return dir, pkgs
}

func testImage(w, h int) image.Image {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, color.NRGBA{uint8(x * 9), uint8(y * 5), uint8(x*y + 30), 255})
		}
	}
	return img
}

// Encode picks the program of the closest aspect ratio and gives what that program gives for the
// processor's patches.
func TestEncoder(t *testing.T) {
	dir, pkgs := writeTiny(t, "16x16", "32x16", "16x32")
	e, err := Load(context.Background(), nil, dir, pkgs)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	cfg, _ := lfm3vl.ParseConfig([]byte(tinyVL))
	proc, err := lfm3vl.LoadProcessor(dir, cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		w, h, pw, ph int
	}{
		{40, 20, 32, 16}, // wide
		{18, 45, 16, 32}, // tall
		{30, 28, 16, 16}, // about square
	} {
		img := testImage(c.w, c.h)
		rows, n, err := e.Encode(context.Background(), img)
		if err != nil {
			t.Fatal(err)
		}
		if n != cfg.Tokens(c.pw, c.ph) || len(rows) != n*cfg.Text.HiddenSize {
			t.Errorf("%dx%d: %d tokens (%d values), want the %dx%d program's %d", c.w, c.h, n, len(rows), c.pw, c.ph, cfg.Tokens(c.pw, c.ph))
			continue
		}
		prog, f, err := milcpu.OpenPackage(filepath.Join(dir, fmt.Sprintf("vision-%dx%d.mlpackage", c.pw, c.ph)))
		if err != nil {
			t.Fatal(err)
		}
		res, err := prog.Run(map[string]milcpu.Tensor{"x": {Shape: prog.Inputs()["x"], Data: proc.Patches(img, c.pw, c.ph)}}, milcpu.Options{})
		f.Close()
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range res["y"].Data {
			if math.Abs(float64(v-rows[i])) > 1e-6 {
				t.Errorf("%dx%d: row value %d is %g, the program gives %g", c.w, c.h, i, rows[i], v)
				break
			}
		}
	}
	if _, err := Load(context.Background(), nil, dir, []string{filepath.Join(dir, "missing.mlpackage")}); err == nil {
		t.Error("a missing package loaded")
	}
}

func TestDecodeDataURL(t *testing.T) {
	var b bytes.Buffer
	png.Encode(&b, testImage(5, 3))
	img, err := DecodeDataURL("data:image/png;base64," + base64.StdEncoding.EncodeToString(b.Bytes()))
	if err != nil || img.Bounds().Dx() != 5 || img.Bounds().Dy() != 3 {
		t.Errorf("decoded %v, %v", img, err)
	}
	for _, u := range []string{"https://example.com/a.png", "data:image/png,abc", "data:image/png;base64,!!", "data:image/png;base64,AAAA"} {
		if _, err := DecodeDataURL(u); err == nil {
			t.Errorf("%s decoded", u)
		}
	}
}
