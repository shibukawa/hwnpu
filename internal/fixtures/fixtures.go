// Package fixtures embeds the small models hwnpu uses for tests and for cmd/report, so they are
// available wherever the module is built. gen_models.py documents and regenerates them.
package fixtures

import (
	"embed"
	"encoding/binary"
	"io/fs"
	"math"
	"os"
	"path/filepath"
)

//go:embed coreml openvino linear_relu_y.f32 mlp_small_y.f32
var files embed.FS

// FS returns the embedded files.
func FS() fs.FS { return files }

// Shape of the "linear_relu" models: y = relu(x @ W^T + b), x (Batch, In), y (Batch, Out).
const (
	Batch = 64
	In    = 512
	Out   = 512
)

// Paths inside the embedded tree (and inside a directory written by Extract).
const (
	CoreMLFP16       = "coreml/linear_relu_fp16.mlpackage"
	CoreMLFP32       = "coreml/linear_relu_fp32.mlpackage"
	CoreMLSmall      = "coreml/mlp_small.mlpackage"     // 3 layer MLP, batch 1: placed on the CPU
	CoreMLSmallANE   = "coreml/mlp_small_ane.mlpackage" // the same, padded to batch 64 by ane_pad.py
	OpenVINOLinear   = "openvino/linear_relu.xml"
	OpenVINOAffine   = "openvino/affine_relu.xml"
	referenceOutputs = "linear_relu_y.f32"
)

// LinearInput returns the input of the linear_relu models: x[i] = sin(i * 0.01).
func LinearInput() []float32 {
	x := make([]float32, Batch*In)
	for i := range x {
		x[i] = float32(math.Sin(float64(i) * 0.01))
	}
	return x
}

// LinearOutput returns the reference output of the linear_relu models, computed in fp32.
func LinearOutput() []float32 {
	raw, err := files.ReadFile(referenceOutputs)
	if err != nil {
		panic(err)
	}
	y := make([]float32, len(raw)/4)
	for i := range y {
		y[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	return y
}

// SmallInput is the input of the mlp_small models, shape (1, 128): x[i] = sin(i * 0.01).
func SmallInput() []float32 {
	x := make([]float32, 128)
	for i := range x {
		x[i] = float32(math.Sin(float64(i) * 0.01))
	}
	return x
}

// SmallOutput returns the fp32 reference output of the mlp_small models, shape (1, 128).
func SmallOutput() []float32 { return readF32("mlp_small_y.f32") }

func readF32(name string) []float32 {
	raw, err := files.ReadFile(name)
	if err != nil {
		panic(err)
	}
	y := make([]float32, len(raw)/4)
	for i := range y {
		y[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	return y
}

// AffineInput is an input of the affine_relu model, y = relu(2x - 1) with x of shape (1, 8).
var AffineInput = []float32{-1, -0.5, 0, 0.25, 0.5, 1, 2, 3}

// AffineOutput computes the expected output of the affine_relu model.
func AffineOutput() []float32 {
	y := make([]float32, len(AffineInput))
	for i, x := range AffineInput {
		y[i] = max(2*x-1, 0)
	}
	return y
}

// Extract writes the embedded tree under dir. Frameworks such as Core ML need real files.
func Extract(dir string) error {
	return fs.WalkDir(files, ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		target := filepath.Join(dir, filepath.FromSlash(path))
		if d.IsDir() {
			return os.MkdirAll(target, 0o755)
		}
		data, err := files.ReadFile(path)
		if err != nil {
			return err
		}
		return os.WriteFile(target, data, 0o644)
	})
}
