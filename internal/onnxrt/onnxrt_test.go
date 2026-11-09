package onnxrt

import (
	"errors"
	"testing"

	"github.com/shibukawa/hwnpu/internal/dl"
	"google.golang.org/protobuf/encoding/protowire"
)

func msg(fields ...func([]byte) []byte) []byte {
	var b []byte
	for _, f := range fields {
		b = f(b)
	}
	return b
}

func bytesField(num protowire.Number, v []byte) func([]byte) []byte {
	return func(b []byte) []byte { return protowire.AppendBytes(protowire.AppendTag(b, num, protowire.BytesType), v) }
}

func intField(num protowire.Number, v uint64) func([]byte) []byte {
	return func(b []byte) []byte { return protowire.AppendVarint(protowire.AppendTag(b, num, protowire.VarintType), v) }
}

// reluModel is an ONNX model of y = Relu(x), x and y declared (2, 3) float.
func reluModel() []byte {
	shape := msg(bytesField(1, msg(intField(1, 2))), bytesField(1, msg(intField(1, 3))))
	typ := msg(bytesField(1, msg(intField(1, 1), bytesField(2, shape))))
	value := func(name string) []byte { return msg(bytesField(1, []byte(name)), bytesField(2, typ)) }
	node := msg(bytesField(1, []byte("x")), bytesField(2, []byte("y")), bytesField(4, []byte("Relu")))
	graph := msg(bytesField(1, node), bytesField(2, []byte("g")), bytesField(11, value("x")), bytesField(12, value("y")),
		bytesField(13, value("y")))
	return msg(intField(1, 7), bytesField(7, graph), bytesField(8, msg(intField(2, 11))))
}

func runtimeOrSkip(t *testing.T) *Runtime {
	r, err := Load()
	if errors.Is(err, dl.ErrNotFound) {
		t.Skip("no ONNX Runtime (HWNPU_ONNXRUNTIME)")
	}
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func TestRun(t *testing.T) {
	r := runtimeOrSkip(t)
	t.Logf("ONNX Runtime %s from %s", r.Version, r.Path)
	free, err := FreeShapes(reluModel())
	if err != nil {
		t.Fatal(err)
	}
	s, err := r.Open(free, Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if len(s.Inputs) != 1 || s.Inputs[0] != "x" || len(s.Outputs) != 1 || s.Outputs[0] != "y" {
		t.Fatalf("inputs %v, outputs %v", s.Inputs, s.Outputs)
	}
	// (4, 5): not the declared (2, 3), which FreeShapes took away
	x := Tensor{Shape: []int64{4, 5}, Data: make([]float32, 20)}
	for i := range x.Data {
		x.Data[i] = float32(i) - 10
	}
	out, err := s.Run(map[string]Tensor{"x": x})
	if err != nil {
		t.Fatal(err)
	}
	y := out["y"]
	if len(y.Shape) != 2 || y.Shape[0] != 4 || y.Shape[1] != 5 {
		t.Fatalf("shape %v", y.Shape)
	}
	for i, v := range y.Data {
		if want := max(0, float32(i)-10); v != want {
			t.Fatalf("y[%d] = %v, want %v", i, v, want)
		}
	}
	// the declared shape is kept without FreeShapes
	s2, err := r.Open(reluModel(), Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer s2.Close()
	if _, err := s2.Run(map[string]Tensor{"x": x}); err == nil {
		t.Error("(4, 5) into a model of (2, 3): no error")
	}
}
