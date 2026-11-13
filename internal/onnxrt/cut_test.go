package onnxrt

import (
	"testing"

	"google.golang.org/protobuf/encoding/protowire"
)

// chainModel is an ONNX model of x -> Neg -> a -> Relu -> b -> Neg -> y, x and y declared
// float.
func chainModel() []byte {
	typ := msg(bytesField(1, msg(intField(1, 1))))
	value := func(name string) []byte { return msg(bytesField(1, []byte(name)), bytesField(2, typ)) }
	node := func(op, in, out string) []byte {
		return msg(bytesField(1, []byte(in)), bytesField(2, []byte(out)), bytesField(4, []byte(op)))
	}
	graph := msg(bytesField(1, node("Neg", "x", "a")), bytesField(1, node("Relu", "a", "b")), bytesField(1, node("Neg", "b", "y")),
		bytesField(2, []byte("g")), bytesField(11, value("x")), bytesField(12, value("y")))
	return msg(intField(1, 7), bytesField(7, graph), bytesField(8, msg(intField(2, 11))))
}

// TestCut checks the graph cut from a to b: only the Relu between them is kept, a is the
// input and b the output.
func TestCut(t *testing.T) {
	cut, err := Cut(chainModel(), []string{"a"}, []string{"b"})
	if err != nil {
		t.Fatal(err)
	}
	graph, err := fieldBytes(cut, modelGraph)
	if err != nil {
		t.Fatal(err)
	}
	var nodes, ins, outs int
	for b := graph; len(b) > 0; {
		num, typ, n := protowire.ConsumeTag(b)
		m := protowire.ConsumeFieldValue(num, typ, b[n:])
		switch num {
		case graphNode:
			nodes++
		case graphInput:
			ins++
		case graphOutput:
			outs++
		}
		b = b[n+m:]
	}
	if nodes != 1 || ins != 1 || outs != 1 {
		t.Fatalf("%d nodes, %d inputs, %d outputs; want 1 each", nodes, ins, outs)
	}
	r := runtimeOrSkip(t)
	s, err := r.Open(cut, Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	out, err := s.Run(map[string]Tensor{"a": {Shape: []int64{3}, Data: []float32{-1, 0, 2}}})
	if err != nil {
		t.Fatal(err)
	}
	if got := out["b"].Data; len(got) != 3 || got[0] != 0 || got[2] != 2 {
		t.Fatalf("b = %v, want [0 0 2]", got)
	}
}

// fieldBytes is the value of msg's length-delimited field num.
func fieldBytes(msg []byte, num protowire.Number) ([]byte, error) {
	for b := msg; len(b) > 0; {
		n, typ, k := protowire.ConsumeTag(b)
		if k < 0 {
			return nil, protowire.ParseError(k)
		}
		if n == num && typ == protowire.BytesType {
			v, _ := protowire.ConsumeBytes(b[k:])
			return v, nil
		}
		m := protowire.ConsumeFieldValue(n, typ, b[k:])
		b = b[k+m:]
	}
	return nil, nil
}
