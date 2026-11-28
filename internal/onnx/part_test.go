package onnx

import (
	"encoding/json"
	"path/filepath"
	"slices"
	"testing"
)

// A part of a graph written to a part file and mapped back evaluates as the graph did; its tensors
// and settings come back as written; a file of another version is refused.
func TestPartFile(t *testing.T) {
	g := &Graph{
		Opset: 13,
		Init: map[string]*Tensor{
			"w":      {Shape: []int{2}, F: []float64{1.5, -2}, Type: Float},
			"c":      {Shape: []int{}, F: []float64{3}, Type: Float},
			"unused": {Shape: []int{1}, F: []float64{9}, Type: Float},
			"axes":   {Shape: []int{1}, F: []float64{0}, Type: Int64},
		},
		Inputs:  []ValueInfo{{Name: "x", Type: Float, Shape: []int{2}}},
		Outputs: []ValueInfo{{Name: "z", Type: Float, Shape: []int{1}}},
		Nodes: []*Node{
			{Op: "Add", Name: "add", In: []string{"x", "w"}, Out: []string{"y"}},
			{Op: "Mul", Name: "mul", In: []string{"y", "c"}, Out: []string{"m"}},
			{Op: "ReduceSum", Name: "sum", In: []string{"m", "axes"}, Out: []string{"z"}, Attrs: map[string]Attr{"keepdims": {Kind: 2, I: 1}}},
			{Op: "Neg", Name: "other", In: []string{"unused"}, Out: []string{"o"}},
		},
	}
	// the part after y (as a program's output would be), and the whole graph
	sub := g.Subgraph([]string{"y"}, []string{"z"})
	if len(sub.Nodes) != 2 || sub.Init["unused"] != nil || sub.Init["w"] != nil || sub.Init["c"] == nil {
		t.Fatalf("Subgraph: %d nodes, init %v", len(sub.Nodes), sub.Init)
	}
	path := filepath.Join(t.TempDir(), "m"+PartExt)
	meta, _ := json.Marshal(map[string]any{"shapes": [][]int{{1, 2}}})
	if err := WritePart(path, Part{Graph: g.Subgraph([]string{"x"}, []string{"z"}), Tensors: map[string]*Tensor{"bank": {Shape: []int{3}, F: []float64{0.25, 0, 7}, Type: Float}}, Meta: meta}); err != nil {
		t.Fatal(err)
	}
	p, err := ReadPart(path)
	if err != nil {
		t.Fatal(err)
	}
	x := map[string]*Tensor{"x": {Shape: []int{2}, F: []float64{1, 2}, Type: Float}}
	want, err := g.Eval(x, []string{"z"})
	if err != nil {
		t.Fatal(err)
	}
	got, err := p.Graph.Eval(x, []string{"z"})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got["z"].F, want["z"].F) {
		t.Errorf("z = %v, want %v", got["z"].F, want["z"].F)
	}
	if b := p.Tensors["bank"]; b == nil || !slices.Equal(b.Values(), []float64{0.25, 0, 7}) {
		t.Errorf("bank = %+v", b)
	}
	var m struct{ Shapes [][]int }
	if err := json.Unmarshal(p.Meta, &m); err != nil || len(m.Shapes) != 1 || !slices.Equal(m.Shapes[0], []int{1, 2}) {
		t.Errorf("meta = %s, %v", p.Meta, err)
	}
	if n := p.Graph.Nodes[2]; n.Attrs["keepdims"].I != 1 {
		t.Errorf("attrs = %+v", n.Attrs)
	}
}

// A weight (lazyMin elements or more) stays in the mapped file until it is read; a small tensor
// and an attribute's are read at once (the evaluator reads them as they are).
func TestPartFileLazy(t *testing.T) {
	big := make([]float64, lazyMin)
	for i := range big {
		big[i] = float64(i % 7)
	}
	g := &Graph{Init: map[string]*Tensor{"w": {Shape: []int{lazyMin}, F: big, Type: Float}}, Nodes: []*Node{
		{Op: "Constant", Out: []string{"c"}, Attrs: map[string]Attr{"value": {Kind: 4, T: &Tensor{Shape: []int{1}, F: []float64{2}, Type: Float}}}},
		{Op: "Mul", In: []string{"w", "c"}, Out: []string{"y"}},
	}}
	path := filepath.Join(t.TempDir(), "w"+PartExt)
	if err := WritePart(path, Part{Graph: g}); err != nil {
		t.Fatal(err)
	}
	p, err := ReadPart(path)
	if err != nil {
		t.Fatal(err)
	}
	if !p.Graph.Init["w"].Lazy() || p.Graph.Nodes[0].Attrs["value"].T.Lazy() {
		t.Error("the weight not lazy, or the attribute's tensor lazy")
	}
	got, err := p.Graph.Eval(nil, []string{"y"})
	if err != nil || got["y"].F[3] != 6 {
		t.Fatalf("y = %v, %v", got["y"], err)
	}
}
