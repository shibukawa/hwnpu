package probe

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/ovir"
)

// The IR must parse with hwnpu's reader and have the weights the format promises. Set
// HWNPU_IR_OUT to a directory to keep the files, e.g. to check them with OpenVINO's read_model.
func TestBuildIR(t *testing.T) {
	for _, f := range []string{"fp16", "int8", "int4"} {
		xmlData, bin, err := BuildIR(f, 2, 1)
		if err != nil {
			t.Fatal(err)
		}
		m, err := ovir.Parse(xmlData)
		if err != nil {
			t.Fatal(f, err)
		}
		if len(m.Inputs) != 1 || len(m.Outputs) != 1 || m.Outputs[0].Name == "" {
			t.Fatalf("%s: inputs %v outputs %v", f, m.Inputs, m.Outputs)
		}
		if w := WeightBytes(f, 2); int64(len(bin)) < w || int64(len(bin)) > w+w/50 {
			t.Errorf("%s: bin %d bytes, weights %d", f, len(bin), w)
		}
		if dir := os.Getenv("HWNPU_IR_OUT"); dir != "" {
			os.WriteFile(filepath.Join(dir, "probe_"+f+".xml"), xmlData, 0o644)
			os.WriteFile(filepath.Join(dir, "probe_"+f+".bin"), bin, 0o644)
		}
	}
}

// A layer must have the same weights in a part as in the whole model, so that running the parts
// in order computes the whole.
func TestSplitWeights(t *testing.T) {
	parts := Split(5, 2)
	if len(parts) != 2 || parts[0] != (Part{0, 3, false}) || parts[1] != (Part{3, 2, true}) {
		t.Fatalf("Split(5, 2) = %v", parts)
	}
	_, whole, err := BuildIR("int8", 2, 7)
	if err != nil {
		t.Fatal(err)
	}
	_, second, err := BuildIRPart("int8", Part{First: 1, Layers: 1, Last: true}, 7)
	if err != nil {
		t.Fatal(err)
	}
	_, first, err := BuildIRPart("int8", Part{Layers: 1}, 7)
	if err != nil {
		t.Fatal(err)
	}
	w := int(WeightBytes("int8", 1))
	if !bytes.Contains(whole, second[len(second)-w:]) || !bytes.Contains(whole, first[len(first)-w:]) {
		t.Error("a part's layer weights are not in the whole model")
	}
	if bytes.Equal(first[len(first)-w:], second[len(second)-w:]) {
		t.Error("two layers have the same weights")
	}
}
