package ovir

import (
	"bytes"
	"encoding/binary"
	"encoding/xml"
	"os"
	"strings"
	"testing"
)

func load(t *testing.T) []byte {
	t.Helper()
	b, err := os.ReadFile("../fixtures/openvino/affine_relu.xml")
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestParse(t *testing.T) {
	m, err := Parse(load(t))
	if err != nil {
		t.Fatal(err)
	}
	if m.Version != 11 || len(m.Inputs) != 1 || len(m.Outputs) != 1 || len(m.Layers) != 7 {
		t.Fatalf("%+v", m)
	}
	in, out := m.Inputs[0], m.Outputs[0]
	if in.Name != "x" || in.Precision != "FP32" || len(in.Dims) != 2 || in.Dims[1] != 8 {
		t.Fatalf("input %+v", in)
	}
	if out.Name != "y" || out.Precision != "FP32" || len(out.Dims) != 2 {
		t.Fatalf("output %+v (name must be the producer's friendly name)", out)
	}
}

func TestIOFlags(t *testing.T) {
	m, _ := Parse(load(t))
	want := `--inputs_precisions="0:FP32" --inputs_layouts="0:NC" --outputs_precisions="0:FP32" --outputs_layouts="0:NC"`
	if got := m.IOFlags(true); got != want {
		t.Fatalf("indices:\n got %s\nwant %s", got, want)
	}
	if got := m.IOFlags(false); !strings.Contains(got, `--inputs_precisions="x:FP32"`) || !strings.Contains(got, `--outputs_layouts="y:NC"`) {
		t.Fatalf("names: %s", got)
	}
	got := m.BuildFlags(true, map[string]string{"PERFORMANCE_HINT": "LATENCY"}, []string{"PERFORMANCE_HINT"})
	if !strings.HasSuffix(got, ` --config PERFORMANCE_HINT="LATENCY"`) {
		t.Fatalf("config: %s", got)
	}
	if !UseIndices(5, 9) || UseIndices(5, 8) || !UseIndices(6, 0) || UseIndices(4, 20) {
		t.Fatal("UseIndices")
	}
}

func TestSerialize(t *testing.T) {
	out := Serialize(7, 3, []byte("<net/>"), []byte{1, 2, 3})
	if binary.LittleEndian.Uint16(out[0:]) != 7 || binary.LittleEndian.Uint16(out[2:]) != 3 ||
		binary.LittleEndian.Uint32(out[4:]) != 2 || binary.LittleEndian.Uint64(out[8:]) != 6 ||
		string(out[16:22]) != "<net/>" || binary.LittleEndian.Uint64(out[22:]) != 3 || !bytes.Equal(out[30:], []byte{1, 2, 3}) {
		t.Fatalf("% x", out)
	}
}

func TestInjectRTInfo(t *testing.T) {
	check := func(name string, src string) {
		t.Helper()
		got, err := InjectRTInfo([]byte(src))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		s := string(got)
		for _, k := range RequiredRTInfo {
			if strings.Count(s, "<"+k+" ") != 1 {
				t.Fatalf("%s: %s missing or duplicated:\n%s", name, k, s)
			}
		}
		var anyXML struct{}
		if err := xml.Unmarshal(got, &anyXML); err != nil {
			t.Fatalf("%s: result is not well-formed XML: %v\n%s", name, err, s)
		}
		again, _ := InjectRTInfo(got)
		if !bytes.Equal(again, got) {
			t.Fatalf("%s: not idempotent", name)
		}
	}
	check("fixture with self-closing rt_info", string(load(t)))
	check("open rt_info", `<net version="11"><layers><layer id="0" name="a" type="Parameter"><rt_info><x/></rt_info></layer></layers><rt_info>
		<is_new_api value="true" />
	</rt_info></net>`)
	check("no rt_info", `<net version="11"><layers/></net>`)

	// layer level rt_info must not be touched
	got, _ := InjectRTInfo([]byte(`<net><layers><layer><rt_info><k/></rt_info></layer></layers></net>`))
	if !strings.Contains(string(got), `<layer><rt_info><k/></rt_info></layer>`) {
		t.Fatalf("layer rt_info modified: %s", got)
	}
}
