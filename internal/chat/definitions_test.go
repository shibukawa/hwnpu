package chat

import (
	"strings"
	"testing"
)

func TestDefinitionsSettingsAndPrograms(t *testing.T) {
	t.Cleanup(func() { LoadModels(nil, false, nil) })
	def := func(programs string) []Definitions {
		return []Definitions{{Name: "app", YAML: []byte("version: 1\nmodels:\n  mine:\n    like: whisper-large-v3-turbo\n    family: \"\"\n    settings: {format: lut4}\n    programs: {iron: [" + programs + "]}\n")}}
	}
	if err := LoadModelsWith(nil, def("/abs/dir"), false, nil); err != nil {
		t.Fatal(err)
	}
	m, ok := Lookup("mine")
	if !ok || m.Settings["format"] != "lut4" || m.Programs["iron"][0] != "/abs/dir" || m.Family != "" || m.Kind != KindTranscribe {
		t.Fatalf("mine: %+v", m)
	}
	if b, _ := Lookup("whisper-large-v3-turbo"); b.Settings != nil || b.Programs != nil {
		t.Errorf("the model it is like has %v %v", b.Settings, b.Programs)
	}
	if err := LoadModelsWith(nil, def("rel/dir"), false, nil); err == nil || !strings.Contains(err.Error(), "absolute") {
		t.Errorf("a relative program: %v", err)
	}
}

// TestLikeKeepsSHA256: a model like another of the same repository keeps its pinned files (an
// app's model of RTMDet-tiny on the CPU names the same ONNX file); one of another repository does
// not.
func TestLikeKeepsSHA256(t *testing.T) {
	t.Cleanup(func() { LoadModels(nil, false, nil) })
	yml := "version: 1\nmodels:\n  same:\n    like: RTMDet-tiny\n    family: \"\"\n    device: {amd: cpu, apple: cpu, intel: cpu}\n  other:\n    like: RTMDet-tiny\n    family: \"\"\n    from: hf:someone/elsewhere\n"
	if err := LoadModelsWith(nil, []Definitions{{Name: "app", YAML: []byte(yml)}}, false, nil); err != nil {
		t.Fatal(err)
	}
	base, _ := Lookup("RTMDet-tiny")
	if m, _ := Lookup("same"); len(m.SHA256) == 0 || m.SHA256["rtmdet-tiny-coco.onnx"] != base.SHA256["rtmdet-tiny-coco.onnx"] {
		t.Errorf("same repository: sha256 %v", m.SHA256)
	}
	if m, _ := Lookup("other"); len(m.SHA256) != 0 {
		t.Errorf("another repository: sha256 %v", m.SHA256)
	}
}
