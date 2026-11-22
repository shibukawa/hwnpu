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
