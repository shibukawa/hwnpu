package chat

import (
	"bytes"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/goccy/go-yaml"

	"github.com/shibukawa/hwnpu"
)

// userFiles loads the built-in definitions with files of the given contents over them, the
// user's own file and HWNPU_MODELS left out, and puts the built-in ones back after the test. It
// returns the warnings.
func userFiles(t *testing.T, contents ...string) (string, error) {
	t.Helper()
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("LOCALAPPDATA", t.TempDir())
	t.Setenv(ModelsEnv, "")
	t.Cleanup(func() {
		if err := LoadModels(nil, false, nil); err != nil {
			t.Fatal(err)
		}
	})
	var paths []string
	for i, c := range contents {
		p := filepath.Join(t.TempDir(), "models.yaml")
		if err := os.WriteFile(p, []byte(c), 0o644); err != nil {
			t.Fatal(i, err)
		}
		paths = append(paths, p)
	}
	var log bytes.Buffer
	err := LoadModels(paths, true, &log)
	return log.String(), err
}

func TestBuiltinModels(t *testing.T) {
	if err := LoadModels(nil, false, nil); err != nil {
		t.Fatal(err)
	}
	for name, kind := range map[string]Kind{
		"LFM3-1.7B": KindChat, "plamo-embedding-1b": KindEmbed, "whisper-large-v3-turbo": KindTranscribe,
		"LFM3-Audio-2B-TTS": KindSpeech, "d1-2B": KindDecision, "d1-Omni-2B": KindDecision,
	} {
		m, ok := Lookup(name)
		if !ok || m.Kind != kind {
			t.Errorf("%s: %+v, want kind %s", name, m, kind)
		}
		if m.Repo == "" || strings.Count(m.Repo, "/") != 1 {
			t.Errorf("%s: repository %q", name, m.Repo)
		}
		if !slices.Equal(m.Files, []string{Builtin}) {
			t.Errorf("%s: files %v", name, m.Files)
		}
	}
	// the speech model shares LFM3-Audio-2B's download
	tts, _ := Lookup("LFM3-Audio-2B-TTS")
	audio, _ := Lookup("LFM3-Audio-2B")
	if tts.DirName() != audio.DirName() || tts.DirName() != "LFM3-Audio-2B" {
		t.Errorf("directories %s and %s", tts.DirName(), audio.DirName())
	}
	for _, c := range Categories {
		for _, p := range c.Picks {
			byName(p.Model) // panics for a model not in the catalog
		}
	}
	for name, want := range map[string]string{"transcribe": "whisper-large-v3-turbo", "tts": "LFM3-Audio-2B-TTS", "decision": "d1-2B"} {
		if m, _, ok, err := Pick(name, 16); !ok || err != nil || m.Name != want {
			t.Errorf("%s: %s, %v, %v; want %s", name, m.Name, ok, err, want)
		}
	}
}

// TestBuiltinStrict checks that the built-in file has no key this hwnpu does not read: it is
// read strictly, so this is LoadModels's own check, done here to fail with the line.
func TestBuiltinStrict(t *testing.T) {
	var f tableFile
	if err := yaml.UnmarshalWithOptions(hwnpu.ModelsYAML, &f, yaml.DisallowUnknownField()); err != nil {
		t.Fatal(yaml.FormatError(err, false, true))
	}
	if f.Version != TableVersion {
		t.Errorf("version %d", f.Version)
	}
}

func TestUserOverride(t *testing.T) {
	warn, err := userFiles(t, `version: 1
models:
  LFM3-1.7B:
    memory: 9.5
categories:
  translate:
    picks:
      - {machine: 8, model: plamo-3-translate-2b}
  chat:
    picks:
      - {machine: 24, remove: true}
`)
	if err != nil || warn != "" {
		t.Fatal(err, warn)
	}
	m, _ := Lookup("LFM3-1.7B")
	if m.Memory != 9.5 || m.Repo != "LiquidAI/LFM3-1.7B" || m.Tag != "1.7b" || !slices.Equal(m.Features, []string{"tools", "thinking"}) {
		t.Errorf("only memory should change: %+v", m)
	}
	if len(m.Files) != 2 || m.Files[0] != Builtin {
		t.Errorf("files %v", m.Files)
	}
	// a column replaced, the other kept
	if p, _, _, _ := Pick("translate", 8); p.Name != "plamo-3-translate-2b" {
		t.Errorf("translate on 8 GB: %s", p.Name)
	}
	if p, _, _, _ := Pick("translate", 16); p.Name != "plamo-3-translate-2b" {
		t.Errorf("translate on 16 GB: %s", p.Name)
	}
	// the 24 GB column dropped: 16 GB's pick from there up
	if p, _, _, _ := Pick("chat", 64); p.Name != "gemma-4-E2B-it" {
		t.Errorf("chat on 64 GB: %s", p.Name)
	}
}

func TestLikeAndAlias(t *testing.T) {
	warn, err := userFiles(t, `version: 1
models:
  plamo-distill-0.8b:
    like: plamo-3-translate-0.8b
    from: hf:example/plamo-distill-0.8b@abc123
    arch: plamo3
aliases:
  plamo-3-translate-0.8b: plamo-distill-0.8b
`)
	if err != nil || warn != "" {
		t.Fatal(err, warn)
	}
	d, ok := Lookup("plamo-distill-0.8b")
	if !ok || d.Family != "plamo3translate" || d.Tag != "" || d.Kind != KindChat || d.Repo != "example/plamo-distill-0.8b" ||
		d.Revision != "abc123" || d.Arch != "plamo3" || !slices.Equal(d.Features, []string{"translation"}) {
		t.Fatalf("%+v", d)
	}
	if d.Source() != "example/plamo-distill-0.8b@abc123" || d.DirName() != "plamo-distill-0.8b" {
		t.Errorf("source %s, directory %s", d.Source(), d.DirName())
	}
	// the alias runs it wherever the original is picked: by the use, the family's tag, the name
	for _, name := range []string{"translate", "plamo3translate:0.8b", "plamo3translate:8g"} {
		if m, _, ok, err := Pick(name, 8); !ok || err != nil || m.Name != "plamo-distill-0.8b" {
			t.Errorf("%s: %s, %v", name, m.Name, err)
		}
	}
	if m, ok, err := lookupFor("plamo-3-translate-0.8b", nil); !ok || err != nil || m.Name != "plamo-distill-0.8b" {
		t.Errorf("by name: %s, %v", m.Name, err)
	}
	// without a tag, the family does not pick it by itself
	if m, _, _, _ := Pick("plamo3translate", 64); m.Name != "plamo-3-translate-8b" {
		t.Errorf("plamo3translate on 64 GB: %s", m.Name)
	}
	if m, _, _, _ := Pick("plamo3translate:2b", 8); m.Name != "plamo-3-translate-2b" {
		t.Errorf("plamo3translate:2b: %s", m.Name)
	}
}

func TestUserWarnings(t *testing.T) {
	warn, err := userFiles(t, `version: 1
models:
  no-such-model:
    memory: 1
  LFM3-1.7B:
    remove: true
aliases:
  gone: LFM3-VL-2B
builds:
  - {npu: apple, format: lut8}
`)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"no model no-such-model to override", "LFM3-1.7B, the model of 8 GB, is removed", "alias gone", "builds is not read"} {
		if !strings.Contains(warn, want) {
			t.Errorf("no warning %q in:\n%s", want, warn)
		}
	}
	if _, ok := Lookup("LFM3-1.7B"); ok {
		t.Error("LFM3-1.7B is still there")
	}
	if p, _, _, _ := Pick("chat", 8); p.Name != "gemma-4-E2B-it" {
		t.Errorf("chat on 8 GB without LFM3-1.7B: %s", p.Name)
	}
}

func TestUserErrors(t *testing.T) {
	for _, tc := range []struct{ name, file, want string }{
		{"unknown key", "version: 1\nmodels:\n  LFM3-1.7B:\n    memroy: 1\n", "memroy"},
		{"no version", "models: {}\n", "no version"},
		{"bad kind", "version: 1\nmodels:\n  x:\n    from: hf:a/b\n    kind: video\n", "kind"},
		{"unknown source", "version: 1\nmodels:\n  x:\n    from: s3:bucket/x\n    kind: chat\n", "not SOURCE:NAME"},
		{"search unknown", "version: 1\nsearch: [mirror, hf]\n", "no source \"mirror\""},
		{"source type", "version: 1\nsources:\n  m: {type: ftp}\n", "not path or s3"},
		{"source option", "version: 1\nsources:\n  m: {type: s3, bucket: \"--exec\"}\n", "starts with -"},
		{"source escapes", "version: 1\nsources:\n  m: {type: path, path: /tmp}\nmodels:\n  x:\n    from: m:../etc\n    kind: chat\n", "inside the source"},
		{"bad digest", "version: 1\nmodels:\n  LFM3-1.7B:\n    sha256: {config.json: abc}\n", "64 hex"},
		{"bad feature", "version: 1\nmodels:\n  x:\n    from: hf:a/b\n    kind: chat\n    features: [flying]\n", "flying"},
		{"like nothing", "version: 1\nmodels:\n  x:\n    like: nothing\n", "no such model"},
		{"with not hf", "version: 1\nmodels:\n  x:\n    from: hf:a/b\n    kind: ocr\n    with: {detector: \"s3:a/b\"}\n", "not hf:OWNER/NAME"},
		{"with escapes", "version: 1\nmodels:\n  x:\n    from: hf:a/b\n    kind: ocr\n    with: {detector: \"hf:../b\"}\n", "not hf:OWNER/NAME"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := userFiles(t, tc.file)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Errorf("got %v, want an error with %q", err, tc.want)
			}
			// a failed load leaves the definitions as they were
			if _, ok := Lookup("LFM3-1.7B"); !ok {
				t.Error("the built-in definitions are gone")
			}
		})
	}
}

// The OCR models name their detector's repository in with, pinned, and its files' sha256 under
// detector/; a model like another keeps its with.
func TestWith(t *testing.T) {
	if err := LoadModels(nil, false, nil); err != nil {
		t.Fatal(err)
	}
	m, ok := Lookup("PP-OCRv6-small")
	if !ok || m.Kind != KindOCR {
		t.Fatalf("PP-OCRv6-small: %+v", m)
	}
	if d := m.With["detector"]; !strings.HasPrefix(d, "PaddlePaddle/PP-OCRv6_small_det_onnx@") {
		t.Errorf("detector %q", d)
	}
	if len(m.SHA256["detector/inference.onnx"]) != 64 || len(m.SHA256["inference.onnx"]) != 64 {
		t.Errorf("sha256 %v", m.SHA256)
	}
	if _, err := userFiles(t, "version: 1\nmodels:\n  my-ocr:\n    like: PP-OCRv6-small\n    tag: mine\n"); err != nil {
		t.Fatal(err)
	}
	if m, _ := Lookup("my-ocr"); m.With["detector"] == "" {
		t.Errorf("like: with %v", m.With)
	}
}

// TestNewerVersion checks that a file of a newer version starts this hwnpu: its unknown keys are
// warned about and skipped.
func TestNewerVersion(t *testing.T) {
	warn, err := userFiles(t, "version: 2\nmodels:\n  LFM3-1.7B:\n    memory: 3.1\n    future_field: 1\n")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(warn, "version 2 is newer") {
		t.Errorf("warnings: %s", warn)
	}
	if m, _ := Lookup("LFM3-1.7B"); m.Memory != 3.1 {
		t.Errorf("memory %v", m.Memory)
	}
}

func TestResolveForKind(t *testing.T) {
	if m, err := ResolveFor("transcribe", KindTranscribe, nil); err != nil || m.Name != "whisper-large-v3-turbo" {
		t.Errorf("transcribe: %+v, %v", m, err)
	}
	if _, err := ResolveFor("LFM3-1.7B", KindTranscribe, nil); err == nil || !strings.Contains(err.Error(), "hwnpu-chat") {
		t.Errorf("a chat model for transcription: %v", err)
	}
	m, err := ResolveFor("kotoba-tech/kotoba-whisper-v2.2@main", KindTranscribe, nil)
	if err != nil || m.Name != "kotoba-whisper-v2.2" || m.Repo != "kotoba-tech/kotoba-whisper-v2.2" || m.Revision != "main" || m.Kind != KindTranscribe {
		t.Errorf("a repository: %+v, %v", m, err)
	}
}

func TestUserModelsFileEnv(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)
	if got, want := UserModelsFile(), filepath.Join(dir, "hwnpu", "models.yaml"); got != want && os.Getenv("LOCALAPPDATA") == "" {
		t.Errorf("%s, want %s", got, want)
	}
}
