package engine

import (
	"os"
	"path/filepath"
	"testing"

	isp "github.com/shibukawa/hwnpu/internal/speech"
	"github.com/shibukawa/hwnpu/internal/speechwire"
	"github.com/shibukawa/hwnpu/internal/transcribe"
)

// TestSpeechwireExports checks that speechwire's copies tell the exports for AMD's NPU as the
// engine's packages do.
func TestSpeechwireExports(t *testing.T) {
	write := func(dir string, files map[string]string) {
		for name, body := range files {
			p := filepath.Join(dir, name)
			if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}
	speech := map[string]string{"model.json": `{"kind":"lfm3speech"}`, "text/manifest.json": "{}", "depth/manifest.json": "{}",
		"codec/manifest.json": "{}", "tables.safetensors": "", "audio_tokenizer/config.json": "{}"}
	whisper := map[string]string{"model.json": `{"kind":"whisper","rows":1500,"config":{"hidden":384}}`,
		"decoder/manifest.json": "{}", "encoder/manifest.json": "{}"}
	dirs := map[string]map[string]string{
		"speech": speech, "whisper": whisper, "empty": {},
		"speech-part":  {"model.json": `{"kind":"lfm3speech"}`, "text/manifest.json": "{}"},
		"whisper-bad":  {"model.json": `{"kind":"whisper","rows":"x"}`, "decoder/manifest.json": "{}", "encoder/manifest.json": "{}"},
		"whisper-part": {"model.json": `{"kind":"whisper"}`, "decoder/manifest.json": "{}"},
		"other":        {"model.json": `{"kind":"other"}`},
	}
	root := t.TempDir()
	for name, files := range dirs {
		dir := filepath.Join(root, name)
		os.MkdirAll(dir, 0o755)
		write(dir, files)
		if got, want := speechwire.IsSpeechExport(dir), isp.IsIronExport(dir); got != want {
			t.Errorf("%s: speechwire.IsSpeechExport %v, speech.IsIronExport %v", name, got, want)
		}
		if got, want := speechwire.IsWhisperExport(dir), transcribe.IsIronExport(dir); got != want {
			t.Errorf("%s: speechwire.IsWhisperExport %v, transcribe.IsIronExport %v", name, got, want)
		}
	}
	if !speechwire.IsSpeechExport(filepath.Join(root, "speech")) || !speechwire.IsWhisperExport(filepath.Join(root, "whisper")) {
		t.Error("an export is not told")
	}
}
