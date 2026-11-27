package speechwire

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// The exports for AMD's NPU that a command's --model may name by their directory. IsSpeechExport
// and IsWhisperExport are copies of speech.IsIronExport and transcribe.IsIronExport (the models'
// code, model-recipes.yaml); ai/speech/engine's tests check that they give the same.

// IsSpeechExport reports whether dir is an export of tools/amd/iron_lfm3speech.py.
func IsSpeechExport(dir string) bool {
	raw, err := os.ReadFile(filepath.Join(dir, "model.json"))
	if err != nil {
		return false
	}
	var m struct {
		Kind string `json:"kind"`
	}
	if json.Unmarshal(raw, &m) != nil || m.Kind != "lfm3speech" {
		return false
	}
	for _, f := range []string{"text/manifest.json", "depth/manifest.json", "codec/manifest.json", "tables.safetensors", "audio_tokenizer/config.json"} {
		if _, err := os.Stat(filepath.Join(dir, f)); err != nil {
			return false
		}
	}
	return true
}

// whisperMeta is model.json of an export of tools/amd/iron_whisper.py (transcribe.IronMeta).
type whisperMeta struct {
	Kind   string `json:"kind"`
	Rows   int    `json:"rows"`
	ConvIn int    `json:"conv_in"`
	Vocab  int    `json:"vocab_pad"`
	Frames int    `json:"frames"`
	Config struct {
		Hidden    int `json:"hidden"`
		Mels      int `json:"mels"`
		DecLayers int `json:"dec_layers"`
		Positions int `json:"positions"`
		Targets   int `json:"targets"`
		Vocab     int `json:"vocab"`
	} `json:"config"`
}

// IsWhisperExport reports whether dir is an export of tools/amd/iron_whisper.py.
func IsWhisperExport(dir string) bool {
	raw, err := os.ReadFile(filepath.Join(dir, "model.json"))
	if err != nil {
		return false
	}
	var m whisperMeta
	if json.Unmarshal(raw, &m) != nil || m.Kind != "whisper" {
		return false
	}
	for _, f := range []string{"decoder/manifest.json", "encoder/manifest.json"} {
		if _, err := os.Stat(filepath.Join(dir, f)); err != nil {
			return false
		}
	}
	return true
}
