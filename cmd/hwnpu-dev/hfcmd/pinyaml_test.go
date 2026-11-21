package hfcmd

import (
	"strings"
	"testing"

	"github.com/goccy/go-yaml"
)

const (
	cA = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	cB = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
)

func TestPinYAML(t *testing.T) {
	base := `version: 1

# the models
models:
  whisper:
    family: whisper
    from: hf:openai/whisper@41f0   # pinned
    machine: 8
  other:
    from: hf:a/b
`
	for _, tc := range []struct {
		name, src, model, repo, target string
		want                           string // a line the result has
	}{
		{"no hub", base, "whisper", "hwnpu/whisper", "amd-xdna2", "    hub:\n      hwnpu/whisper:\n        amd-xdna2: " + cA + "\n    machine: 8"},
		{"another repo", strings.Replace(base, "    machine: 8\n", "    hub:\n      hwnpu/whisper:\n        apple: "+cB+"\n    machine: 8\n", 1), "whisper", "me/whisper", "apple",
			"        apple: " + cB + "\n      me/whisper:\n        apple: " + cA + "\n    machine: 8"},
		{"a target added", strings.Replace(base, "    machine: 8\n", "    hub:\n      hwnpu/whisper:\n        apple: "+cB+"\n    machine: 8\n", 1), "whisper", "hwnpu/whisper", "amd-xdna2",
			"        apple: " + cB + "\n        amd-xdna2: " + cA + "\n    machine: 8"},
		{"replaced", strings.Replace(base, "    machine: 8\n", "    hub:\n      hwnpu/whisper:\n        apple: "+cB+"  # M2 checked\n    machine: 8\n", 1), "whisper", "hwnpu/whisper", "apple",
			"        apple: " + cA + "  # M2 checked\n    machine: 8"},
		{"flow", strings.Replace(base, "    machine: 8\n", "    hub:\n      hwnpu/whisper: {apple: "+cB+"}\n    machine: 8\n", 1), "whisper", "hwnpu/whisper", "amd-xdna2",
			"      hwnpu/whisper: {apple: " + cB + ", amd-xdna2: " + cA + "}"},
		{"no model", base, "LFM3", "me/LFM3", "apple", "    from: hf:a/b\n  LFM3:\n    hub:\n      me/LFM3:\n        apple: " + cA},
		{"no models", "version: 1\nhub: {push: me}\n", "LFM3", "me/LFM3", "apple", "\nmodels:\n  LFM3:\n    hub:\n      me/LFM3:\n        apple: " + cA},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out, err := pinYAML([]byte(tc.src), tc.model, tc.repo, tc.target, cA)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(string(out), tc.want) {
				t.Errorf("no\n%s\nin\n%s", tc.want, out)
			}
			if !strings.HasSuffix(string(out), "\n") || strings.Count(string(out), "# the models") != strings.Count(tc.src, "# the models") {
				t.Errorf("the file's end or comments changed:\n%s", out)
			}
			var v struct {
				Models map[string]struct {
					Hub map[string]map[string]string `yaml:"hub"`
				} `yaml:"models"`
			}
			if err := yaml.Unmarshal(out, &v); err != nil {
				t.Fatalf("%v\n%s", err, out)
			}
			if got := v.Models[tc.model].Hub[tc.repo][tc.target]; got != cA {
				t.Errorf("reads %q:\n%s", got, out)
			}
		})
	}
	if _, err := pinYAML([]byte("version: 1\nmodels:\n  w:\n    hub: {x/y: {apple: "+cB+"}}\n"), "w", "x/y", "apple", cA); err == nil {
		t.Error("a hub in one line was rewritten")
	}
}
