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

func TestFieldYAML(t *testing.T) {
	src := "version: 1\nmodels:\n  a:\n    from: hf:x/a\n    bytes: {amd: 1e9}  # packed\n    about: x\n  b:\n    from: hf:x/b\n    disk: {amd: 1e9}  # old\n"
	out, err := fieldYAML([]byte(src), "a", "disk", "{amd: 2.00e9}", "bytes")
	if err != nil || !strings.Contains(string(out), "    bytes: {amd: 1e9}  # packed\n    disk: {amd: 2.00e9}\n    about: x\n") {
		t.Errorf("%v\n%s", err, out)
	}
	out, err = fieldYAML([]byte(src), "b", "disk", "{amd: 3.00e9}", "bytes")
	if err != nil || !strings.Contains(string(out), "    disk: {amd: 3.00e9}  # old\n") || strings.Count(string(out), "disk:") != 1 {
		t.Errorf("%v\n%s", err, out)
	}
	if _, err := fieldYAML([]byte(src), "c", "disk", "1", "bytes"); err == nil {
		t.Error("a model not in the file")
	}
}

// TestExportsYAML writes models.MODEL.exports.TARGET.EXPORT, a pin, as nestedYAML does for
// hwnpu-dev models pin, and replaces it.
func TestExportsYAML(t *testing.T) {
	src := "version: 1\nmodels:\n  LFM3-1.7B:\n    from: hf:LiquidAI/LFM3-1.7B\n    machine: 8\n"
	d := strings.Repeat("ab", 32)
	out, err := nestedYAML([]byte(src), "LFM3-1.7B", "exports", "amd-xdna2", "LFM3-1.7B.amd-npu2-q4", "v0-7bfc0b02 sha256:"+d)
	if err != nil {
		t.Fatal(err)
	}
	if out, err = nestedYAML(out, "LFM3-1.7B", "exports", "amd-xdna2", "LFM3-1.7B.amd-npu2-q6", "v1 sha256:"+d); err != nil {
		t.Fatal(err)
	}
	if out, err = nestedYAML(out, "LFM3-1.7B", "exports", "amd-xdna2", "LFM3-1.7B.amd-npu2-q4", "v0-11111111 sha256:"+d); err != nil {
		t.Fatal(err)
	}
	var back struct {
		Models map[string]struct {
			Exports map[string]map[string]string `yaml:"exports"`
			Machine int                          `yaml:"machine"`
		} `yaml:"models"`
	}
	if err := yaml.Unmarshal(out, &back); err != nil {
		t.Fatalf("%v\n%s", err, out)
	}
	got := back.Models["LFM3-1.7B"]
	if got.Exports["amd-xdna2"]["LFM3-1.7B.amd-npu2-q4"] != "v0-11111111 sha256:"+d || got.Exports["amd-xdna2"]["LFM3-1.7B.amd-npu2-q6"] != "v1 sha256:"+d || got.Machine != 8 {
		t.Errorf("read back %+v from\n%s", got, out)
	}
}

// hwnpu-dev models hf push's selector takes the CPU's Core ML packages (coreml) as a group of
// their own, pinned under that target.
func TestCoreMLSelector(t *testing.T) {
	sel, err := parseSelector("1:YuNet/coreml")
	if err != nil || sel.group != groupCoreML || sel.layer != 1 || sel.which != "" {
		t.Fatalf("parseSelector = %+v, %v", sel, err)
	}
	src := []byte("models:\n  YuNet:\n    kind: face-detect\n")
	out, err := pinYAML(src, "YuNet", "hwnpu/YuNet", groupCoreML, strings.Repeat("a", 40))
	if err != nil || !strings.Contains(string(out), "coreml: "+strings.Repeat("a", 40)) {
		t.Fatalf("pinYAML:\n%s\n%v", out, err)
	}
}
