package chat

import (
	"fmt"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/shibukawa/hwnpu/internal/cache"
	shared "github.com/shibukawa/hwnpu/internal/store"
)

func init() { cache.ExportStale = stalePinned }

// stalePinned reports whether the export at dir has an entry.json (it was fetched or made by
// hwnpu) whose bytes are not those models.yaml pins for it (by its entry's model and target, and
// its folder's name): an export of before a pin moved. An export made by hand (no entry.json),
// or one nothing pins, is not.
func stalePinned(dir string) bool {
	e, err := shared.ReadEntry(dir)
	if err != nil || e.Model == "" {
		return false
	}
	pin, ok := PinOf(modelNamed(e.Model), e.Target, filepath.Base(dir))
	if !ok {
		return false
	}
	return e.Digest() != pin.Digest || !FormatTaken(e)
}

var (
	pinFolder = regexp.MustCompile(`^v[0-9]+(-[0-9a-f]{8,64})?$`)
	pinDigest = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)
)

// ParsePin reads a pin as models.yaml writes it: "FOLDER sha256:DIGEST".
func ParsePin(s string) (Pin, error) {
	f := strings.Fields(s)
	if len(f) != 2 || !pinFolder.MatchString(f[0]) || !pinDigest.MatchString(strings.ToLower(f[1])) {
		return Pin{}, fmt.Errorf("%q is not FOLDER sha256:DIGEST (v0-1a2b3c4d sha256:<64 hex digits>)", s)
	}
	return Pin{At: f[0], Digest: strings.ToLower(f[1])}, nil
}

// toPins reads a model's exports.
func toPins(raw map[string]map[string]string) (map[string]map[string]Pin, error) {
	out := map[string]map[string]Pin{}
	for target, exports := range raw {
		if !safeName(target) || strings.Contains(target, "/") {
			return nil, fmt.Errorf("%q is not a target", target)
		}
		for export, v := range exports {
			if !safeName(export) || strings.HasPrefix(export, "/") || strings.Contains(export, "..") {
				return nil, fmt.Errorf("%s: %q is not an export's name", target, export)
			}
			if strings.TrimSpace(v) == "none" {
				continue // a file over another drops the pin
			}
			p, err := ParsePin(v)
			if err != nil {
				return nil, fmt.Errorf("%s: %s: %w", target, export, err)
			}
			if out[target] == nil {
				out[target] = map[string]Pin{}
			}
			out[target][export] = p
		}
	}
	return out, nil
}

// PinOf is the pin of model m's export for target ("" for any), if models.yaml has one.
func PinOf(m Model, target, export string) (Pin, bool) {
	if target == "" {
		target = "any"
	}
	p, ok := m.Pins[target][export]
	return p, ok
}

// PinOfEntry is the pin of the export an entry of the sources is a version of (by its model,
// target and folder converted/MODEL/TARGET/EXPORT/VERSION), if models.yaml has one.
func PinOfEntry(e shared.Entry) (Pin, bool) {
	return PinOf(modelNamed(e.Model), e.Target, exportOfPath(e.Path))
}

// exportOfPath is EXPORT of converted/MODEL/TARGET/EXPORT.../VERSION ("" for another path).
func exportOfPath(p string) string {
	parts := strings.Split(p, "/")
	if len(parts) < 5 || parts[0] != "converted" {
		return ""
	}
	return strings.Join(parts[3:len(parts)-1], "/")
}

// PinnedPath is the folder of a pinned export in a source.
func PinnedPath(model, target, export string, p Pin) string {
	if target == "" {
		target = "any"
	}
	return shared.EntryPath(model, target, export, p.At)
}

// PinnedAMDModel is the model of the definitions whose exports for AMD's NPU models.yaml pins
// one named export of (DA3-SMALL's da3-small-mv3.amd-npu2), if one does.
func PinnedAMDModel(export string) (Model, bool) {
	for _, m := range Catalog {
		if _, ok := m.Pins[cache.AMDTarget][export]; ok {
			return m, true
		}
	}
	return Model{}, false
}
