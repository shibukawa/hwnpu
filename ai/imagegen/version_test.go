package imagegen

import (
	"fmt"
	"testing"

	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/imagegen/anima"
	"github.com/shibukawa/hwnpu/internal/imagegen/flux2klein"
	"github.com/shibukawa/hwnpu/internal/imagegen/flux3klein"
)

// TestCurrentExport: a store's record of a Core AI export of an earlier version is not usable
// (FLUX.2 klein's v6 under a hwnpu of v7 failed to load), one of this version, AMD's and a
// definition's export by path are.
func TestCurrentExport(t *testing.T) {
	files := func(fam string, given bool, paths map[string]string) airole.Files {
		var f airole.Files
		for k, p := range paths {
			f.Set(k, p)
		}
		f.SetValue("family", fam)
		if given {
			f.SetValue(valueGiven, "true")
		}
		return f
	}
	klein := func(v int) string {
		return fmt.Sprintf("/c/models/FLUX.2-klein/flux.2-klein-4b-8sizes-4d02f40a-5edits-624c2d9e-lut8-lut8-v%d.coreai", v)
	}
	kv := flux2klein.ExportVersion
	for _, c := range []struct {
		name string
		f    airole.Files
		ok   bool
	}{
		{"klein now", files(famKlein, false, map[string]string{"export": klein(kv)}), true},
		{"klein earlier", files(famKlein, false, map[string]string{"export": klein(kv - 1)}), false},
		{"klein device earlier", files(famKlein, false, map[string]string{"export": klein(kv), "device": klein(kv-1)[:len(klein(kv-1))-len(".coreai")] + "-device.coreai"}), false},
		{"klein given", files(famKlein, true, map[string]string{"export": klein(kv - 1)}), true},
		{"klein AMD", files(famKlein, false, map[string]string{"export": "/c/entries/amd-xdna2/flux.2-klein-4b.amd-npu2"}), true},
		{"flux3 now", files(famFlux3, false, map[string]string{"export": fmt.Sprintf("/c/models/x/flux-3-image-klein-512+1024-t80+160-lut8-lut8x2-v%d.coreai", flux3klein.ExportVersion)}), true},
		{"anima earlier", files(famAnima, false, map[string]string{"export": "/c/models/Anima/anima-turbo-v1.1-8sizes-ffe2f7fe-int8-int8-g4-q512-v5.coreai"}), false},
		{"anima now", files(famAnima, false, map[string]string{"export": fmt.Sprintf("/c/models/Anima/anima-turbo-v1.1-8sizes-ffe2f7fe-int8-int8-g4-q512-v%d.coreai", anima.CoreAIExportVersion)}), true},
		{"anima Intel", files(famAnima, false, map[string]string{"export": "/c/models/Anima/anima-turbo-v1.1-1024-fp16-fp16-g4-q512-v4.intel"}), true},
	} {
		if err := currentExport(c.f); (err == nil) != c.ok {
			t.Errorf("%s: %v, want usable %v", c.name, err, c.ok)
		}
	}
}
