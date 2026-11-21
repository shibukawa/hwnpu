package chroma

import (
	"slices"
	"testing"

	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/imagegen/aspect"
	"github.com/shibukawa/hwnpu/internal/imagegen/flux2klein"
	"github.com/shibukawa/hwnpu/internal/llm"
)

// Every Chroma has models.yaml's eight Apple sizes (Chroma1-HD by its like), and one Core AI
// export of them is named by their count and a hash; one size keeps its old name.
func TestExportNameSizes(t *testing.T) {
	var sizes [][2]int
	for _, name := range []string{"Chroma1-Flash", "Chroma1-HD"} {
		m, ok := chat.Lookup(name)
		if !ok {
			t.Fatalf("%s is not in models.yaml", name)
		}
		as, err := flux2klein.NPUAspects(m, "apple", "")
		if err != nil {
			t.Fatal(err)
		}
		if len(as) != 8 {
			t.Fatalf("%s: %d Apple sizes", name, len(as))
		}
		if sizes == nil {
			sizes = aspect.SizesOf(as)
		} else if !slices.Equal(sizes, aspect.SizesOf(as)) {
			t.Errorf("%s's sizes %v, Chroma1-Flash's %v", name, aspect.SizesOf(as), sizes)
		}
	}
	o := ConvertOptions{Backend: llm.BackendCoreAI, Width: 1024, Height: 1024, Sizes: sizes}
	if n := ExportName("Chroma1-Flash", o); n != "chroma1-flash-"+aspect.Tag(sizes)+"-lut8+wofp16-lut8-v3.coreai" {
		t.Errorf("export name %s", n)
	}
	if n := ExportName("Chroma1-Flash", ConvertOptions{Backend: llm.BackendCoreAI, Width: 512}); n != "chroma1-flash-512x512-lut8+wofp16-lut8-v3.coreai" {
		t.Errorf("export name of one size %s", n)
	}
}

// At gives the parts of the size asked for, and fails on a size the export does not hold.
func TestManifestAt(t *testing.T) {
	m := &ProgramManifest{Dit: Config{Width: 1024, Height: 1024}, Vae: []string{"vae_s0.aimodel#s1024x1024"},
		Sizes: []SizeParts{{Width: 1024, Height: 1024, Vae: []string{"vae_s0.aimodel#s1024x1024"}},
			{Width: 768, Height: 1152, Vae: []string{"vae_s0.aimodel#s768x1152"}, Doubles: []Group{{"dit_double_0.aimodel#s768x1152", 0, 4}}}}}
	if got := m.SizeList(); !slices.Equal(got, [][2]int{{1024, 1024}, {768, 1152}}) {
		t.Errorf("sizes %v", got)
	}
	if err := m.At(768, 1152); err != nil {
		t.Fatal(err)
	}
	if m.Dit.Width != 768 || m.Dit.Height != 1152 || m.Vae[0] != "vae_s0.aimodel#s768x1152" || m.Doubles[0].Name != "dit_double_0.aimodel#s768x1152" {
		t.Errorf("at 768x1152: %+v %v %v", m.Dit, m.Vae, m.Doubles)
	}
	if err := m.At(640, 640); err == nil {
		t.Error("640x640: no error")
	}
}
