package prep

import (
	"cmp"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/imagegen/anima"
	"github.com/shibukawa/hwnpu/internal/imagegen/aspect"
	"github.com/shibukawa/hwnpu/internal/imagegen/chroma"
	"github.com/shibukawa/hwnpu/internal/imagegen/flux2klein"
	"github.com/shibukawa/hwnpu/internal/imagegen/flux3klein"
	"github.com/shibukawa/hwnpu/internal/llm"
)

// image: the Core AI exports hwnpu-image generate draws with, of every size it takes (--size):
// FLUX.2 klein one of every size of models.yaml npu.apple with the edit's programs, Anima one of
// every aspect, FLUX.3 klein and Chroma as hwnpu-image generate makes them now (FLUX.3 klein's
// default sizes, Chroma one for 1:1 and one for 1:1s, until they hold every aspect in one as
// docs/design/new-model-checklist.md 11.4 says).
func init() {
	Register(&Preparer{Name: "image", Kinds: []chat.Kind{chat.KindImage}, Prepare: coreAIOnly(func(e airole.PrepareEnv, f *airole.Files) error {
		m := e.Model
		ctx := chat.WithModel(e.Ctx, m)
		switch {
		case strings.HasPrefix(m.Family, "flux2-klein"):
			as, err := flux2klein.NPUAspects(m, "apple", flux2klein.NPUGeneration("apple"))
			if err != nil {
				return err
			}
			// every size of models.yaml npu.apple in one export (the weights once), as hwnpu-image
			// generate makes it, and the edit's programs of those sizes beside it
			spec := m.NPU["apple"]
			setup := flux2klein.Setup{Dir: filepath.Join(chat.ModelsDir(), "FLUX.2-klein"), Repo: m.Repo, Name: m.Name,
				ConvertOptions: flux2klein.ConvertOptions{Width: as[0].Width, Height: as[0].Height, Sizes: aspect.SizesOf(as), TextFormat: spec.Text, DitFormat: spec.Dit}, Log: e.Log}
			dir, err := flux2klein.Prepare(ctx, setup)
			if err != nil {
				return err
			}
			man, err := flux2klein.ReadProgramManifest(dir)
			if err != nil {
				return err
			}
			ed, err := flux2klein.PrepareEdit(ctx, setup, flux2klein.EditOptions{Patterns: flux2klein.EditPatterns, DitFormat: man.Formats["dit"]})
			if err != nil {
				return fmt.Errorf("the edit: %w", err)
			}
			f.Set("export", dir)
			f.Set("edit", ed)
		case strings.HasPrefix(m.Family, "flux3-klein"):
			dir, err := flux3klein.Prepare(ctx, flux3klein.Setup{Dir: filepath.Join(chat.ModelsDir(), "FLUX.3-klein"), Repo: m.Repo, Name: m.Name, Log: e.Log})
			if err != nil {
				return err
			}
			f.Set("export", dir)
		case strings.HasPrefix(m.Family, "chroma"):
			for _, side := range []int{1024, 512} { // 1:1 and 1:1s
				dir, err := chroma.Prepare(ctx, chroma.Setup{Dir: filepath.Join(chat.ModelsDir(), "Chroma"), Repo: m.Repo, Name: m.Name,
					ConvertOptions: chroma.ConvertOptions{Backend: llm.BackendCoreAI, Width: side, Height: side}, Log: e.Log})
				if err != nil {
					return fmt.Errorf("%dx%d: %w", side, side, err)
				}
				f.Set("export", dir)
			}
		case strings.HasPrefix(m.Family, "anima"):
			// every aspect ratio in one export, as hwnpu-image generate makes it (its --size 1:1 and
			// --qchunk 512 by default)
			w, h, err := anima.ParseAspect("1:1")
			if err != nil {
				return err
			}
			o := anima.ConvertOptions{Backend: llm.BackendCoreAI, Width: w, Height: h, Dit: m.File, QChunk: 512, Sizes: aspect.SizesOf(aspect.All)}
			dir, err := anima.Prepare(ctx, anima.Setup{Dir: filepath.Join(chat.ModelsDir(), "Anima"), ConvertOptions: o, Log: e.Log})
			if err != nil {
				return err
			}
			f.Set("export", dir)
		default:
			return fmt.Errorf("no image model of hwnpu-image generate of the family %q", cmp.Or(m.Family, "(none)"))
		}
		return nil
	})})
}
