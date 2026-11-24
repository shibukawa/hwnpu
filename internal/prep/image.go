package prep

import (
	"cmp"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/shibukawa/bonsaiai/internal/airole"
	"github.com/shibukawa/bonsaiai/internal/chat"
	"github.com/shibukawa/bonsaiai/internal/imagegen/anima"
	"github.com/shibukawa/bonsaiai/internal/imagegen/aspect"
	"github.com/shibukawa/bonsaiai/internal/imagegen/flux2klein"
	"github.com/shibukawa/bonsaiai/internal/imagegen/flux3klein"
	"github.com/shibukawa/bonsaiai/internal/llm"
)

// image: the Core AI exports bonsaiai-image generate draws with, of every size it takes (--size):
// FLUX.2 klein one of every size and edit of models.yaml npu.apple and one beside it of the edits
// of 1024, Anima one of
// every aspect, FLUX.3 klein as bonsaiai-image generate makes it now (its default sizes, until it
// holds every aspect in one as docs/design/new-model-checklist.md 11.4 says).
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
			// every size and edit of models.yaml npu.apple in one export (the weights once), as
			// bonsaiai-image generate makes it, and the export beside it of the edits of a reference of
			// 1024 (flux2klein.DeviceExport)
			o, err := flux2klein.AppleOptions(m, flux2klein.NPUGeneration("apple"), as[0].Width, as[0].Height, "", "")
			if err != nil {
				return err
			}
			setup := flux2klein.Setup{Dir: filepath.Join(chat.ModelsDir(), "FLUX.2-klein"), Repo: m.Repo, Name: m.Name, ConvertOptions: o, Log: e.Log}
			dir, err := flux2klein.Prepare(ctx, setup)
			if err != nil {
				return err
			}
			dd, err := flux2klein.PrepareDevice(ctx, setup)
			if err != nil {
				return fmt.Errorf("the edits of 1024: %w", err)
			}
			f.Set("export", dir)
			if dd != "" {
				f.Set("device", dd)
			}
		case strings.HasPrefix(m.Family, "flux3-klein"):
			dir, err := flux3klein.Prepare(ctx, flux3klein.Setup{Dir: filepath.Join(chat.ModelsDir(), "FLUX.3-klein"), Repo: m.Repo, Name: m.Name, Log: e.Log})
			if err != nil {
				return err
			}
			f.Set("export", dir)
		case strings.HasPrefix(m.Family, "anima"):
			// every aspect ratio in one export, as bonsaiai-image generate makes it (its --size 1:1, the
			// rest from the recipe)
			w, h, err := anima.ParseAspect("1:1")
			if err != nil {
				return err
			}
			o := anima.ConvertOptions{Backend: llm.BackendCoreAI, Width: w, Height: h, Dit: m.File, Sizes: aspect.SizesOf(aspect.All)}
			if err := anima.FromRecipe(m.Name, &o); err != nil {
				return err
			}
			dir, err := anima.Prepare(ctx, anima.Setup{Dir: filepath.Join(chat.ModelsDir(), "Anima"), ConvertOptions: o, Log: e.Log})
			if err != nil {
				return err
			}
			f.Set("export", dir)
		default:
			return fmt.Errorf("no image model of bonsaiai-image generate of the family %q", cmp.Or(m.Family, "(none)"))
		}
		return nil
	})})
}
