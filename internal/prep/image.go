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
// FLUX.2 klein one of every size and edit of models.yaml npu.apple and one beside it of the edits
// of 1024, Anima one of
// every aspect, Chroma one of every size of npu.apple, FLUX.3 klein as hwnpu-image generate makes
// it now (its default sizes, until it holds every aspect in one as
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
			// every size and edit of models.yaml npu.apple in one export (the weights once), as
			// hwnpu-image generate makes it, and the export beside it of the edits of a reference of
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
		case strings.HasPrefix(m.Family, "chroma"):
			// every size of models.yaml npu.apple in one export (the weights once), 1:1 first, as
			// ai/imagegen makes it
			as, err := flux2klein.NPUAspects(m, "apple", flux2klein.NPUGeneration("apple"))
			if err != nil {
				return err
			}
			dir, err := chroma.Prepare(ctx, chroma.Setup{Dir: filepath.Join(chat.ModelsDir(), "Chroma"), Repo: m.Repo, Name: m.Name,
				ConvertOptions: chroma.ConvertOptions{Backend: llm.BackendCoreAI, Width: as[0].Width, Height: as[0].Height, Sizes: aspect.SizesOf(as)}, Log: e.Log})
			if err != nil {
				return err
			}
			f.Set("export", dir)
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
