package prep

import (
	"cmp"
	"fmt"
	"path/filepath"
	"slices"
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
// FLUX.2 klein an export of each aspect of models.yaml npu.apple (1:1 and 1:1s one) with the
// edit's programs beside each, FLUX.3 klein its one, Chroma one for 1:1 and one for 1:1s, Anima
// one of every aspect.
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
			spec := m.NPU["apple"]
			var dirs []string
			for _, a := range as {
				setup := flux2klein.Setup{Dir: filepath.Join(chat.ModelsDir(), "FLUX.2-klein"), Repo: m.Repo, Name: m.Name,
					ConvertOptions: flux2klein.ConvertOptions{Width: a.Width, Height: a.Height, TextFormat: spec.Text, DitFormat: spec.Dit}, Log: e.Log}
				dir, err := flux2klein.Prepare(ctx, setup)
				if err != nil {
					return fmt.Errorf("%dx%d: %w", a.Width, a.Height, err)
				}
				if slices.Contains(dirs, dir) {
					continue // 1:1 and 1:1s are one export
				}
				dirs = append(dirs, dir)
				man, err := flux2klein.ReadProgramManifest(dir)
				if err != nil {
					return err
				}
				ed, err := flux2klein.PrepareEdit(ctx, setup, flux2klein.EditOptions{Sizes: man.SizeList(), DitFormat: man.Formats["dit"]})
				if err != nil {
					return fmt.Errorf("%dx%d: the edit: %w", a.Width, a.Height, err)
				}
				f.Set("edit", ed)
			}
			f.Set("export", dirs...)
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
