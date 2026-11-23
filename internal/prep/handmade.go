package prep

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"

	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/imagegen/anima"
	"github.com/shibukawa/hwnpu/internal/imagegen/aspect"
	"github.com/shibukawa/hwnpu/internal/recipe"
)

// The exports for AMD's NPU that their command takes as made by hand with a script of tools/amd
// (chat.AMDPictureExport: here, or put in the sources): package runs the script as one would by
// hand, in IRON's environment, on the checkpoint it downloads, and puts the export in the sources
// written back to as made by hand (chat.ShareMadeByHand), under the name the command looks for.
func init() {
	amd := []string{cache.AMDTarget}
	Register(&Preparer{Name: "depth", Kinds: []chat.Kind{chat.KindDepth}, Targets: amd, Prepare: amdOnly(func(e airole.PrepareEnv, f *airole.Files) error {
		m := e.Model
		tool, files := "iron_da3.py", []string{"config.json", "model.safetensors"}
		if m.Family == "moge" {
			tool, files = "iron_moge.py", []string{"model.pt"}
		}
		dir, err := handMade(e, tool, func() (string, error) { return fetchRepo(e, files) }, nil)
		f.Set("export", dir)
		return err
	})})
	Register(&Preparer{Name: "label", Kinds: []chat.Kind{chat.KindLabel}, Targets: amd, Prepare: amdOnly(func(e airole.PrepareEnv, f *airole.Files) error {
		dir, err := handMade(e, "iron_siglip2.py", func() (string, error) {
			return fetchRepo(e, []string{"config.json", "tokenizer.json", "model.safetensors"})
		}, nil)
		f.Set("export", dir)
		return err
	})})
	Register(&Preparer{Name: "faces", Kinds: []chat.Kind{chat.KindFace}, Targets: amd, Prepare: onAMDNPU(func(e airole.PrepareEnv, f *airole.Files) error {
		dir, err := handMade(e, "iron_yunet.py", func() (string, error) {
			return chat.Fetch(e.Ctx, e.Model.Source(), "face_detection_yunet_2023mar.onnx", e.Log)
		}, nil)
		f.Set("export", dir)
		return err
	})})
}

// imageHandMade is the export of every aspect ratio of Anima's model m for AMD's NPU
// (tools/amd/iron_anima.py as its recipe says: {sizes} those of aspect.Of(16), {dit} the model's
// file).
func imageHandMade(e airole.PrepareEnv) (string, error) {
	var sizes []string
	for _, a := range aspect.Of(16) {
		if a.Width == a.Height {
			sizes = append(sizes, fmt.Sprint(a.Width))
		} else {
			sizes = append(sizes, fmt.Sprintf("%dx%d", a.Width, a.Height))
		}
	}
	vars := map[string][]string{recipe.Sizes: sizes}
	// one of an earlier hwnpu by the same name may hold 1024 x 1024 alone: made again
	holds := func(sizesOf func(string) [][2]int) func(string) bool {
		return func(dir string) bool {
			have := sizesOf(dir)
			for _, a := range aspect.Of(16) {
				if !slices.Contains(have, [2]int{a.Width, a.Height}) {
					return false
				}
			}
			return true
		}
	}
	return handMadeOf(e, "iron_anima.py", func() (string, error) {
		var want []chat.File
		for _, file := range anima.Files(e.Model.File) {
			want = append(want, chat.File{Repo: file.Repo, Name: file.Name})
		}
		if err := chat.FetchFiles(e.Ctx, want, e.Log); err != nil {
			return "", err
		}
		return animaRoot()
	}, withDiT(vars, e.Model.File), holds(anima.IronSizes))
}

// withDiT is vars with recipe.DiT file.
func withDiT(vars map[string][]string, file string) map[string][]string {
	vars[recipe.DiT] = []string{file}
	return vars
}

// animaRoot is the folder tools/amd/iron_anima.py reads (diffusion_models/, text_encoders/,
// vae/, diffusers/), links to the two repositories' files under the cache.
func animaRoot() (string, error) {
	root := cache.Sub("work", "anima-root")
	split := filepath.Join(chat.RepoDir(anima.RepoSplit), "split_files")
	links := map[string]string{
		"diffusion_models": filepath.Join(split, "diffusion_models"),
		"text_encoders":    filepath.Join(split, "text_encoders"),
		"vae":              filepath.Join(split, "vae"),
		"diffusers":        chat.RepoDir(anima.RepoDiffusers),
	}
	if err := os.MkdirAll(root, 0o755); err != nil {
		return "", err
	}
	for name, to := range links {
		at := filepath.Join(root, name)
		os.Remove(at)
		if err := os.Symlink(to, at); err != nil {
			return "", err
		}
	}
	return root, nil
}

// fetchRepo downloads files of the model's repository and returns its folder (chat.RepoDir).
func fetchRepo(e airole.PrepareEnv, files []string) (string, error) {
	for _, file := range files {
		if _, err := chat.Fetch(e.Ctx, e.Model.Source(), file, e.Log); err != nil {
			return "", err
		}
	}
	return chat.RepoDir(e.Model.Source()), nil
}

// handMade is the export of tool for model m by the name its command looks for
// (chat.AMDPictureNames): here or in the sources (chat.AMDPictureExport), else made as its recipe
// says (model-recipes.yaml, whose script must be tool) on what fetch gives (recipe.In) and vars,
// in IRON's environment, and put in the sources written back to as made by hand.
func handMade(e airole.PrepareEnv, tool string, fetch func() (string, error), vars map[string][]string) (string, error) {
	return handMadeOf(e, tool, fetch, vars, nil)
}

// handMadeOf is handMade that takes an export here or in the sources only when holds says it
// is whole (nil: any).
func handMadeOf(e airole.PrepareEnv, tool string, fetch func() (string, error), vars map[string][]string, holds func(dir string) bool) (string, error) {
	m := e.Model
	ctx := chat.WithModel(e.Ctx, m)
	if dir, err := chat.AMDPictureExport(ctx, m, "tools/amd/"+tool, 0, e.Log); err == nil && (holds == nil || holds(dir)) {
		return dir, nil
	}
	python, err := chat.IRONPython(ctx)
	if err != nil {
		return "", fmt.Errorf("no export of tools/amd/%s here or in the sources, and making one needs IRON: %w (INSTALL_AMD.md, %s)", tool, err, chat.IRONEnv)
	}
	root, err := fetch()
	if err != nil {
		return "", err
	}
	out := filepath.Join(chat.AMDRoot(), chat.AMDPictureNames(m, 0)[0])
	if vars == nil {
		vars = map[string][]string{}
	}
	vars[recipe.In] = []string{root}
	if err := chat.RunAMDRecipe(ctx, python, m, vars, out, e.Log, tool); err != nil {
		return "", err
	}
	chat.ShareMadeByHand(m, out, e.Log)
	return out, nil
}
