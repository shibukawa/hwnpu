package modelscmd

import (
	"path/filepath"
	"slices"
	"testing"

	"github.com/alecthomas/kong"

	"github.com/shibukawa/bonsaiai/cmd/bonsaiai-dev/devopt"
	"github.com/shibukawa/bonsaiai/internal/airole"
	"github.com/shibukawa/bonsaiai/internal/backend/coreai"
	"github.com/shibukawa/bonsaiai/internal/cache"
	"github.com/shibukawa/bonsaiai/internal/chat"
	"github.com/shibukawa/bonsaiai/internal/prep"
	"github.com/shibukawa/bonsaiai/internal/recipe"
)

func TestParseJobs(t *testing.T) {
	jobs, err := parseJobs([]string{"LFM3-1.7B/coreai", "whisper-large-v3-turbo/amd-xdna2", "LFM3-1.7B/coreai-h14g", "LFM3-1.7B/coreai-h13g"})
	if err != nil {
		t.Fatal(err)
	}
	if len(jobs) != 2 || jobs[0].model.Name != "LFM3-1.7B" || jobs[0].target != cache.CoreAITarget || !slices.Equal(jobs[0].archs, []string{"h14g", "h13g"}) || jobs[1].target != cache.AMDTarget {
		t.Errorf("jobs = %+v", jobs)
	}
	if jobs, err := parseJobs([]string{"LFM3-1.7B/coreai-all"}); err != nil || len(jobs) != 1 || len(jobs[0].archs) != len(coreai.Archs) {
		t.Errorf("coreai-all = %+v, %v", jobs, err)
	}
	// No target named: the Core AI exports, compiled for the recipe's architectures for a model
	// its recipe prebuilds alone (an image generator), on a Mac; the architectures are of the
	// Core AI job, not of the AMD one beside it
	recipes, err := recipe.Builtin()
	if err != nil {
		t.Fatal(err)
	}
	for model, prebuilt := range map[string]bool{"FLUX.2-klein-4B": true, "flux-3-image-klein": true, "Anima-Turbo": true, "Anima-Aesthetic": true,
		"LFM3-1.7B": false, "whisper-large-v3-turbo": false, "MOSS-SoundEffect-v2.0": false, "DA3-BASE": false} {
		jobs, err := parseJobs([]string{model})
		if err != nil {
			t.Errorf("%s: %v", model, err)
			continue
		}
		var want []string
		if prebuilt && canCompile() == nil {
			want = recipes.AppleOf(model).Archs
			if len(want) == 0 {
				t.Errorf("%s: the recipe names no architectures", model)
			}
		}
		core := 0
		for _, j := range jobs {
			switch {
			case j.target == cache.CoreAITarget:
				core++
				if !slices.Equal(j.archs, want) {
					t.Errorf("%s: compiled for %v, want %v", model, j.archs, want)
				}
			case len(j.archs) > 0:
				t.Errorf("%s: %s has architectures %v", model, j.target, j.archs)
			}
		}
		if core != 1 {
			t.Errorf("%s: %d Core AI jobs in %+v", model, core, jobs)
		}
	}
	for _, bad := range []string{"no-such-model", "LFM3-1.7B/coreai-h99", "LFM3-1.7B/intel"} {
		if _, err := parseJobs([]string{bad}); err == nil {
			t.Errorf("%s: no error", bad)
		}
	}
}

// TestParseSel: a Core AI export folder is compiled alone, for the architecture after it.
func TestParseSel(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "x.coreai")
	jobs, folders, err := parseSel([]string{dir + "/coreai-h14g", dir, "LFM3-1.7B/coreai"})
	if err != nil {
		t.Fatal(err)
	}
	if len(jobs) != 1 || len(folders) != 2 || folders[0].paths[0] != dir || !slices.Equal(folders[0].archs, []string{"h14g"}) || folders[1].archs != nil {
		t.Errorf("jobs %+v, folders %+v", jobs, folders)
	}
}

// TestForms: LFM3-1.7B has no whole program on 8 GB (tier: none) and has one from 16 GB, two
// exports; asked for 16 GB alone, one.
func TestForms(t *testing.T) {
	m, ok := chat.Lookup("LFM3-1.7B")
	if !ok {
		t.Fatal("no LFM3-1.7B")
	}
	fs := forms(m, nil)
	if len(fs) != 2 || fs[0].Run.Tier != "none" || fs[1].Run.Tier != "auto" {
		t.Errorf("forms = %+v", fs)
	}
	if fs := forms(m, []int{16, 32}); len(fs) != 1 || fs[0].Run.Tier != "auto" {
		t.Errorf("forms of 16 and 32 GB = %+v", fs)
	}
}

// TestRoles: every model of the definitions has what package makes it with for Core AI (prep,
// else a role of the library), and the targets are NPUs that convert without a device.
func TestRoles(t *testing.T) {
	for _, m := range chat.Catalog {
		if prep.For(m, cache.CoreAITarget) == nil && airole.For(m) == nil {
			t.Errorf("%s (%s): nothing makes it for coreai", m.Name, m.Kind)
		}
	}
	n, err := airole.ForTarget(cache.CoreAITarget)
	if err != nil || n.Format != "coreai" || n.AMD() {
		t.Errorf("coreai = %+v, %v", n, err)
	}
	if n, err := airole.ForTarget(cache.AMDTarget); err != nil || !n.AMD() {
		t.Errorf("amd-xdna2 = %+v, %v", n, err)
	}
}

func TestActAsChip(t *testing.T) {
	was := chat.Chip()
	undo := chat.ActAsChip("M1")
	if chat.Chip() != "M1" {
		t.Errorf("chip = %q", chat.Chip())
	}
	undo()
	if chat.Chip() != was {
		t.Errorf("chip after = %q, want %q", chat.Chip(), was)
	}
}

func TestCoreAIPaths(t *testing.T) {
	e := filepath.Join("m", "a.coreai")
	got := coreAIPaths(map[string][]string{
		"packages": {e, filepath.Join(e, "model.aimodel")},
		"encoder":  {filepath.Join("m", "enc.aimodel")},
		"config":   {filepath.Join("r", "config.json")},
	})
	slices.Sort(got)
	if want := []string{e, filepath.Join("m", "enc.aimodel")}; !slices.Equal(got, want) {
		t.Errorf("paths = %v, want %v", got, want)
	}
}

// TestParse: the commands parse under models with the definitions' flag bound.
func TestParse(t *testing.T) {
	var c struct {
		Models Cmd `cmd:""`
	}
	p, err := kong.New(&c, kong.Vars(devopt.Vars))
	if err != nil {
		t.Fatal(err)
	}
	k, err := p.Parse([]string{"models", "--models", "x.yaml", "package", "LFM3-1.7B/coreai", "--memory", "16", "--chip", "M1"})
	if err != nil {
		t.Fatal(err)
	}
	if k.Command() != "models package <MODEL[/TARGET]>" || c.Models.Models != "x.yaml" || !slices.Equal(c.Models.Package.Memory, []int{16}) {
		t.Errorf("parsed %q %+v", k.Command(), c.Models)
	}
	for _, args := range [][]string{{"models", "push", "LFM3-1.7B"}, {"models", "hf", "plan"}, {"models", "fetch", "Qwen3-0.6B/coreai", "--no-checkpoint", "--chip", "M1"}, {"models", "package", "x.coreai/coreai-all", "-j", "2", "--no-push"}, {"models", "fetch", "LFM3-1.7B/coreai-h14g"}} {
		if _, err := p.Parse(args); err != nil {
			t.Errorf("%v: %v", args, err)
		}
	}
}

// Every model that has Core AI exports (a preparer or a role makes them) has a version of them in
// model-recipes.yaml (apple.version), so that a change of what makes them is seen
// (testdata/apple-versions.txt); its Targets carry it to the sources (vN).
func TestAppleVersionsOfEveryModel(t *testing.T) {
	recipes, err := recipe.Builtin()
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, m := range chat.Catalog {
		if prep.For(m, cache.CoreAITarget) == nil && airole.For(m) == nil {
			continue
		}
		n++
		v := recipes.AppleVersion(m.Recipe)
		if v == 0 {
			t.Errorf("%s: no apple.version in model-recipes.yaml (models.%s.apple)", m.Name, m.Name)
		} else if m.Targets[cache.CoreAITarget] != v {
			t.Errorf("%s: Targets[coreai] %d, its recipe's version %d", m.Name, m.Targets[cache.CoreAITarget], v)
		}
	}
	if n < 40 {
		t.Errorf("%d models with Core AI exports", n)
	}
}
