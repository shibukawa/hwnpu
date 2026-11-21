package modelscmd

import (
	"path/filepath"
	"slices"
	"testing"

	"github.com/alecthomas/kong"

	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/devopt"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/backend/coreai"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
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

// TestRoles: every kind package takes is found a role (the packages register theirs), and the
// targets are NPUs that convert without a device.
func TestRoles(t *testing.T) {
	for _, name := range []string{"LFM3-1.7B", "plamo-3-translate-2b", "whisper-large-v3-turbo", "LFM3-Audio-2B-TTS", "FLUX.2-klein-4B"} {
		m, ok := chat.Lookup(name)
		if !ok {
			t.Fatalf("no %s", name)
		}
		if airole.For(m) == nil {
			t.Errorf("%s (%s): no role", name, m.Kind)
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
