// Package recipe reads model-recipes.yaml: how hwnpu-dev makes the models' conversions, what only
// the build needs (models.yaml holds what a run needs). hwnpu-dev models package and hwnpu-dev
// prebuild build with it, so that whoever runs them makes the same files (docs/model-recipes.md).
package recipe

import (
	"fmt"
	"path"
	"regexp"
	"slices"
	"sort"
	"strings"
	"sync"

	"github.com/goccy/go-yaml"

	"github.com/shibukawa/hwnpu"
)

// File is a recipes file.
type File struct {
	Version int              `yaml:"version"`
	Tools   Tools            `yaml:"tools"`
	Apple   Apple            `yaml:"apple"`
	Models  map[string]Model `yaml:"models"`
}

// Tools are the versions of the tools the conversions were checked with.
type Tools struct {
	// CoreAIBuild is coreai-build's version (coreai-build --version: "coreai-build 3605.6.3").
	CoreAIBuild string `yaml:"coreai-build"`
}

// Apple is how Core AI exports are built: their programs prebuilt for Archs, Jobs at a time,
// but what a rule of Prebuild leaves to the device.
type Apple struct {
	Archs    []string `yaml:"archs"`
	Jobs     int      `yaml:"jobs"`
	Prebuild []Rule   `yaml:"prebuild"`
}

// Model is a model's recipe.
type Model struct {
	Apple *Apple `yaml:"apple"`
}

// Rule is what becomes of some programs of some exports for some architectures.
type Rule struct {
	// Arch are the architectures (every one when empty).
	Arch Strings `yaml:"arch"`
	// Export is a pattern of the exports' names (path.Match; every export when empty).
	Export string `yaml:"export"`
	// Programs are patterns of the programs' names without .aimodel (path.Match).
	Programs Strings `yaml:"programs"`
	// Do is what is done with them: Device.
	Do string `yaml:"do"`
	// Why is what the build did with them, Checked the day it was seen (YYYY-MM-DD).
	Why     string `yaml:"why"`
	Checked string `yaml:"checked"`
}

// Device is the rule's Do for programs the device's compiler compiles at their first run, not
// prebuilt.
const Device = "device"

// Strings is a list of strings, written as one or as a list.
type Strings []string

// UnmarshalYAML reads one string or a list of them.
func (s *Strings) UnmarshalYAML(b []byte) error {
	var one string
	if err := yaml.Unmarshal(b, &one); err == nil {
		*s = Strings{one}
		return nil
	}
	var list []string
	if err := yaml.Unmarshal(b, &list); err != nil {
		return err
	}
	*s = list
	return nil
}

// Archs are the architectures a rule may name (coreai-build's, internal/backend/coreai.Archs;
// the backend's list is not imported, so that the package builds without it).
var Archs = []string{
	"h13c", "h13d", "h13g", "h13s",
	"h14c", "h14d", "h14g", "h14s",
	"h15c", "h15d", "h15g", "h15m", "h15s",
	"h16c", "h16g", "h16s",
	"h17c", "h17d", "h17g", "h17p", "h17s",
	"h18g",
}

var day = regexp.MustCompile(`^\d{4}-\d{2}-\d{2}$`)

// Parse reads a recipes file named name and checks it.
func Parse(name string, raw []byte) (*File, error) {
	var f File
	if err := yaml.UnmarshalWithOptions(raw, &f, yaml.DisallowUnknownField()); err != nil {
		return nil, fmt.Errorf("recipe: %s: %s", name, yaml.FormatError(err, false, true))
	}
	if f.Version != 1 {
		return nil, fmt.Errorf("recipe: %s: version %d (1 is read)", name, f.Version)
	}
	if err := f.Apple.check("apple"); err != nil {
		return nil, fmt.Errorf("recipe: %s: %w", name, err)
	}
	for _, m := range f.sortedModels() {
		if a := f.Models[m].Apple; a != nil {
			if err := a.check("models." + m + ".apple"); err != nil {
				return nil, fmt.Errorf("recipe: %s: %w", name, err)
			}
		}
	}
	return &f, nil
}

func (a Apple) check(at string) error {
	for _, arch := range a.Archs {
		if !slices.Contains(Archs, arch) {
			return fmt.Errorf("%s.archs: no architecture %q (%s)", at, arch, strings.Join(Archs, " "))
		}
	}
	if a.Jobs < 0 {
		return fmt.Errorf("%s.jobs: %d", at, a.Jobs)
	}
	for i, r := range a.Prebuild {
		at := fmt.Sprintf("%s.prebuild[%d]", at, i)
		for _, arch := range r.Arch {
			if !slices.Contains(Archs, arch) {
				return fmt.Errorf("%s.arch: no architecture %q", at, arch)
			}
		}
		if len(r.Programs) == 0 {
			return fmt.Errorf("%s: no programs", at)
		}
		for _, p := range append([]string{r.Export}, r.Programs...) {
			if _, err := path.Match(p, ""); err != nil {
				return fmt.Errorf("%s: pattern %q: %v", at, p, err)
			}
		}
		if r.Do != Device {
			return fmt.Errorf("%s.do: %q (device: compiled on the device)", at, r.Do)
		}
		if r.Why == "" {
			return fmt.Errorf("%s: no why (what the build did with the programs)", at)
		}
		if !day.MatchString(r.Checked) {
			return fmt.Errorf("%s.checked: %q (the day it was seen, YYYY-MM-DD)", at, r.Checked)
		}
	}
	return nil
}

func (f *File) sortedModels() []string {
	var out []string
	for m := range f.Models {
		out = append(out, m)
	}
	sort.Strings(out)
	return out
}

var (
	builtinOnce sync.Once
	builtin     *File
	builtinErr  error
)

// Builtin is the embedded model-recipes.yaml.
func Builtin() (*File, error) {
	builtinOnce.Do(func() { builtin, builtinErr = Parse("model-recipes.yaml", hwnpu.RecipesYAML) })
	return builtin, builtinErr
}

// AppleOf is how model's Core AI exports are built: the file's apple, with the model's archs and
// jobs over its own and the model's rules after its own. A model without a recipe is built as
// apple says.
func (f *File) AppleOf(model string) Apple {
	a := Apple{Archs: f.Apple.Archs, Jobs: f.Apple.Jobs, Prebuild: slices.Clone(f.Apple.Prebuild)}
	if m, ok := f.Models[model]; ok && m.Apple != nil {
		if len(m.Apple.Archs) > 0 {
			a.Archs = m.Apple.Archs
		}
		if m.Apple.Jobs > 0 {
			a.Jobs = m.Apple.Jobs
		}
		a.Prebuild = append(a.Prebuild, m.Apple.Prebuild...)
	}
	return a
}

// Left is the rule that leaves the program (its name, X or X.aimodel) of the export (its name,
// NAME.coreai) to the device on arch, if one does.
func (a Apple) Left(export, program, arch string) (Rule, bool) {
	program = strings.TrimSuffix(program, ".aimodel")
	for _, r := range a.Prebuild {
		if r.Do != Device || len(r.Arch) > 0 && !slices.Contains(r.Arch, arch) {
			continue
		}
		if r.Export != "" {
			if ok, _ := path.Match(r.Export, export); !ok {
				continue
			}
		}
		for _, p := range r.Programs {
			if ok, _ := path.Match(p, program); ok {
				return r, true
			}
		}
	}
	return Rule{}, false
}

// Suggest is the rule a recipe would take to leave program to the device on arch, as YAML for
// model's prebuild: what a build that failed on it says.
func Suggest(model, program, arch, why, today string) string {
	program = strings.TrimSuffix(program, ".aimodel")
	why = strings.ReplaceAll(strings.Join(strings.Fields(why), " "), `"`, `'`)
	return fmt.Sprintf("models:\n  %s:\n    apple:\n      prebuild:\n        - arch: %s\n          programs: [%s]\n          do: device\n          why: \"%s\"\n          checked: %s\n", model, arch, program, why, today)
}
