// Package recipe reads model-recipes.yaml: how hwnpu-dev makes the models' conversions, what only
// the build needs (models.yaml holds what a run needs). hwnpu-dev models package and hwnpu-dev
// prebuild build with it, so that whoever runs them makes the same files (docs/model-recipes.md).
package recipe

import (
	"fmt"
	"maps"
	"path"
	"regexp"
	"slices"
	"sort"
	"strings"
	"sync"

	"github.com/goccy/go-yaml"

	"github.com/shibukawa/hwnpu"
	devenv "github.com/shibukawa/hwnpu/internal/dev"
)

// File is a recipes file.
type File struct {
	Version int              `yaml:"version"`
	Tools   Tools            `yaml:"tools"`
	Apple   Apple            `yaml:"apple"`
	AMD     AMD              `yaml:"amd"`
	Models  map[string]Model `yaml:"models"`
}

// Tools are the versions of the tools the conversions were checked with.
type Tools struct {
	// CoreAIBuild is coreai-build's version (coreai-build --version: "coreai-build 3605.6.3").
	CoreAIBuild string `yaml:"coreai-build"`
}

// Apple is how Core AI exports are built: their programs prebuilt for Archs, Jobs at a time,
// but what a rule of Prebuild leaves to the device; their operations written as Rewrite says
// for each chip.
//
// A model's has Version, the version of what it makes (raised when Convert or its Code
// changes: testdata/apple-versions.txt), Code, the Go packages that write its programs after the
// file's Code (those every conversion writes with), and Convert, the values its conversion is
// made with.
type Apple struct {
	Archs    []string  `yaml:"archs"`
	Jobs     int       `yaml:"jobs"`
	Prebuild []Rule    `yaml:"prebuild"`
	Rewrite  []Rewrite `yaml:"rewrite"`
	Version  int       `yaml:"version"`
	Code     Strings   `yaml:"code"`
	Convert  Convert   `yaml:"convert"`
}

// Rewrite is how the programs of the chips of Arch write an operation some Neural Engines do not
// run: Op written As one of its Ways. It is chosen at the conversion (an export holds only its
// programs' bytecode), so the chips of different rewrites take different exports, the rewrites
// in their names (Suffix). A chip no rule names takes the operation's default (Ways' first).
type Rewrite struct {
	// Arch are the architectures (every one when empty).
	Arch Strings `yaml:"arch"`
	Op   string  `yaml:"op"`
	As   string  `yaml:"as"`
	// Why is what was seen of the ways on those chips, Checked the day (YYYY-MM-DD).
	Why     string `yaml:"why"`
	Checked string `yaml:"checked"`
}

// Ways are the ways of writing each operation a rule may rewrite, the default first: the models
// that use one write it so (llm.Options.Rewrites).
var Ways = map[string][]string{
	// sin: by range reduction and a series of multiplications and additions (M1's and M2's
	// Neural Engine have no sin), or the operation (M3's and later have it)
	"sin": {"series", "native"},
}

// Model is a model's recipe: its Core AI exports' (Apple) and its exports' for AMD's NPU (AMD).
type Model struct {
	Apple *Apple    `yaml:"apple"`
	AMD   *AMDModel `yaml:"amd"`
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
	if f.Apple.Version != 0 || f.Apple.Convert != nil {
		return nil, fmt.Errorf("recipe: %s: apple: version and convert are a model's (models.NAME.apple)", name)
	}
	if err := f.AMD.check("amd"); err != nil {
		return nil, fmt.Errorf("recipe: %s: %w", name, err)
	}
	for _, m := range f.sortedModels() {
		if a := f.Models[m].Apple; a != nil {
			if err := a.check("models." + m + ".apple"); err != nil {
				return nil, fmt.Errorf("recipe: %s: %w", name, err)
			}
			if a.Version < 1 {
				return nil, fmt.Errorf("recipe: %s: models.%s.apple.version: %d (the version of what it makes, from 1)", name, m, a.Version)
			}
			if err := a.Convert.check("models." + m + ".apple.convert"); err != nil {
				return nil, fmt.Errorf("recipe: %s: %w", name, err)
			}
		}
		if a := f.Models[m].AMD; a != nil {
			if err := a.check("models." + m + ".amd"); err != nil {
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
	for i, r := range a.Rewrite {
		at := fmt.Sprintf("%s.rewrite[%d]", at, i)
		for _, arch := range r.Arch {
			if !slices.Contains(Archs, arch) {
				return fmt.Errorf("%s.arch: no architecture %q", at, arch)
			}
		}
		ways, ok := Ways[r.Op]
		if !ok {
			return fmt.Errorf("%s.op: %q (%s)", at, r.Op, strings.Join(slices.Sorted(maps.Keys(Ways)), " "))
		}
		if !slices.Contains(ways, r.As) {
			return fmt.Errorf("%s.as: %q (%s: %s)", at, r.As, r.Op, strings.Join(ways, " "))
		}
		if r.Why == "" {
			return fmt.Errorf("%s: no why (what was seen of the ways)", at)
		}
		if !day.MatchString(r.Checked) {
			return fmt.Errorf("%s.checked: %q (the day it was seen, YYYY-MM-DD)", at, r.Checked)
		}
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
	a := Apple{Archs: f.Apple.Archs, Jobs: f.Apple.Jobs, Prebuild: slices.Clone(f.Apple.Prebuild), Rewrite: slices.Clone(f.Apple.Rewrite)}
	if m, ok := f.Models[model]; ok && m.Apple != nil {
		if len(m.Apple.Archs) > 0 {
			a.Archs = m.Apple.Archs
		}
		if m.Apple.Jobs > 0 {
			a.Jobs = m.Apple.Jobs
		}
		a.Prebuild = append(a.Prebuild, m.Apple.Prebuild...)
		a.Rewrite = append(a.Rewrite, m.Apple.Rewrite...)
	}
	return a
}

// Rewrites are the ways arch's programs write the operations a rule rewrites to other than its
// default, by the operation (the last rule naming it wins); nil when none.
func (a Apple) Rewrites(arch string) map[string]string {
	var out map[string]string
	for _, r := range a.Rewrite {
		if len(r.Arch) > 0 && !slices.Contains(r.Arch, arch) {
			continue
		}
		if out == nil {
			out = map[string]string{}
		}
		out[r.Op] = r.As
	}
	for op, way := range out {
		if way == Ways[op][0] {
			delete(out, op)
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// RewriteArchs are an architecture of each different Rewrites, of the architectures of
// coreai-build (Archs), one of the defaults first: what a build converts an export for each of,
// so that every chip finds its own.
func (a Apple) RewriteArchs() []string {
	var out, seen []string
	for _, arch := range Archs {
		s := Suffix(a.Rewrites(arch))
		if !slices.Contains(seen, s) {
			seen, out = append(seen, s), append(out, arch)
		}
	}
	for i, s := range seen {
		if s == "" {
			out[0], out[i] = out[i], out[0]
		}
	}
	return out
}

// Suffix is what an export's name takes of rewrites, of the operations ops when given (those its
// programs have): "-sin-native" for {sin: native}, the operations in order; "" for none.
func Suffix(rewrites map[string]string, ops ...string) string {
	var b strings.Builder
	for _, op := range slices.Sorted(maps.Keys(rewrites)) {
		if len(ops) == 0 || slices.Contains(ops, op) {
			fmt.Fprintf(&b, "-%s-%s", op, rewrites[op])
		}
	}
	return b.String()
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

// RewritesFor are the rewrites of model's programs for arch (Apple.Rewrites of the built-in
// recipes); for this Mac's (here), with the developers' rewrite items over them (HWNPU_DEV=rewrite=sin:series),
// to compare the ways on one chip.
func RewritesFor(model, arch string, here bool) (map[string]string, error) {
	f, err := Builtin()
	if err != nil {
		return nil, err
	}
	rw := f.AppleOf(model).Rewrites(arch)
	if v := devenv.Value("rewrite"); here && v != "" {
		rw = maps.Clone(rw)
		if rw == nil {
			rw = map[string]string{}
		}
		for _, it := range strings.Split(v, "+") {
			op, way, _ := strings.Cut(it, ":")
			if !slices.Contains(Ways[op], way) {
				return nil, fmt.Errorf("%s rewrite=%s: no way %q of %q (%v)", devenv.Env, v, way, op, Ways)
			}
			rw[op] = way
			if way == Ways[op][0] {
				delete(rw, op)
			}
		}
	}
	return rw, nil
}
