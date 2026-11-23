package recipe

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	amdtools "github.com/shibukawa/hwnpu/tools/amd"
)

// AMD is how the exports for AMD's NPU are made: for each generation of Generations, with the
// arguments Extra gives that generation after every script's own.
type AMD struct {
	Generations []string `yaml:"generations"`
	// Extra are arguments of every script for a generation (xdna1: --device npu1).
	Extra map[string][]string `yaml:"extra"`
}

// AMDModel is how a model's export for AMD's NPU is made: Script of tools/amd with Args, those
// of a machine of more memory replaced by Variants (by its GB, as models.yaml's variants), and
// Extra's of the generation after them. Generations, when given, replace the file's; Skip names
// the generations the export is not made for. Version is the version of what it makes (the
// target's, models.yaml targets did it before: the sources keep the export as vN), raised when
// the recipe changes (testdata/amd-versions.txt holds what each version is).
type AMDModel struct {
	Version     int                 `yaml:"version"`
	Generations []string            `yaml:"generations"`
	Script      string              `yaml:"script"`
	Args        Strings             `yaml:"args"`
	Variants    map[int]Strings     `yaml:"variants"`
	Extra       map[string][]string `yaml:"extra"`
	Skip        []Skip              `yaml:"skip"`
}

// Skip is a generation a model's export is not made for: Why it is not (what was seen), Checked
// the day (YYYY-MM-DD).
type Skip struct {
	Generation string `yaml:"generation"`
	Why        string `yaml:"why"`
	Checked    string `yaml:"checked"`
}

// Generations are the generations of AMD's NPU a recipe may name: xdna1 (Phoenix, Hawk Point:
// IRON's npu1) and xdna2 (Strix, Krackan: npu2).
var Generations = []string{"xdna1", "xdna2"}

// Built are the generations hwnpu runs the exports of, so that the build makes them: those of
// the others are skipped (amd-xdna1 is not read yet).
var Built = []string{"xdna2"}

// AMDTarget is the target of generation gen's exports (amd-xdna2).
func AMDTarget(gen string) string { return "amd-" + gen }

// GenerationOf is the generation of target amd-GEN, "" for another target.
func GenerationOf(target string) string {
	gen, ok := strings.CutPrefix(target, "amd-")
	if !ok || !slices.Contains(Generations, gen) {
		return ""
	}
	return gen
}

// The placeholders of a recipe's args, each a whole argument, put in by the code that runs the
// script: In what it prepared for it (the checkpoint's folder, the ONNX file), Out the export's
// folder (left as it is: chat.ExportOut), and values of the model's kind (Sizes, the sizes of
// models.yaml's npu.amd or of the code; Size; Outputs, the ONNX outputs the code chose; Text,
// the text encoder's format; DiT, the DiT's file). Sizes and Outputs are lists.
const (
	In      = "{in}"
	Out     = "{out}"
	Sizes   = "{sizes}"
	Size    = "{size}"
	Outputs = "{outputs}"
	Text    = "{text}"
	DiT     = "{dit}"
)

var placeholders = []string{In, Out, Sizes, Size, Outputs, Text, DiT}

// isPlaceholder reports whether arg is written as one ({...}).
func isPlaceholder(arg string) bool {
	return strings.HasPrefix(arg, "{") && strings.HasSuffix(arg, "}")
}

func (a AMD) check(at string) error {
	if err := checkGenerations(at+".generations", a.Generations); err != nil {
		return err
	}
	return checkExtra(at+".extra", a.Extra)
}

func checkGenerations(at string, gens []string) error {
	for _, g := range gens {
		if !slices.Contains(Generations, g) {
			return fmt.Errorf("%s: no generation %q (%s)", at, g, strings.Join(Generations, " "))
		}
	}
	return nil
}

func checkExtra(at string, extra map[string][]string) error {
	for g, args := range extra {
		if !slices.Contains(Generations, g) {
			return fmt.Errorf("%s: no generation %q (%s)", at, g, strings.Join(Generations, " "))
		}
		if err := checkArgs(at+"."+g, args); err != nil {
			return err
		}
	}
	return nil
}

func checkArgs(at string, args []string) error {
	for _, a := range args {
		if strings.ContainsAny(a, "{}") && !slices.Contains(placeholders, a) {
			return fmt.Errorf("%s: %q: a placeholder is a whole argument, one of %s", at, a, strings.Join(placeholders, " "))
		}
	}
	return nil
}

func (m AMDModel) check(at string) error {
	switch {
	case m.Script == "":
		return fmt.Errorf("%s: no script (one of tools/amd)", at)
	case !amdtools.Has(m.Script):
		return fmt.Errorf("%s.script: %q is not a script of tools/amd", at, m.Script)
	}
	if m.Version < 0 {
		return fmt.Errorf("%s.version: %d", at, m.Version)
	}
	if err := checkGenerations(at+".generations", m.Generations); err != nil {
		return err
	}
	if err := checkArgs(at+".args", m.Args); err != nil {
		return err
	}
	if !slices.Contains(m.Args, Out) {
		return fmt.Errorf("%s.args: no %s (the export's folder)", at, Out)
	}
	for gb, args := range m.Variants {
		at := fmt.Sprintf("%s.variants.%d", at, gb)
		if err := checkArgs(at, args); err != nil {
			return err
		}
		if !slices.Contains(args, Out) {
			return fmt.Errorf("%s: no %s (the export's folder)", at, Out)
		}
	}
	if err := checkExtra(at+".extra", m.Extra); err != nil {
		return err
	}
	for i, s := range m.Skip {
		at := fmt.Sprintf("%s.skip[%d]", at, i)
		if !slices.Contains(Generations, s.Generation) {
			return fmt.Errorf("%s.generation: %q (%s)", at, s.Generation, strings.Join(Generations, " "))
		}
		if s.Why == "" {
			return fmt.Errorf("%s: no why (what was seen of it on that generation)", at)
		}
		if !day.MatchString(s.Checked) {
			return fmt.Errorf("%s.checked: %q (the day it was seen, YYYY-MM-DD)", at, s.Checked)
		}
	}
	return nil
}

// AMDOf is model's recipe for AMD's NPU, if it has one.
func (f *File) AMDOf(model string) (AMDModel, bool) {
	m, ok := f.Models[model]
	if !ok || m.AMD == nil {
		return AMDModel{}, false
	}
	return *m.AMD, true
}

// AMDGenerations are the generations model's export is made for: its own, else the file's,
// less those it skips; and the skips.
func (f *File) AMDGenerations(model string) ([]string, []Skip) {
	m, ok := f.AMDOf(model)
	if !ok {
		return nil, nil
	}
	gens := m.Generations
	if len(gens) == 0 {
		gens = f.AMD.Generations
	}
	var out []string
	for _, g := range gens {
		if !slices.ContainsFunc(m.Skip, func(s Skip) bool { return s.Generation == g }) {
			out = append(out, g)
		}
	}
	return out, m.Skip
}

// Command is a script of tools/amd with its arguments, chat.ExportOut ({out}) standing for the
// export's folder; Options are the arguments but those of In and Out (what keys an export).
type Command struct {
	Script  string
	Args    []string
	Options []string
}

// AMDCommand is how model's export for generation gen of AMD's NPU is made on a machine of gb
// GB of memory (0: the model's own; a variant's args from its GB up): the recipe's args with the
// values of vars put in for their placeholders (every one but Out must be given), then the
// generation's extra arguments, the file's and the model's.
func (f *File) AMDCommand(model string, gb int, gen string, vars map[string][]string) (Command, error) {
	m, ok := f.AMDOf(model)
	if !ok {
		return Command{}, fmt.Errorf("recipe: %s has no recipe for AMD's NPU (model-recipes.yaml models.%s.amd)", model, model)
	}
	if !slices.Contains(Generations, gen) {
		return Command{}, fmt.Errorf("recipe: no generation %q of AMD's NPU (%s)", gen, strings.Join(Generations, " "))
	}
	args := append(slices.Clone(m.argsOn(gb)), f.AMDExtra(model, gen)...)
	c := Command{Script: m.Script}
	for _, a := range args {
		switch {
		case a == Out:
			c.Args = append(c.Args, Out)
		case isPlaceholder(a):
			v, ok := vars[a]
			if !ok {
				return Command{}, fmt.Errorf("recipe: %s: the args of %s name %s, which its code does not give (%s)", model, m.Script, a, strings.Join(keys(vars), " "))
			}
			c.Args = append(c.Args, v...)
			if a != In {
				c.Options = append(c.Options, v...)
			}
		default:
			c.Args = append(c.Args, a)
			c.Options = append(c.Options, a)
		}
	}
	return c, nil
}

// argsOn are the args of a machine of gb GB: those of the variant of the most memory up to gb,
// else the model's.
func (m AMDModel) argsOn(gb int) []string {
	best := 0
	for v := range m.Variants {
		if v <= gb && v > best {
			best = v
		}
	}
	if best > 0 {
		return m.Variants[best]
	}
	return m.Args
}

func keys(m map[string][]string) []string {
	var out []string
	for k := range m {
		out = append(out, k)
	}
	slices.Sort(out)
	return out
}

// AMDExtra are the arguments of generation gen after model's own: the file's, then the model's.
func (f *File) AMDExtra(model, gen string) []string {
	m, _ := f.AMDOf(model)
	return append(slices.Clone(f.AMD.Extra[gen]), m.Extra[gen]...)
}

// Fingerprint is what the version of a model's recipe stands for: the SHA-256 of all of it but
// the version, in hex (testdata/amd-versions.txt).
func (m AMDModel) Fingerprint() string {
	m.Version = 0
	b, _ := json.Marshal(m) // a map's keys in order
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}
