package chat

import (
	"errors"
	"fmt"
	"io"
	"maps"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"strings"

	"github.com/goccy/go-yaml"

	"github.com/shibukawa/hwnpu"
)

// The definitions of the models (docs/model-table.md): hwnpu's models.yaml, embedded, and files
// of the user's that add to it and override it, written as differences (UserModelsFile,
// ModelsEnv, -models). Catalog, Categories and the aliases are made from them.

// ModelsEnv lists more definition files, separated as PATH is (':', ';' on Windows).
const ModelsEnv = "HWNPU_MODELS"

// ModelsUsage is the usage of the -models flag of the commands that pick models.
const ModelsUsage = "definition files read over the built-in models.yaml and yours (~/.config/hwnpu/models.yaml, %LOCALAPPDATA%\\hwnpu\\models.yaml on Windows, HWNPU_MODELS), comma separated: models, uses and aliases of your own (docs/model-table.md)"

// LoadUserModels is LoadModels of the user's files and those of -models (comma separated), with
// warnings to log.
func LoadUserModels(flagValue string, log io.Writer) error {
	return LoadModels(strings.Split(flagValue, ","), true, log)
}

// TableVersion is the version of the definition files this hwnpu reads.
const TableVersion = 1

// Builtin is what ModelFiles calls the embedded models.yaml.
const Builtin = "built-in models.yaml"

// The definitions read (Catalog, Categories, aliases), set by LoadModels; the built-in ones
// until it is called.
var (
	// ModelFiles are the definition files read, the built-in first.
	ModelFiles []string
	// Aliases run the model of the value wherever the key is picked.
	Aliases map[string]string
)

func init() {
	if err := LoadModels(nil, false, nil); err != nil {
		panic(err)
	}
}

// UserModelsFile is the user's own definition file: models.yaml in hwnpu's directory of
// settings, ~/.config/hwnpu ($XDG_CONFIG_HOME) and %LOCALAPPDATA%\hwnpu on Windows, which is
// read when it is there.
func UserModelsFile() string {
	if runtime.GOOS == "windows" {
		if d := os.Getenv("LOCALAPPDATA"); d != "" {
			return filepath.Join(d, "hwnpu", "models.yaml")
		}
	}
	if d := os.Getenv("XDG_CONFIG_HOME"); d != "" {
		return filepath.Join(d, "hwnpu", "models.yaml")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".config", "hwnpu", "models.yaml")
}

// LoadModels reads the definitions: the built-in models.yaml, then with user set the user's
// file (UserModelsFile, when it is there), the files of ModelsEnv and files (-models), each over
// the ones before. Warnings (an override of a model none defines, a newer version's keys) go to
// log.
func LoadModels(files []string, user bool, log io.Writer) error {
	t := newTable(log)
	if err := t.add(Builtin, hwnpu.ModelsYAML, true); err != nil {
		return err
	}
	if user {
		var paths []string
		if p := UserModelsFile(); p != "" && exists(p) {
			paths = append(paths, p)
		}
		for _, p := range filepath.SplitList(os.Getenv(ModelsEnv)) {
			if p = strings.TrimSpace(p); p != "" {
				paths = append(paths, p)
			}
		}
		for _, p := range files {
			if p = strings.TrimSpace(p); p != "" {
				paths = append(paths, p)
			}
		}
		for _, p := range paths {
			raw, err := os.ReadFile(p)
			if err != nil {
				return fmt.Errorf("chat: definitions: %w", err)
			}
			if err := t.add(p, raw, false); err != nil {
				return err
			}
		}
	}
	catalog, categories, aliases, err := t.build()
	if err != nil {
		return err
	}
	Catalog, Categories, Aliases, ModelFiles = catalog, categories, aliases, t.files
	Sources, Search = t.sources, t.search
	digests = map[string]map[string]string{}
	for _, m := range catalog {
		for f, d := range m.SHA256 {
			if digests[m.Source()] == nil {
				digests[m.Source()] = map[string]string{}
			}
			digests[m.Source()][f] = d
		}
	}
	foundMu.Lock()
	clear(found)
	foundMu.Unlock()
	return nil
}

// modelEntry is a model in a definition file: the fields it gives, nil for those it leaves.
type modelEntry struct {
	Family   *string    `yaml:"family"`
	Tag      *string    `yaml:"tag"`
	Kind     *string    `yaml:"kind"`
	From     *string    `yaml:"from"`
	Arch     *string    `yaml:"arch"`
	Machine  *int       `yaml:"machine"`
	Memory   *PerVendor `yaml:"memory"`
	Bytes    *PerVendor `yaml:"bytes"`
	Features *[]string  `yaml:"features"`
	// SHA256 pins the repository's files by name (config.json, model.safetensors, ...): a file
	// fetched from any source must have it (docs/model-table.md 5.3).
	SHA256  *map[string]string `yaml:"sha256"`
	NoThink *bool              `yaml:"no_think"`
	// Device is where the model runs by the vendor of the machine's NPU (amd: cpu); a file
	// over another changes the vendors it names and keeps the others.
	Device *map[string]string `yaml:"device"`
	// AMDExport is how tools/amd/iron_export_lm.py exports the model for AMD's NPU: its model
	// and options (lfm3 --max-seq-len 12288 --q4), run when the export is not here.
	AMDExport *string `yaml:"amd_export"`
	// File is the model's checkpoint file in a repository that holds several models (Anima's
	// split_files/diffusion_models/anima-turbo-v1.1.safetensors: the file's name).
	File *string `yaml:"file"`
	// With are the other repositories the model reads, by role (detector:
	// hf:PaddlePaddle/PP-OCRv6_small_det_onnx@REV): each an hf:OWNER/NAME, pinned or not. Their
	// files' sha256 are under ROLE/FILE.
	With   *map[string]string `yaml:"with"`
	About  *string            `yaml:"about"`
	Like   string             `yaml:"like"`
	Remove bool               `yaml:"remove"`
}

type pickEntry struct {
	Machine int    `yaml:"machine"`
	Model   string `yaml:"model"`
	Remove  bool   `yaml:"remove"`
}

type categoryEntry struct {
	Aliases *[]string   `yaml:"aliases"`
	About   *string     `yaml:"about"`
	Picks   []pickEntry `yaml:"picks"`
	Remove  bool        `yaml:"remove"`
}

// tableFile is a definition file. NPUs, Builds and Cache are designed (docs/model-table.md) but
// not read yet: a file that has them gets a warning.
type tableFile struct {
	Version    int                       `yaml:"version"`
	Models     map[string]*modelEntry    `yaml:"models"`
	Categories map[string]*categoryEntry `yaml:"categories"`
	Aliases    map[string]string         `yaml:"aliases"`
	Sources    map[string]*sourceEntry   `yaml:"sources"`
	Search     *[]string                 `yaml:"search"`
	NPUs       any                       `yaml:"npus"`
	Builds     any                       `yaml:"builds"`
	Cache      any                       `yaml:"cache"`
}

// order is a file's models and categories in the order written (maps lose it).
type order struct {
	Models     yaml.MapSlice `yaml:"models"`
	Categories yaml.MapSlice `yaml:"categories"`
}

// category is a category being merged: its picks by machine.
type category struct {
	aliases []string
	about   string
	picks   map[int]pick
}

type pick struct {
	model string
	file  string // where it was written
	user  bool
}

// table merges definition files.
type table struct {
	log        io.Writer
	files      []string
	models     map[string]*modelEntry
	origin     map[string][]string // the files that wrote each model
	modelOrder []string
	cats       map[string]*category
	catOrder   []string
	aliases    map[string]string
	aliasFile  map[string]string
	removed    map[string]string // models a file removed, by the file
	sources    map[string]Source
	search     []string
	searchFile string
}

func newTable(log io.Writer) *table {
	return &table{log: log, models: map[string]*modelEntry{}, origin: map[string][]string{},
		cats: map[string]*category{}, aliases: map[string]string{}, aliasFile: map[string]string{}, removed: map[string]string{},
		sources: map[string]Source{SourceHF: {Name: SourceHF, Type: "hf"}}, search: []string{SourceHF}, searchFile: Builtin}
}

func (t *table) warn(file, format string, args ...any) {
	if t.log != nil {
		fmt.Fprintf(t.log, "hwnpu: %s: %s\n", file, fmt.Sprintf(format, args...))
	}
}

// add merges a file. The built-in one is read strictly: anything this hwnpu does not know is an
// error. A user's file is too, except that a file of a newer version may have keys this hwnpu
// does not know (warned and skipped), so that a file written for a newer hwnpu still starts an
// older one.
func (t *table) add(name string, raw []byte, builtin bool) error {
	var head struct {
		Version int `yaml:"version"`
	}
	if err := yaml.Unmarshal(raw, &head); err != nil {
		return fmt.Errorf("chat: %s: %w", name, err)
	}
	opts := []yaml.DecodeOption{yaml.DisallowUnknownField()}
	switch {
	case head.Version == TableVersion:
	case head.Version > TableVersion && !builtin:
		t.warn(name, "version %d is newer than this hwnpu's (%d): what it does not know is skipped", head.Version, TableVersion)
		opts = nil
	case head.Version == 0:
		return fmt.Errorf("chat: %s: no version (write version: %d at its top)", name, TableVersion)
	default:
		return fmt.Errorf("chat: %s: version %d is not one this hwnpu reads (%d)", name, head.Version, TableVersion)
	}
	var f tableFile
	if err := yaml.UnmarshalWithOptions(raw, &f, opts...); err != nil {
		return fmt.Errorf("chat: %s: %s", name, yaml.FormatError(err, false, true))
	}
	var o order
	if err := yaml.Unmarshal(raw, &o); err != nil {
		return fmt.Errorf("chat: %s: %w", name, err)
	}
	if builtin && (f.Sources != nil || f.Search != nil) {
		return fmt.Errorf("chat: %s: sources and search are the user's to set", name)
	}
	for key, e := range f.Sources {
		s, err := toSource(key, e)
		if err != nil {
			return fmt.Errorf("chat: %s: %w", name, err)
		}
		t.sources[key] = s
	}
	if f.Search != nil {
		if len(*f.Search) == 0 {
			return fmt.Errorf("chat: %s: search lists no source", name)
		}
		t.search, t.searchFile = slices.Clone(*f.Search), name
	}
	for key, v := range map[string]any{"npus": f.NPUs, "builds": f.Builds, "cache": f.Cache} {
		if v != nil {
			if builtin {
				return fmt.Errorf("chat: %s: %s is not read yet", name, key)
			}
			t.warn(name, "%s is not read by this version of hwnpu yet (docs/model-table.md); skipped", key)
		}
	}
	t.files = append(t.files, name)
	for _, item := range o.Models {
		key := fmt.Sprint(item.Key)
		e := f.Models[key]
		if e == nil {
			continue
		}
		if err := t.addModel(name, key, e, builtin); err != nil {
			return err
		}
	}
	for _, item := range o.Categories {
		key := fmt.Sprint(item.Key)
		e := f.Categories[key]
		if e == nil {
			continue
		}
		if err := t.addCategory(name, key, e, builtin); err != nil {
			return err
		}
	}
	for from, to := range f.Aliases {
		t.aliases[from], t.aliasFile[from] = to, name
	}
	return nil
}

func (t *table) addModel(file, name string, e *modelEntry, builtin bool) error {
	old, ok := t.models[name]
	switch {
	case e.Remove:
		if !ok {
			t.warn(file, "remove: no model %s", name)
			return nil
		}
		delete(t.models, name)
		t.modelOrder = slices.DeleteFunc(t.modelOrder, func(n string) bool { return n == name })
		t.removed[name] = file
		return nil
	case !ok && e.From == nil && e.Like == "":
		// an override of a model none defines: the built-in file may have dropped it
		if builtin {
			return fmt.Errorf("chat: %s: model %s has no from", file, name)
		}
		t.warn(file, "no model %s to override (it has no from or like); skipped", name)
		return nil
	case !ok:
		t.models[name] = &modelEntry{}
		t.modelOrder = append(t.modelOrder, name)
		old = t.models[name]
	}
	merge(old, e)
	t.origin[name] = append(t.origin[name], file)
	return nil
}

// merge sets the fields of dst that src gives.
func merge(dst, src *modelEntry) {
	// memory and bytes: a table changes the vendors it names
	for _, f := range []struct{ d, s **PerVendor }{{&dst.Memory, &src.Memory}, {&dst.Bytes, &src.Bytes}} {
		if *f.s != nil && *f.d != nil {
			m := (**f.d).over(**f.s)
			defer func(d **PerVendor) { *d = &m }(f.d)
		}
	}
	if src.Device != nil && dst.Device != nil {
		m := maps.Clone(*dst.Device)
		maps.Copy(m, *src.Device)
		defer func() { dst.Device = &m }()
	}
	d, s := reflect.ValueOf(dst).Elem(), reflect.ValueOf(src).Elem()
	for i := range s.NumField() {
		f := s.Field(i)
		switch f.Kind() {
		case reflect.Pointer:
			if !f.IsNil() {
				d.Field(i).Set(f)
			}
		case reflect.String:
			if f.String() != "" {
				d.Field(i).Set(f)
			}
		}
	}
}

func (t *table) addCategory(file, name string, e *categoryEntry, builtin bool) error {
	c, ok := t.cats[name]
	if e.Remove {
		if ok {
			delete(t.cats, name)
			t.catOrder = slices.DeleteFunc(t.catOrder, func(n string) bool { return n == name })
		}
		return nil
	}
	if !ok {
		c = &category{picks: map[int]pick{}}
		t.cats[name] = c
		t.catOrder = append(t.catOrder, name)
	}
	if e.Aliases != nil {
		c.aliases = *e.Aliases
	}
	if e.About != nil {
		c.about = *e.About
	}
	for _, p := range e.Picks {
		switch {
		case p.Machine <= 0:
			return fmt.Errorf("chat: %s: category %s: a pick needs machine (GB of memory)", file, name)
		case p.Remove:
			delete(c.picks, p.Machine)
		case p.Model == "":
			return fmt.Errorf("chat: %s: category %s: the pick of %d GB has no model", file, name, p.Machine)
		default:
			c.picks[p.Machine] = pick{model: p.Model, file: file, user: !builtin}
		}
	}
	return nil
}

// kinds are the kinds a model may have, with the commands that run them.
var kinds = map[Kind]string{
	KindChat:       "hwnpu-chat (chat, agent, serve)",
	KindEmbed:      "hwnpu-chat (search, serve)",
	KindTranscribe: "hwnpu-audio transcribe",
	KindSpeech:     "hwnpu-audio tts",
	KindDecision:   "hwnpu-chat systemone",
	KindImage:      "hwnpu-image generate",
	KindDepth:      "hwnpu-image depth",
	KindFace:       "hwnpu-image face-detect",
	KindPitch:      "hwnpu-audio pitch",
	KindPose:       "hwnpu-image pose",
	KindDetect:     "hwnpu-image detect",
	KindOCR:        "hwnpu-image ocr",
	KindNotes:      "hwnpu-audio notes",
	KindMusic:      "hwnpu-audio generate",
	KindSFX:        "hwnpu-audio generate -m sfx",
}

// build makes the catalog, the categories and the aliases of what was merged.
func (t *table) build() ([]Model, []Category, map[string]string, error) {
	var errs []error
	for _, s := range t.search {
		if _, ok := t.sources[s]; !ok {
			errs = append(errs, fmt.Errorf("chat: %s: search: no source %q (sources defines them; hf is Hugging Face)", t.searchFile, s))
		}
	}
	var catalog []Model
	byName := map[string]bool{}
	for _, name := range t.modelOrder {
		e, err := t.resolveLike(name, nil)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		m, err := toModel(name, e, t.sources)
		if err != nil {
			errs = append(errs, fmt.Errorf("chat: %s: %w", strings.Join(t.origin[name], ", "), err))
			continue
		}
		m.Files = t.origin[name]
		catalog = append(catalog, m)
		byName[name] = true
	}
	var categories []Category
	for _, name := range t.catOrder {
		c := t.cats[name]
		cat := Category{Name: name, Aliases: c.aliases, About: c.about}
		for machine, p := range c.picks {
			if !byName[p.model] {
				if file, ok := t.removed[p.model]; ok {
					t.warn(file, "category %s: %s, the model of %d GB, is removed; the column is dropped", name, p.model, machine)
					continue
				}
				if p.user {
					t.warn(p.file, "category %s: no model %s for %d GB; skipped", name, p.model, machine)
					continue
				}
				errs = append(errs, fmt.Errorf("chat: %s: category %s: no model %s", p.file, name, p.model))
				continue
			}
			cat.Picks = append(cat.Picks, CategoryPick{Machine: machine, Model: p.model})
		}
		if len(cat.Picks) == 0 {
			continue
		}
		slices.SortFunc(cat.Picks, func(a, b CategoryPick) int { return a.Machine - b.Machine })
		categories = append(categories, cat)
	}
	aliases := map[string]string{}
	for from, to := range t.aliases {
		file := t.aliasFile[from]
		switch {
		case from == to:
		case !byName[from] || !byName[to]:
			if file == Builtin && t.removed[from] == "" && t.removed[to] == "" {
				errs = append(errs, fmt.Errorf("chat: %s: alias %s: %s: no such model", file, from, to))
			} else {
				t.warn(file, "alias %s: %s: no such model; skipped", from, to)
			}
		default:
			aliases[from] = to
		}
	}
	return catalog, categories, aliases, errors.Join(errs...)
}

// resolveLike is a model's entry with the fields of the model it is like that it leaves (all but
// the tag).
func (t *table) resolveLike(name string, seen []string) (*modelEntry, error) {
	e, ok := t.models[name]
	if !ok {
		return nil, fmt.Errorf("chat: %s: like %s: no such model", strings.Join(t.origin[seen[len(seen)-1]], ", "), name)
	}
	if e.Like == "" {
		return e, nil
	}
	if slices.Contains(seen, name) {
		return nil, fmt.Errorf("chat: models %s are each like the next", strings.Join(append(seen, name), ", "))
	}
	base, err := t.resolveLike(e.Like, append(seen, name))
	if err != nil {
		return nil, err
	}
	out := *base
	out.Tag, out.Like, out.SHA256 = nil, "", nil // the files of another repository are not the base's
	merge(&out, &modelEntry{Family: e.Family, Tag: e.Tag, Kind: e.Kind, From: e.From, Arch: e.Arch, Machine: e.Machine,
		Memory: e.Memory, Bytes: e.Bytes, Features: e.Features, NoThink: e.NoThink, About: e.About, SHA256: e.SHA256, Device: e.Device, AMDExport: e.AMDExport, File: e.File, With: e.With})
	return &out, nil
}

// toModel is the catalog's entry of a merged model.
func toModel(name string, e *modelEntry, sources map[string]Source) (Model, error) {
	m := Model{Name: name}
	get := func(p *string) string {
		if p == nil {
			return ""
		}
		return *p
	}
	m.Family, m.Tag, m.About, m.Arch, m.File = get(e.Family), get(e.Tag), get(e.About), get(e.Arch), get(e.File)
	m.AMDExport = strings.Fields(get(e.AMDExport))
	m.Kind = Kind(get(e.Kind))
	if _, ok := kinds[m.Kind]; !ok {
		var ks []string
		for k := range kinds {
			ks = append(ks, string(k))
		}
		slices.Sort(ks)
		return m, fmt.Errorf("model %s: kind %q is not one of %s", name, m.Kind, strings.Join(ks, ", "))
	}
	src := get(e.From)
	sn, repo, ok := strings.Cut(src, ":")
	switch s, known := sources[sn]; {
	case !ok || !known:
		return m, fmt.Errorf("model %s: from %q is not SOURCE:NAME (hf:OWNER/NAME, or a source of sources)", name, src)
	case s.Type == "hf":
		repo, m.Revision, _ = strings.Cut(repo, "@")
		if strings.Count(repo, "/") != 1 || !safeName(repo) || strings.HasSuffix(repo, "/") {
			return m, fmt.Errorf("model %s: from %q is not hf:OWNER/NAME", name, src)
		}
	default:
		if !safeName(repo) || strings.Contains(repo, "@") {
			return m, fmt.Errorf("model %s: from %q: NAME must be a folder inside the source", name, src)
		}
		m.SourceName = sn
	}
	m.Repo = repo
	if e.SHA256 != nil {
		for f, d := range *e.SHA256 {
			if len(d) != 64 || strings.Trim(strings.ToLower(d), "0123456789abcdef") != "" {
				return m, fmt.Errorf("model %s: sha256 of %s is not 64 hex digits", name, f)
			}
		}
		m.SHA256 = *e.SHA256
	}
	if e.Machine != nil {
		m.Machine = *e.Machine
	}
	if e.Memory != nil {
		m.MemoryBy, m.Memory = maps.Clone(*e.Memory), e.Memory.For("")
	}
	if e.Bytes != nil {
		m.BytesBy, m.Bytes = maps.Clone(*e.Bytes), e.Bytes.For("")
	}
	if e.NoThink != nil {
		m.NoThink = *e.NoThink
	}
	if e.Features != nil {
		for _, f := range *e.Features {
			if !slices.ContainsFunc(Features, func(k struct{ Name, About string }) bool { return k.Name == f }) {
				return m, fmt.Errorf("model %s: feature %q is not one of hwnpu-chat list's", name, f)
			}
		}
		m.Features = slices.Clone(*e.Features)
	}
	if e.Device != nil {
		for v, d := range *e.Device {
			if !slices.Contains(Vendors, v) || !slices.Contains(Devices, d) {
				return m, fmt.Errorf("model %s: device %s: %s: the vendors are %s, the devices %s", name, v, d,
					strings.Join(Vendors, ", "), strings.Join(Devices, ", "))
			}
		}
		m.Device = maps.Clone(*e.Device)
	}
	if e.With != nil {
		m.With = map[string]string{}
		for role, src := range *e.With {
			repo, ok := strings.CutPrefix(src, "hf:")
			r, _, _ := strings.Cut(repo, "@")
			if !ok || role == "" || !safeName(role) || strings.Contains(role, "/") || strings.Count(r, "/") != 1 || !safeName(r) || strings.HasSuffix(r, "/") {
				return m, fmt.Errorf("model %s: with %s: %q is not hf:OWNER/NAME", name, role, src)
			}
			m.With[role] = repo
		}
	}
	if m.Tag != "" && m.Family == "" {
		return m, fmt.Errorf("model %s: a tag needs a family", name)
	}
	return m, nil
}
