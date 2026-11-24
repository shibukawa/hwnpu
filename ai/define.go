package ai

import (
	"cmp"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"maps"
	"slices"
	"strings"

	"github.com/goccy/go-yaml"

	"github.com/shibukawa/bonsaiai/internal/chat"
	devenv "github.com/shibukawa/bonsaiai/internal/dev"
	"github.com/shibukawa/bonsaiai/modelstore"
)

// Definition is a model an app makes as it runs, as a definition file would (models.yaml): most
// often another model of the definitions (Like) with settings of its own, a weight format, the
// CPU, or programs converted beforehand. It is a model of its own name, prepared and kept apart
// from the one it is like, so that a model always runs as its definition says. A command makes
// one from its flags (--format, --cpu, the files of programs converted by hand) and runs it by
// its name:
//
//	d := ai.Definition{Like: "whisper-large-v3-turbo", Settings: map[string]string{"format": "lut4"}}
//	app := ai.NewApp("transcriber").Define(d)
//	stt := speech.Transcribe(d.ID())
type Definition struct {
	// Name is the model's name; "" makes one from Like and the rest (ID).
	Name string
	// Like is the model of the definitions whose fields this one takes (models.yaml's like): its
	// name, not a family or a use (Catalog.Find gives the name a family or a use picks).
	Like string
	// Kind is what the model is (chat, transcribe, speech, ...), and From where it is downloaded
	// from: hf:OWNER/NAME or hf:OWNER/NAME@REVISION. Both are Like's when "".
	Kind, From string
	// Arch is the structure (bonsaiai-dev convert -archs), "" for Like's or config.json's.
	Arch string
	// About says what it is, for the lists.
	About string
	// Device is where it runs: npu, or cpu (bonsaiai's CPU interpreter or the CPU's runtime, as its
	// role runs it), on every NPU; "" for Like's.
	Device string
	// Format, Context, Paged, Prefill and Tier are how a chat model runs (models.yaml's format,
	// context, paged, prefill, tier); the zero value for Like's.
	Format         string
	Context, Paged int
	Prefill, Tier  string
	// MemoryGB is what it takes running, 0 for Like's.
	MemoryGB float64
	// Settings are how its role prepares and runs it beyond the fields above, by name (models.yaml's
	// settings): each role says the names it reads (the speech models' format, a chat model's
	// draft). Programs are files converted beforehand, by the name its role reads them as
	// (models.yaml's programs: absolute paths): the role uses them as they are.
	Settings map[string]string
	Programs map[string][]string
}

// ID is the model's name: Name, else Like with a hash of the rest (whisper-large-v3-turbo+3f2a9c1d),
// the same for the same definition, so that a command run again with the same flags finds the
// model it prepared before.
func (d Definition) ID() string {
	if d.Name != "" {
		return d.Name
	}
	raw, _ := d.yaml("")
	sum := sha256.Sum256(raw)
	return cmp.Or(d.Like, "model") + "+" + hex.EncodeToString(sum[:4])
}

// entry is the model as a definition file writes it.
type entry struct {
	Like     string              `yaml:"like,omitempty"`
	Family   *string             `yaml:"family,omitempty"`
	Kind     string              `yaml:"kind,omitempty"`
	From     string              `yaml:"from,omitempty"`
	Arch     string              `yaml:"arch,omitempty"`
	About    string              `yaml:"about,omitempty"`
	Device   map[string]string   `yaml:"device,omitempty"`
	Memory   float64             `yaml:"memory,omitempty"`
	Format   string              `yaml:"format,omitempty"`
	Context  int                 `yaml:"context,omitempty"`
	Paged    int                 `yaml:"paged,omitempty"`
	Prefill  string              `yaml:"prefill,omitempty"`
	Tier     string              `yaml:"tier,omitempty"`
	Settings map[string]string   `yaml:"settings,omitempty"`
	Programs map[string][]string `yaml:"programs,omitempty"`
}

// yaml is the definition as a definition file of its own, the model named name.
func (d Definition) yaml(name string) ([]byte, error) {
	e := entry{Like: d.Like, Kind: d.Kind, From: d.From, Arch: d.Arch, About: d.About, Memory: d.MemoryGB,
		Format: d.Format, Context: d.Context, Paged: d.Paged, Prefill: d.Prefill, Tier: d.Tier,
		Settings: maps.Clone(d.Settings), Programs: maps.Clone(d.Programs)}
	if d.Like != "" {
		// not of Like's family: a family, or a use, never picks a model an app made
		none := ""
		e.Family = &none
	}
	if d.Device != "" {
		e.Device = map[string]string{}
		for _, v := range chat.Vendors {
			e.Device[v] = d.Device
		}
	}
	file := struct {
		Version int               `yaml:"version"`
		Models  map[string]*entry `yaml:"models"`
	}{chat.TableVersion, map[string]*entry{name: &e}}
	return yaml.Marshal(file)
}

// Define adds models the app makes as it runs (Definition) to the definitions its Store reads:
// the app's uses may then name them by their ID.
func (a *App) Define(defs ...Definition) *App {
	a.Store.Definitions = append(a.Store.Definitions, defs...)
	return a
}

// definitions are the Store's Definitions as the texts modelstore reads.
func definitions(defs []Definition) ([]modelstore.Definitions, error) {
	var out []modelstore.Definitions
	seen := map[string]bool{}
	for _, d := range defs {
		id := d.ID()
		if strings.ContainsAny(id, ":/\\ ") {
			return nil, fmt.Errorf("ai: a defined model's name %q has a colon, a slash or a space", id)
		}
		if d.Like == "" && (d.Kind == "" || d.From == "") {
			return nil, fmt.Errorf("ai: the defined model %s needs Like, or Kind and From", id)
		}
		if seen[id] {
			continue
		}
		seen[id] = true
		raw, err := d.yaml(id)
		if err != nil {
			return nil, fmt.Errorf("ai: %s: %w", id, err)
		}
		out = append(out, modelstore.Definitions{Name: "the app's definition of " + id, YAML: raw})
	}
	return out, nil
}

// spec is what makes two preparations of a model differ beyond its name and its run (record.Run):
// its device, settings and programs.
func spec(m chat.Model) string {
	var parts []string
	for _, v := range slices.Sorted(maps.Keys(m.Device)) {
		parts = append(parts, "device."+v+"="+m.Device[v])
	}
	for _, k := range slices.Sorted(maps.Keys(m.Settings)) {
		parts = append(parts, "settings."+k+"="+m.Settings[k])
	}
	for _, k := range slices.Sorted(maps.Keys(m.Programs)) {
		parts = append(parts, "programs."+k+"="+strings.Join(m.Programs[k], ","))
	}
	// the developers' rewrites over the recipe's (internal/recipe.RewritesFor): what is prepared
	// with them is not what is prepared without
	if v := devenv.Value("rewrite"); v != "" {
		parts = append(parts, "dev.rewrite="+v)
	}
	return strings.Join(parts, ";")
}

// base is the model of the definitions m was made from by the app (Definition.Like, through the
// app's definitions made from others), m itself for one it did not make.
func (s Store) base(m chat.Model) chat.Model {
	for range len(s.Definitions) + 1 {
		i := slices.IndexFunc(s.Definitions, func(d Definition) bool { return d.ID() == m.Name })
		if i < 0 || s.Definitions[i].Like == "" {
			return m
		}
		b, ok := chat.Lookup(s.Definitions[i].Like)
		if !ok {
			return m
		}
		m = b
	}
	return m
}
