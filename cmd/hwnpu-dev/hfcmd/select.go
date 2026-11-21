package hfcmd

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path"
	"path/filepath"
	"slices"
	"sort"
	"strings"

	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/sourcescmd"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/speech"
	shared "github.com/shibukawa/hwnpu/internal/store"
	"github.com/shibukawa/hwnpu/internal/transcribe"
	"github.com/shibukawa/hwnpu/internal/vision/ironcnn"
)

// The groups of a model's conversions on Hugging Face, each a row of hwnpu-dev models hf plan and a
// target of models.yaml's hub: the exports for AMD's NPU, and the Macs' (the Core AI exports
// with their prebuilt programs and the tables they read).
const (
	groupAMD   = "amd-xdna2"
	groupApple = "apple"
)

// found is an entry of the sources that a model's conversions have now.
type found struct {
	want  shared.Entry // as the commands look for it now
	got   shared.Entry // as it goes to Hugging Face: at want's place (released from v0-HASH: with want's version)
	raw   []byte       // entry.json as the layer holds it; nil when it is written anew (released)
	layer int          // its layer (layers())
	from  string       // its folder in the layer: want.Path, or chat.Unversioned(want)
	// tools is the tools' hash the entry was made with when it is not this hwnpu's (an entry
	// with no version of the same recipe but for the tools, taken for a published version as
	// the commands take one at vN): what the tools changed since may change the export.
	tools string
}

// rel is the entry's folder in its model's repository: TARGET/EXPORT/VERSION.
func (f found) rel() string { return hubRel(f.want.Path) }

// hubRel is the folder in a model's repository of an entry of the sources (converted/MODEL/REST).
func hubRel(p string) string {
	parts := strings.SplitN(p, "/", 3)
	if len(parts) < 3 {
		return p
	}
	return parts[2]
}

// row is a model's group of conversions as the sources hold them now.
type row struct {
	model    chat.Model
	group    string
	entries  []found
	prebuilt []found  // the prebuilt programs of the Core AI exports (their programs checked when pushed)
	why      []string // what keeps it from going to Hugging Face, the first thing to do first
	chips    []string // apple: the architectures that have prebuilt programs
}

// layers are the layers hwnpu-dev models hf takes conversions from: this machine's cache and the
// sources (not the bundled models' nor Hugging Face's own).
func layers() []shared.Source {
	out := []shared.Source{shared.Local()}
	for _, s := range shared.Sources() {
		if !shared.IsHub(s) && s.From != "bundle" {
			out = append(out, s)
		}
	}
	return out
}

// lookup finds want in the layers (only, when it is >= 0): at its place, else (want is of a
// published version) at its place with no version with the same key, which is the same bytes,
// else in another folder with no version of its export (held: the folders the layers hold of
// the model) of the same recipe but for the tools' hash, the newest.
func lookup(ctx context.Context, ls []shared.Source, only int, want shared.Entry, held map[string]int, loose bool) (found, bool) {
	released := want.Version > 0 && !want.Develop
	for i, s := range ls {
		if only >= 0 && i != only {
			continue
		}
		if raw, e, err := readEntry(ctx, s, want.Path); err == nil && chat.Takes(want, e) {
			return found{want: want, got: e, raw: raw, layer: i, from: want.Path}, true
		}
		if released {
			from := chat.Unversioned(want)
			if _, e, err := readEntry(ctx, s, from); err == nil && e.Key == want.Key {
				return found{want: want, got: chat.Released(want, e), layer: i, from: from}, true
			}
		}
	}
	if !released && !loose {
		return found{}, false
	}
	var best found
	ok := false
	parent := path.Dir(want.Path) + "/"
	for dir, i := range held {
		if !strings.HasPrefix(dir, parent) || strings.Contains(strings.TrimPrefix(dir, parent), "/") || !strings.HasPrefix(path.Base(dir), "v0-") {
			continue
		}
		_, e, err := readEntry(ctx, ls[i], dir)
		if err != nil || e.Kind != want.Kind || !butTools(e.Recipe, want.Recipe) {
			continue
		}
		if !ok || e.Made.After(best.got.Made) {
			// it keeps its own key and recipe (of its tools): a reader of vN takes it all the same
			best = found{want: want, got: chat.Released(want, e), layer: i, from: dir, tools: e.Recipe["tools"]}
			ok = true
		}
	}
	return best, ok
}

// butTools reports whether two recipes are the same but for the tools' hash.
func butTools(a, b map[string]string) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		if w, ok := b[k]; k != "tools" && (!ok || w != v) {
			return false
		}
	}
	return true
}

// readEntry reads the entry.json at rel in s, as it is and parsed.
func readEntry(ctx context.Context, s shared.Source, rel string) ([]byte, shared.Entry, error) {
	raw, err := s.Store.Read(ctx, rel+"/"+shared.EntryFile)
	if err != nil {
		return nil, shared.Entry{}, err
	}
	var e shared.Entry
	if err := json.Unmarshal(raw, &e); err != nil {
		return nil, e, err
	}
	if e.Format != shared.EntryFormat {
		return nil, e, fmt.Errorf("%s: format %d", rel, e.Format)
	}
	return raw, e, nil
}

// amdWants are the entries of model m's exports for AMD's NPU the commands look for now, or why
// none is known.
func amdWants(ctx context.Context, m chat.Model) ([]shared.Entry, error) {
	var out []shared.Entry
	pinned := func(r map[string]string, tool, name string) error {
		if r == nil {
			return errors.New("pin its files (sha256 in models.yaml)")
		}
		out = append(out, chat.PlaceEntry(chat.ExportKind(tool), m, cache.AMDTarget, name, r))
		return nil
	}
	switch {
	case len(m.AMDExport) > 0:
		for _, f := range sourcescmd.AMDForms(m) {
			r := chat.ExportRecipe(ctx, f, "iron_export_lm.py", f.AMDExport)
			if r == nil {
				return nil, errors.New("the commit of its files is not known")
			}
			out = append(out, chat.PlaceEntry("iron-lm", f, cache.AMDTarget, filepath.Base(chat.AMDExportDir(f)), r))
		}
		return out, nil
	case m.Kind == chat.KindTranscribe:
		return out, pinned(transcribe.AMDRecipe(m), transcribe.AMDTool, transcribe.AMDExportName(m.Name))
	case m.Kind == chat.KindSpeech:
		return out, pinned(speech.AMDRecipe(m), speech.AMDTool, speech.AMDExportName(m.Name))
	}
	if tool, version, ok := sourcescmd.FirstRunExport(m); ok {
		return out, pinned(chat.PinnedRecipe(m, tool, version), tool, filepath.Base(ironcnn.Dir(m.Name)))
	}
	return nil, errors.New("hwnpu-dev models hf does not know its export for AMD's NPU yet")
}

// listed are the entries the layers hold of model m, by target: each entry's folder
// (converted/MODEL/TARGET/.../VERSION) and the layer of the first that holds it.
func listed(ctx context.Context, ls []shared.Source, only int, m chat.Model) map[string]map[string]int {
	out := map[string]map[string]int{}
	prefix := "converted/" + m.Name + "/"
	for i, s := range ls {
		if only >= 0 && i != only {
			continue
		}
		keys, err := s.Store.List(ctx, prefix)
		if err != nil {
			continue
		}
		for _, k := range keys {
			dir, ok := strings.CutSuffix(k, "/"+shared.EntryFile)
			if !ok {
				continue
			}
			target, _, _ := strings.Cut(strings.TrimPrefix(dir, prefix), "/")
			if out[target] == nil {
				out[target] = map[string]int{}
			}
			if _, ok := out[target][dir]; !ok {
				out[target][dir] = i
			}
		}
	}
	return out
}

// exportOf is the export's name of an entry's folder converted/MODEL/TARGET/EXPORT.../VERSION.
func exportOf(dir string) string {
	parts := strings.Split(dir, "/")
	if len(parts) < 5 {
		return ""
	}
	return strings.Join(parts[3:len(parts)-1], "/")
}

// rows are model m's groups as the layers (only that one, when >= 0) hold them.
// loose takes, for a version not written yet too, an entry of the same recipe but for the tools
// (hwnpu-dev models hf sizes: its size).
func rows(ctx context.Context, ls []shared.Source, only int, m chat.Model, loose bool) []*row {
	var common []string
	if !chat.HubCommitPinned(m) {
		common = append(common, fmt.Sprintf("pin from (hf:%s@COMMIT)", m.Repo))
	}
	if m.Develop {
		common = append(common, "drop develop: true")
	}
	amd := &row{model: m, group: groupAMD, why: slices.Clone(common)}
	apple := &row{model: m, group: groupApple, why: slices.Clone(common)}
	if len(common) > 0 && !loose {
		return []*row{amd, apple} // the keys need the commit: no lookups
	}
	held := listed(ctx, ls, only, m)

	// AMD: the exports the commands look for
	if m.Targets[cache.AMDTarget] == 0 {
		amd.why = append(amd.why, "write targets.amd-xdna2.version")
	}
	wants, err := amdWants(ctx, m)
	switch {
	case err != nil:
		amd.why = append(amd.why, err.Error())
	default:
		for _, w := range wants {
			if f, ok := lookup(ctx, ls, only, w, held[cache.AMDTarget], loose); ok {
				amd.entries = append(amd.entries, f)
			} else {
				amd.why = append(amd.why, fmt.Sprintf("make %s (hwnpu-dev models package %s, or its command on AMD's NPU)", path.Base(path.Dir(w.Path)), m.Name))
			}
		}
	}

	// the Macs: the Core AI exports the sources hold that are what this hwnpu makes now
	if m.Targets[chat.CoreAITarget] == 0 {
		apple.why = append(apple.why, "write targets.coreai.version")
	}
	exports := map[string]bool{}
	for dir := range held[chat.CoreAITarget] {
		if e := exportOf(dir); e != "" {
			exports[e] = true
		}
	}
	for _, e := range slices.Sorted(func(yield func(string) bool) {
		for k := range exports {
			if !yield(k) {
				return
			}
		}
	}) {
		want, ok := chat.CoreAIWant(ctx, m, e)
		if !ok {
			continue
		}
		// only at its version: the sources keep Core AI exports of names this hwnpu makes no
		// more under keys it still makes (Anima's of each size beside the one of every size), so
		// what a Mac took or made as the version is what goes (docs/prebuilt.md 4.1)
		if want.Version == 0 || want.Develop {
			continue
		}
		if f, ok := lookup(ctx, ls, only, want, nil, false); ok && f.from == want.Path {
			apple.entries = append(apple.entries, f)
		}
	}
	if len(apple.entries) == 0 && m.Targets[chat.CoreAITarget] > 0 {
		apple.why = append(apple.why, fmt.Sprintf("hwnpu-dev models package %s/coreai (on any machine): its Core AI exports go in the sources as v%d", m.Name, m.Targets[chat.CoreAITarget]))
	}
	// their prebuilt programs: checked against the export's programs when pushed
	names := map[string]bool{}
	for _, f := range apple.entries {
		names[exportOf(f.want.Path)] = true
	}
	chips := map[string]bool{}
	for target, dirs := range held {
		if !strings.HasPrefix(target, chat.CoreAITarget+"-") {
			continue
		}
		for _, dir := range slices.Sorted(mapKeys(dirs)) {
			raw, e, err := readEntry(ctx, ls[dirs[dir]], dir)
			if err != nil {
				continue
			}
			export, arch, ok := chat.IsPrebuilt(e)
			if !ok || !names[export] {
				continue
			}
			want := chat.PrebuiltWant(m, export, arch, e.Recipe["programs"])
			if want.Path != dir || want.Key != e.Key {
				continue
			}
			apple.prebuilt = append(apple.prebuilt, found{want: want, got: e, raw: raw, layer: dirs[dir], from: dir})
			chips[arch] = true
		}
	}
	apple.chips = slices.Sorted(mapKeys(chips))

	// the tables: the Core AI exports' (an export for AMD's NPU holds its own)
	var tables []found
	seen := map[string]bool{}
	for _, dir := range slices.Sorted(mapKeys(held["any"])) {
		_, e, err := readEntry(ctx, ls[held["any"][dir]], dir)
		if err != nil {
			continue
		}
		want, ok := chat.TablesWant(ctx, m, e)
		if !ok || seen[want.Path] {
			continue
		}
		if f, ok := lookup(ctx, ls, only, want, held["any"], loose); ok {
			seen[want.Path] = true
			tables = append(tables, f)
		}
	}
	if len(apple.entries) > 0 {
		apple.entries = append(apple.entries, tables...)
	}
	return []*row{amd, apple}
}

func mapKeys[V any](m map[string]V) func(func(string) bool) {
	return func(yield func(string) bool) {
		for k := range m {
			if !yield(k) {
				return
			}
		}
	}
}

// layerLabel is the layers a row's entries come from: 1, 0+1, or - for none.
func (r *row) layerLabel() string {
	set := map[int]bool{}
	for _, f := range r.entries {
		set[f.layer] = true
	}
	if len(set) == 0 {
		return "-"
	}
	var ns []int
	for n := range set {
		ns = append(ns, n)
	}
	sort.Ints(ns)
	var s []string
	for _, n := range ns {
		s = append(s, fmt.Sprint(n))
	}
	return strings.Join(s, "+")
}

// bytes is the size of what goes to Hugging Face of the row (packed as the layers hold it).
func (r *row) bytes() int64 {
	var n int64
	for _, f := range slices.Concat(r.entries, r.prebuilt) {
		if f.got.Pack != nil {
			n += f.got.Pack.Size
		} else {
			n += f.got.Bytes()
		}
	}
	return n
}
