// Package sourcescmd is hwnpu-dev sources: the upkeep of the sources (docs/cache-sharing-design.md
// 7): the cache's root and the sources in effect, what they hold, putting folders made by hand in
// them, moving a repository's lock to a new commit, checking an entry and removing entries. Also
// the parts of hwnpu-dev models package and push that are IRON's (AMDExport, PushCmd).
package sourcescmd

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"path"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/alecthomas/kong"

	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/devopt"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/envfile"
	"github.com/shibukawa/hwnpu/internal/finish"
	"github.com/shibukawa/hwnpu/internal/pose"
	rcp "github.com/shibukawa/hwnpu/internal/recipe"
	shared "github.com/shibukawa/hwnpu/internal/store"
	"github.com/shibukawa/hwnpu/internal/vision/detect"
	"github.com/shibukawa/hwnpu/internal/vision/ironcnn"
	"github.com/shibukawa/hwnpu/internal/vision/segment"
)

// Cmd is hwnpu-dev sources.
type Cmd struct {
	devopt.Defs `embed:""`

	List       ListCmd       `cmd:"" default:"1" help:"Show the cache's root, the sources in effect (HWNPU_MODEL_SOURCE, _1, ...), whether each is written back to, whether Hugging Face is read, and the .env files read (hwnpu-dev sources alone does it)."`
	Ls         LsCmd         `cmd:"" help:"List the conversions the sources hold (or one source: its number in hwnpu-dev sources list); with --original the repositories' locks."`
	Import     ImportCmd     `cmd:"" help:"Put folders made by hand (~/models) in the sources written back to: a checkpoint's files that are its Hugging Face repository's (original/), an export for AMD's NPU (converted/)."`
	Tokenizers TokenizersCmd `cmd:"" help:"Put the tokenizers of the models that run from their checkpoint (Apple's Neural Engine, Intel's NPU, the CPU) in the sources written back to, as their binary files (chat.PrepareTokenizer): a first run fetches them in place of the tokenizer's text."`
	Update     UpdateCmd     `cmd:"" help:"Move the repositories' files in the sources written back to to the commit Hugging Face serves now (their hwnpu-lock.json): the files that changed are downloaded again."`
	Verify     VerifyCmd     `cmd:"" help:"Check directories against their entry.json: the sizes, and the SHA-256 with --sums."`
	GC         GCCmd         `cmd:"" name:"gc" help:"Remove conversions from a source (by model, kind, key or age); shows what would go unless --apply. What a writeback cut off left: hwnpu cache clean."`
}

// AfterApply gives the subcommands the definitions' flag.
func (c *Cmd) AfterApply(k *kong.Context) error { c.Bind(k); return nil }

func load(d *devopt.Defs) error { return d.Load() }

func signalContext() (context.Context, context.CancelFunc) {
	return signal.NotifyContext(context.Background(), os.Interrupt)
}

// ListCmd is hwnpu-dev sources list.
type ListCmd struct{}

func (ListCmd) Run(d *devopt.Defs) error {
	if err := load(d); err != nil {
		return err
	}
	fmt.Printf("root: %s", cache.Root())
	if os.Getenv(cache.RootEnv) != "" {
		fmt.Printf(" (%s)", cache.RootEnv)
	}
	fmt.Printf("\nAMD exports: %s (and read from %s)\nIRON's kernels: %s\n", chat.AMDRoot(), cache.LegacyExports(), chat.IRONCache())
	for _, f := range envfile.Loaded {
		fmt.Printf(".env: %s", f.Path)
		if len(f.Set) > 0 {
			fmt.Printf(" (%s)", strings.Join(f.Set, ", "))
		}
		if f.Err != nil {
			fmt.Printf(": %v", f.Err)
		}
		fmt.Println()
	}
	ss := shared.Sources()
	w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "#\tsource\twriteback")
	for i, s := range layers() {
		wb := fmt.Sprint(s.Writeback)
		switch {
		case i == 0:
			wb = "this machine's"
		case s.From == "bundle":
			wb = "bundled, read only"
		}
		fmt.Fprintf(w, "%d\t%s\t%s\n", i, s, wb)
	}
	if shared.HuggingFace() {
		fmt.Fprintf(w, "-\tHugging Face\t-\n")
	}
	w.Flush()
	if len(ss) == 0 {
		fmt.Printf("(no sources: %s, %s_1, ... set them)\n", shared.SourceEnv, shared.SourceEnv)
	}
	if !shared.HuggingFace() {
		fmt.Printf("(%s=false: Hugging Face is not read)\n", shared.HuggingFaceEnv)
	}
	return nil
}

// layers are the layers of hwnpu-dev sources list, by their numbers: 0 this machine's cache,
// then the sources of the environment and the bundled models.
func layers() []shared.Source { return append([]shared.Source{shared.Local()}, shared.Sources()...) }

// source is the layer of number n of hwnpu-dev sources list (0: this machine's cache).
func source(n int) (shared.Source, error) {
	ls := layers()
	if n < 0 || n >= len(ls) {
		return shared.Source{}, fmt.Errorf("no source %d: hwnpu-dev sources list shows 0 to %d", n, len(ls)-1)
	}
	return ls[n], nil
}

// LsCmd is hwnpu-dev sources ls.
type LsCmd struct {
	Source   int  `arg:"" optional:"" default:"-1" help:"The source's number in hwnpu-dev sources list (0: this machine's cache; all when left out)."`
	Original bool `help:"List the repositories' locks (original/) instead."`
}

func (l LsCmd) Run(d *devopt.Defs) error {
	if err := load(d); err != nil {
		return err
	}
	ctx, stop := signalContext()
	defer stop()
	w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	if l.Original {
		fmt.Fprintln(w, "source\trepository\tcommit\tfiles\tGB\tupdated")
	} else {
		fmt.Fprintln(w, "source\tkind\tGB\tmade\tpath")
	}
	for i, s := range layers() {
		if l.Source >= 0 && l.Source != i {
			continue
		}
		if l.Original {
			repos, err := chat.LockedRepos(ctx, s)
			if err != nil {
				fmt.Fprintf(os.Stderr, "%s: %v\n", s, err)
				continue
			}
			for _, r := range repos {
				lk, err := shared.ReadLock(ctx, s, r)
				if err != nil {
					fmt.Fprintf(w, "%d\t%s\t?\t?\t?\t%v\n", i, r, err)
					continue
				}
				var n int64
				for _, f := range lk.Files {
					n += f.Size
				}
				fmt.Fprintf(w, "%d\t%s\t%.12s\t%d\t%.2f\t%s\n", i, r, lk.Commit, len(lk.Files), float64(n)/1e9, lk.Updated.Local().Format(time.DateTime))
			}
			continue
		}
		entries, err := shared.ListEntries(ctx, s)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", s, err)
			continue
		}
		for _, rel := range entries {
			e, err := shared.ReadEntryAt(ctx, s, rel)
			if err != nil {
				fmt.Fprintf(w, "%d\t?\t?\t?\t%s\n", i, rel)
				continue
			}
			fmt.Fprintf(w, "%d\t%s\t%.2f\t%s\t%s\n", i, e.Kind, float64(e.Bytes())/1e9, e.Made.Local().Format(time.DateTime), strings.TrimPrefix(rel, "converted/"))
		}
	}
	return w.Flush()
}

// AMDForms is m and each of its variants (machines of more memory) whose export for AMD's NPU differs (its recipe's args):
// gpt-oss-20b's 16 experts a layer on 16 GB and all of them from 24 GB are two exports.
func AMDForms(m chat.Model) []chat.Model {
	forms := []chat.Model{m}
	gbs := make([]int, 0, len(m.Variants))
	for gb := range m.Variants {
		gbs = append(gbs, gb)
	}
	sort.Ints(gbs)
	for _, gb := range gbs {
		v := m.On(gb)
		if len(v.AMDExport) == 0 || slices.ContainsFunc(forms, func(f chat.Model) bool {
			return slices.Equal(f.AMDExport, v.AMDExport)
		}) {
			continue
		}
		forms = append(forms, v)
	}
	return forms
}

// AMDFormsOn is AMDForms, or given memory the forms of machines of those GB alone (each
// recipe's args once).
func AMDFormsOn(m chat.Model, memory []int) []chat.Model {
	if len(memory) == 0 {
		return AMDForms(m)
	}
	var forms []chat.Model
	for _, gb := range memory {
		v := m.On(gb)
		if !slices.ContainsFunc(forms, func(f chat.Model) bool { return slices.Equal(f.AMDExport, v.AMDExport) }) {
			forms = append(forms, v)
		}
	}
	return forms
}

// AMDExport is hwnpu-dev models package of chat models for amd-xdna2: chat.PrepareAMD of each
// model without running it (the export compiles on the CPU: no NPU is opened).
type AMDExport struct {
	Model []string
}

// Make is hwnpu-dev models package of x's models, the forms of machines of memory GB alone when
// given (else every form, AMDForms).
func (x AMDExport) Make(ctx context.Context, memory []int) error {
	writes := shared.Writebacks()
	if len(writes) == 0 {
		return errors.New("no source is written back to: give one ?writeback=true (" + shared.SourceEnv + ")")
	}
	log := console.Log(os.Stderr)
	var made []string
	var forms []chat.Model
	for _, name := range x.Model {
		m, ok := chat.Lookup(name)
		if !ok || len(m.AMDExport) == 0 {
			return fmt.Errorf("%s is not a model of the definitions with a recipe of iron_export_lm.py (model-recipes.yaml)", name)
		}
		forms = append(forms, AMDFormsOn(m, memory)...)
	}
	for _, m := range forms {
		start := time.Now()
		dir, err := chat.PrepareAMD(ctx, m, log)
		if err != nil {
			return fmt.Errorf("%s: %w", m.Name, err)
		}
		shared.Flush(ctx)
		if ctx.Err() != nil {
			return ctx.Err()
		}
		e, err := shared.ReadEntry(dir)
		if err != nil {
			return fmt.Errorf("%s: %w", m.Name, err)
		}
		line := fmt.Sprintf("%s\t%s\t%.2f", m.Name, filepath.Base(dir), float64(e.Bytes())/1e9)
		for _, s := range writes {
			if t, err := shared.ReadEntryAt(ctx, s, e.Path); err == nil && t.Key == e.Key && t.Pack != nil {
				line += fmt.Sprintf("\t%.2f", float64(t.Pack.Size)/1e9)
			} else {
				line += "\t-"
			}
		}
		line += "\t" + time.Since(start).Round(time.Second).String() + "\t" + e.Path
		fmt.Fprintf(os.Stderr, "%s: %s\n", m.Name, line)
		made = append(made, line)
	}
	w := tabwriter.NewWriter(os.Stdout, 0, 2, 2, ' ', 0)
	fmt.Fprintln(w, "model\texport\tGB\tpacked GB\ttook\tpath")
	for _, l := range made {
		fmt.Fprintln(w, l)
	}
	return w.Flush()
}

// PushCmd is hwnpu-dev models push.
type PushCmd struct {
	Model []string `arg:"" help:"Models of the definitions (their exports for AMD's NPU: made as model-recipes.yaml says, or on the first run of hwnpu-image detect, pose, segment) to put in the layers."`
}

func (p PushCmd) Run(d *devopt.Defs) error {
	if err := load(d); err != nil {
		return err
	}
	ctx, stop := signalContext()
	defer stop()
	writes := shared.Writebacks()
	if len(writes) == 0 {
		return errors.New("no source is written back to: give one ?writeback=true (" + shared.SourceEnv + ")")
	}
	for _, name := range p.Model {
		m, ok := chat.Lookup(name)
		if !ok {
			return fmt.Errorf("%s is not a model of the definitions", name)
		}
		if tool, version, ok := FirstRunExport(m); ok {
			if err := pushFirstRun(ctx, writes, m, tool, version); err != nil {
				return err
			}
			continue
		}
		dir := chat.AMDExportDir(m)
		if dir == "" {
			return fmt.Errorf("%s has no recipe of iron_export_lm.py (model-recipes.yaml)", m.Name)
		}
		if _, err := os.Stat(filepath.Join(dir, "manifest.json")); err != nil {
			// one made before Root held the exports: under ~/models
			if d := chat.FindAMDExport(m); filepath.Base(d) == filepath.Base(dir) {
				dir = d
			} else {
				return fmt.Errorf("%s: no export %s here", m.Name, filepath.Base(dir))
			}
		}
		recipe := chat.ExportRecipe(ctx, m, "iron_export_lm.py", m.AMDExport)
		if recipe == nil {
			return fmt.Errorf("%s: the commit of its files is not known (pin it, or reach Hugging Face)", m.Name)
		}
		// an export made by hand (or before entries) gets its entry.json: what it would be made
		// from now. Made with other tools than these, it is not this recipe's: --force is not
		// offered, the export is to be made again with this hwnpu
		if err := describeAndPut(ctx, writes, m, chat.PlaceEntry("iron-lm", m, cache.AMDTarget, filepath.Base(dir), recipe), dir); err != nil {
			return err
		}
	}
	return nil
}

// describeAndPut gives dir the entry.json of want (described again unless it has it) and puts it
// in the sources writes.
func describeAndPut(ctx context.Context, writes []shared.Source, m chat.Model, want shared.Entry, dir string) error {
	e, err := shared.ReadEntry(dir)
	if err != nil || e.Key != want.Key || e.Path != want.Path {
		fmt.Fprintf(os.Stderr, "%s: describing %s (the SHA-256 of its files)\n", m.Name, dir)
		want.MadeBy = map[string]string{"note": "described by hwnpu-dev models push"}
		if e, err = shared.Describe(dir, want); err != nil {
			return err
		}
	}
	for _, s := range writes {
		fmt.Fprintf(os.Stderr, "%s: putting %.2f GB in %s (%s)\n", m.Name, float64(e.Bytes())/1e9, s, e.Path)
		if err := shared.PutEntry(ctx, s, dir); err != nil {
			return fmt.Errorf("%s: %s: %w", m.Name, s, err)
		}
	}
	return nil
}

// FirstRunExport is the script and the version of the export hwnpu-image makes of m on its
// first run on AMD's NPU, if it makes one.
func FirstRunExport(m chat.Model) (tool string, version int, ok bool) {
	switch m.Kind {
	case chat.KindDetect:
		return "iron_onnxcnn.py", detect.ExportVersion, true
	case chat.KindPose:
		return "iron_onnxcnn.py", pose.ExportVersion, true
	case chat.KindSegment:
		return "iron_efficientsam.py", segment.ExportVersion, true
	}
	return "", 0, false
}

// pushFirstRun puts the export of a first run (ironcnn.Dir) in the layers writes.
func pushFirstRun(ctx context.Context, writes []shared.Source, m chat.Model, tool string, version int) error {
	dir := ironcnn.Dir(m.Name)
	if _, err := os.Stat(filepath.Join(dir, "model.json")); err != nil {
		return fmt.Errorf("%s: no export %s here (made by its first run on AMD's NPU)", m.Name, filepath.Base(dir))
	}
	recipe := chat.PinnedRecipe(m, tool, version)
	if recipe == nil {
		return fmt.Errorf("%s: the definitions pin none of its files (sha256)", m.Name)
	}
	return describeAndPut(ctx, writes, m, chat.PlaceEntry(chat.ExportKind(tool), m, cache.AMDTarget, filepath.Base(dir), recipe), dir)
}

// GCCmd is hwnpu-dev sources gc.
type GCCmd struct {
	Source    int      `arg:"" help:"The source's number in hwnpu-dev sources list (0: this machine's cache)."`
	Model     []string `help:"Remove the entries of these models." placeholder:"NAME"`
	Kind      []string `help:"Remove the entries of these kinds (iron-lm, iron-onnxcnn...)." placeholder:"KIND"`
	Key       []string `help:"Remove the entries whose key starts with these (the HASH of vN-HASH)." placeholder:"KEY"`
	OlderThan int      `name:"older-than" help:"Remove the entries made more than this many days ago." placeholder:"DAYS"`
	Target    []string `help:"Only the entries of these targets (amd-xdna2, coreai...): with the others, which choose among them." placeholder:"TARGET"`
	Apply     bool     `help:"Remove them (without it, only show what would go)."`
	Pinned    bool     `help:"Remove the entries models.yaml pins (exports) too: without it they stay, whatever the others choose."`
}

func (g GCCmd) Run(d *devopt.Defs) error {
	if err := load(d); err != nil {
		return err
	}
	ctx, stop := signalContext()
	defer stop()
	l, err := source(g.Source)
	if err != nil {
		return err
	}
	if len(g.Model) == 0 && len(g.Kind) == 0 && len(g.Key) == 0 && g.OlderThan == 0 {
		return errors.New("say what to remove: --model, --kind, --key, or --older-than (what a writeback cut off left: hwnpu cache clean)")
	}
	before := time.Now().AddDate(0, 0, -g.OlderThan)
	kept := 0
	chosen := func(e shared.Entry) bool {
		if len(g.Target) > 0 && !slices.Contains(g.Target, e.Target) {
			return false
		}
		for _, m := range g.Model {
			if strings.EqualFold(e.Model, m) {
				return true
			}
		}
		for _, k := range g.Kind {
			if e.Kind == k {
				return true
			}
		}
		for _, k := range g.Key {
			if k != "" && strings.HasPrefix(e.Key, k) {
				return true
			}
		}
		return g.OlderThan > 0 && e.Made.Before(before)
	}
	drop := func(e shared.Entry) bool {
		if !chosen(e) {
			return false
		}
		if pin, ok := chat.PinOfEntry(e); ok && !g.Pinned && path.Base(e.Path) == pin.At {
			kept++
			return false
		}
		return true
	}
	plan, err := shared.PlanGC(ctx, l, drop)
	if err != nil {
		return err
	}
	for _, e := range plan.Entries {
		fmt.Printf("%s %s %.2f GB %s\n", e.Kind, e.Path, float64(e.Bytes())/1e9, e.Made.Local().Format(time.DateTime))
	}
	verb := "would remove"
	if g.Apply {
		verb = "removing"
	}
	fmt.Printf("%s: %s %d entries (%.2f GB)\n", l, verb, len(plan.Entries), float64(plan.Bytes)/1e9)
	if kept > 0 {
		fmt.Printf("kept %d entries models.yaml pins (--pinned removes them too)\n", kept)
	}
	if !g.Apply {
		if len(plan.Entries) > 0 {
			fmt.Println("(--apply removes them)")
		}
		return nil
	}
	return plan.Apply(ctx, l)
}

// VerifyCmd is hwnpu-dev sources verify.
type VerifyCmd struct {
	Dir  []string `arg:"" help:"Directories with entry.json." type:"existingdir"`
	Sums bool     `help:"Check the SHA-256 of every file too (reads them all)."`
}

func (v VerifyCmd) Run(*devopt.Defs) error {
	bad := 0
	for _, d := range v.Dir {
		if err := shared.Verify(d, v.Sums); err != nil {
			fmt.Printf("%s: %v\n", d, err)
			bad++
			continue
		}
		fmt.Printf("%s: ok\n", d)
	}
	if bad > 0 {
		return fmt.Errorf("%d of %d differ", bad, len(v.Dir))
	}
	return nil
}

// UpdateCmd is hwnpu-dev sources update.
type UpdateCmd struct {
	Repo   []string `arg:"" optional:"" help:"Models of the definitions or repositories (OWNER/NAME[@REVISION]) to move (all the sources hold when left out)."`
	Source int      `default:"-1" help:"Only this source (its number in hwnpu-dev sources list; 0: this machine's cache, which otherwise follows the sources)."`
}

func (u UpdateCmd) Run(d *devopt.Defs) error {
	if err := load(d); err != nil {
		return err
	}
	ctx, stop := signalContext()
	defer stop()
	writes := shared.Writebacks()
	if len(writes) == 0 {
		writes = []shared.Source{shared.Local()} // no shared source: this machine's cache moves itself
	}
	if u.Source >= 0 {
		s, err := source(u.Source)
		if err != nil {
			return err
		}
		writes = []shared.Source{s}
	}
	if len(writes) == 0 {
		return errors.New("no source is written back to: give one ?writeback=true (" + shared.SourceEnv + ")")
	}
	failed := 0
	for _, s := range writes {
		repos := u.Repo
		if len(repos) == 0 {
			var err error
			if repos, err = chat.LockedRepos(ctx, s); err != nil {
				return fmt.Errorf("%s: %w", s, err)
			}
		}
		for _, r := range repos {
			if m, ok := chat.Lookup(r); ok {
				r = m.Source()
			}
			res, err := chat.UpdateLock(ctx, s, r, console.Log(os.Stderr))
			switch {
			case errors.Is(err, shared.ErrNotFound):
				fmt.Printf("%s: %s: none there\n", s, r)
			case err != nil:
				fmt.Printf("%s: %s: %v\n", s, r, err)
				failed++
			case res.From == res.To:
				fmt.Printf("%s: %s: %.12s, up to date\n", s, r, res.To)
			default:
				fmt.Printf("%s: %s: %.12s -> %.12s (%d files the same, %d downloaded again, %d dropped)\n", s, r, res.From, res.To, res.Same, res.Changed, res.Dropped)
			}
		}
	}
	if failed > 0 {
		return fmt.Errorf("%d repositories were not moved", failed)
	}
	return nil
}

// ImportCmd is hwnpu-dev sources import.
type ImportCmd struct {
	Dir    []string `arg:"" help:"Folders: a checkpoint (DIR, or DIR=OWNER/NAME[@REVISION] when its name is no model's of the definitions) or an export for AMD's NPU (NAME.amd-npu2*, with manifest.json or model.json)."`
	DryRun bool     `name:"dry-run" help:"Only show what each folder is taken for."`
	Finish bool     `help:"Do to an export what its model's recipe's finish says (model-recipes.yaml models.NAME.amd.finish: its tokenizer files made binary) before putting it, in its folder: an export of the version before a recipe's finish was added, put as the new version without making it again."`
}

// importRepos are the repositories of folders whose name is no model's of the definitions.
var importRepos = map[string]string{
	"gpt-oss-20b": "openai/gpt-oss-20b",
	"yunet":       "opencv/face_detection_yunet",
}

// animaPaths map the folder of tools/amd/iron_anima.py (diffusion_models/, text_encoders/,
// diffusers/) to its two repositories.
var animaPaths = []struct{ repo, local, remote string }{
	{"circlestone-labs/Anima", "diffusion_models/", "split_files/diffusion_models/"},
	{"circlestone-labs/Anima", "text_encoders/", "split_files/text_encoders/"},
	{"circlestone-labs/Anima", "vae/", "split_files/vae/"},
	{"circlestone-labs/Anima-Base-v1.0-Diffusers", "diffusers/", ""},
}

func (ic ImportCmd) Run(d *devopt.Defs) error {
	if err := load(d); err != nil {
		return err
	}
	ctx, stop := signalContext()
	defer stop()
	writes := shared.Writebacks()
	if len(writes) == 0 && !ic.DryRun {
		return errors.New("no source is written back to: give one ?writeback=true (" + shared.SourceEnv + ")")
	}
	failed := 0
	for _, arg := range ic.Dir {
		dir, repo, _ := strings.Cut(arg, "=")
		dir = filepath.Clean(dir)
		if err := ic.one(ctx, writes, dir, repo); err != nil {
			fmt.Printf("%s: %v\n", dir, err)
			failed++
		}
	}
	if failed > 0 {
		return fmt.Errorf("%d folders were not put", failed)
	}
	return nil
}

func (ic ImportCmd) one(ctx context.Context, writes []shared.Source, dir, repo string) error {
	name := filepath.Base(dir)
	if strings.Contains(name, ".amd-npu2") {
		return ic.export(ctx, writes, dir)
	}
	type part struct {
		repo string
		path func(string) string
	}
	var parts []part
	switch {
	case repo != "":
		parts = []part{{repo, nil}}
	case name == "Anima":
		for _, a := range animaPaths {
			parts = append(parts, part{a.repo, func(p string) string {
				if rest, ok := strings.CutPrefix(p, a.local); ok {
					return a.remote + rest
				}
				return ""
			}})
		}
	default:
		if r, ok := importRepos[name]; ok {
			parts = []part{{r, nil}}
			break
		}
		for _, m := range chat.Catalog {
			if m.Repo != "" && m.DirName() == name {
				parts = []part{{m.Source(), nil}}
				break
			}
		}
	}
	if len(parts) == 0 {
		return errors.New("no model of the definitions is of this name: give DIR=OWNER/NAME")
	}
	for _, p := range parts {
		if ic.DryRun {
			fmt.Printf("%s: original %s\n", dir, p.repo)
			continue
		}
		for _, s := range writes {
			res, err := chat.ImportOriginal(ctx, s, p.repo, dir, p.path, console.Log(os.Stderr))
			if err != nil {
				return fmt.Errorf("%s: %w", p.repo, err)
			}
			fmt.Printf("%s: %s@%.12s: %d files put (%.2f GB), %d there, %d differ, %d not of it\n",
				dir, p.repo, res.Commit, res.Put, float64(res.Bytes)/1e9, res.Had, res.Bad, res.Skipped)
		}
	}
	return nil
}

// export puts an export for AMD's NPU made by hand: one of a model's recipe of iron_export_lm.py or first run
// as hwnpu-dev models push does, any other under the model whose name begins it (its own name
// when none), described by its name.
func (ic ImportCmd) export(ctx context.Context, writes []shared.Source, dir string) error {
	have := func(f string) bool { _, err := os.Stat(filepath.Join(dir, f)); return err == nil }
	if !have("manifest.json") && !have("model.json") && !have("dit_blocks") { // Anima's has neither
		return errors.New("not a whole export (no manifest.json or model.json)")
	}
	name := filepath.Base(dir)
	stem, _, _ := strings.Cut(name, ".amd-npu2")
	var m chat.Model
	for _, c := range chat.Catalog {
		n := strings.ToLower(c.Name)
		if l := strings.ToLower(stem); (l == n || strings.HasPrefix(l, n+"-") || strings.HasPrefix(l, n+".")) && len(c.Name) > len(m.Name) {
			m = c
		}
	}
	kind := chat.ImportKind
	var recipe map[string]string
	switch {
	case m.Name != "" && len(m.AMDExport) > 0 && filepath.Base(chat.AMDExportDir(m)) == name:
		kind, recipe = "iron-lm", chat.ExportRecipe(ctx, m, "iron_export_lm.py", m.AMDExport)
	case m.Name != "":
		if tool, version, ok := FirstRunExport(m); ok && strings.EqualFold(name, strings.ToLower(m.Name)+".amd-npu2") {
			kind, recipe = chat.ExportKind(tool), chat.PinnedRecipe(m, tool, version)
		}
	default:
		m = chat.Model{Name: stem}
	}
	if recipe == nil {
		recipe = chat.ImportedRecipe(m, cache.AMDTarget, name)
	}
	e := chat.PlaceEntry(kind, m, cache.AMDTarget, name, recipe)
	if ic.DryRun {
		fmt.Printf("%s: %s %s\n", dir, kind, e.Path)
		return nil
	}
	if ic.Finish {
		f, err := rcp.Builtin()
		if err != nil {
			return err
		}
		if err := finish.Run(dir, f.AMDFinish(m.Recipe)); err != nil {
			return err
		}
	} else if left, err := finish.Leftover(dir); err == nil && len(left) > 0 {
		return fmt.Errorf("the export holds %s, which its recipe's finish makes binary: --finish does it", strings.Join(left, ", "))
	}
	return describeAndPut(ctx, writes, m, e, dir)
}
