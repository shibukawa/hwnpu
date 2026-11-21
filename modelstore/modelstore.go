// Package modelstore finds and fetches the files of the models hwnpu runs, as hwnpu's commands
// do (docs/cache-sharing-design.md 7). A file is looked for in layers, in order:
//
//	0  this machine's cache  Options.Root (HWNPU_CACHE, else hwnpu in the user cache directory),
//	                         what is read from the other layers: written back to always
//	1  the sources           Options.Sources (HWNPU_MODEL_SOURCE, HWNPU_MODEL_SOURCE_1, ...):
//	                         file://, http(s):// (WebDAV, or any server read only), s3:// URLs;
//	                         ?writeback=true puts in one what is read from a later layer
//	2  the bundled models    Options.Bundle: the models the application installs with it
//	                         (read only, used where they are)
//	3  Hugging Face          unless Options.HuggingFace is false (HWNPU_MODEL_HUGGING_FACE)
//
// A repository's files in a layer are those its hwnpu-lock.json names, of one commit: the
// commit a model pins, else the one of the first lock after this machine's cache, else this
// machine's, else Hugging Face's main. Every file is kept here in Store.Dir of its repository
// (original/OWNER/NAME under the root), whichever layer it came from.
//
// The layers are a setting of the process: Open sets them for everything in it, hwnpu's own
// code too.
package modelstore

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"

	"github.com/shibukawa/configdir"

	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	shared "github.com/shibukawa/hwnpu/internal/store"
)

// Options say where the models come from. The zero value is what hwnpu's commands read: the
// environment (and its .env files), without bundled models.
type Options struct {
	// Root is this machine's cache, layer 0: "" for HWNPU_CACHE, else hwnpu in the user cache
	// directory (~/.cache/hwnpu, ~/Library/Caches/hwnpu, %LOCALAPPDATA%\hwnpu).
	Root string
	// Sources are the URLs of the sources, layer 1 (as HWNPU_MODEL_SOURCE's): nil for the
	// environment's, empty for none.
	Sources []string
	// HuggingFace says whether Hugging Face is read last: nil for HWNPU_MODEL_HUGGING_FACE (true
	// when it is not set).
	HuggingFace *bool
	// Bundle is the application whose resource folders (configdir's QueryResourceFolders: next
	// to the executable, X.app/Contents/Resources, PREFIX/share/APP...) hold the models it
	// installs with it, in BundleFolder: layer 2. nil for none.
	Bundle *configdir.ConfigDir
	// BundleFolder is the folder of each resource folder with the models, in the layout of a
	// source (original/OWNER/NAME/ with hwnpu-lock.json, converted/): "models" when empty.
	BundleFolder string
	// Limits are what this machine's cache may take of the disk: nil for HWNPU_CACHE_MAX_SIZE and
	// HWNPU_CACHE_MIN_FREE (10GB free unless set). A download that would leave less free, or put
	// the cache over MaxSize, removes the units least recently used first when there is a MaxSize,
	// and fails (ErrNoRoom) when that is not enough.
	Limits *Limits
	// Files are definition files of the application's own (models.yaml's format: its models,
	// uses and aliases), read over hwnpu's, the machine's and the user's, as hwnpu-chat's
	// -models are.
	Files []string
	// Log is where downloads and copies are shown (nil for nowhere).
	Log io.Writer
}

// Limits are what the cache may take of the disk (cache.Limits).
type Limits = cache.Limits

// Unit is a part of the cache that is removed whole: a repository's files (original/OWNER/NAME),
// a model's programs (models/NAME, converted/...), an export for AMD's NPU, a kernel IRON
// compiled, a program compiled for this machine.
type Unit = cache.Unit

// Usage is what this machine's cache holds.
type Usage = cache.Usage

// PruneOptions say what Prune removes.
type PruneOptions = cache.PruneOptions

// ErrNoRoom is a download's error when the limits do not let it in.
var ErrNoRoom = cache.ErrNoRoom

// Store fetches the models' files from the layers Open set.
type Store struct {
	log io.Writer
}

// Layer is a layer of a Store, in order.
type Layer struct {
	Name string // where it is (a URL's password left out)
	// Writeback says it gets what is read from a later layer.
	Writeback bool
	// Bundled says it is a folder of models installed with the application (read only).
	Bundled bool
}

// Open sets the layers of the process as o says and returns the Store that reads them. It reads
// the model definitions (hwnpu's models.yaml, the machine's, the user's and o.Files) too.
func Open(o Options) (*Store, error) {
	cache.SetRoot(o.Root)
	cache.SetLimits(o.Limits)
	if err := chat.LoadModels(o.Files, true, o.Log); err != nil {
		return nil, err
	}
	sources, hf := shared.Sources(), shared.HuggingFace()
	if o.Sources != nil {
		sources = nil
		for i, u := range o.Sources {
			s, err := shared.ParseSource("modelstore source "+strconv.Itoa(i), u)
			if err != nil {
				return nil, err
			}
			sources = append(sources, s)
		}
	} else {
		sources = withoutBundles(sources)
	}
	if o.HuggingFace != nil {
		hf = *o.HuggingFace
	}
	shared.SetSources(sources, hf)
	var bundles []string
	if o.Bundle != nil {
		bundles = chat.BundleDirs(*o.Bundle, o.BundleFolder)
	}
	shared.SetBundles(bundles)
	log := o.Log
	if f, ok := log.(*os.File); ok {
		log = console.Log(f) // a terminal's: the bars of the downloads drawn in place
	}
	return &Store{log: log}, nil
}

func withoutBundles(ss []shared.Source) []shared.Source {
	var out []shared.Source
	for _, s := range ss {
		if s.From != "bundle" {
			out = append(out, s)
		}
	}
	return out
}

// File returns the path of a file of a Hugging Face repository (OWNER/NAME, or
// OWNER/NAME@REVISION) here, which it puts there from the layers unless it is there. A file
// the definitions pin (models.yaml's sha256) is checked.
func (s *Store) File(ctx context.Context, repo, name string) (string, error) {
	return chat.Fetch(ctx, repo, name, s.log)
}

// Weights puts a model's checkpoint here: model.safetensors, or its index and the files it
// names. It returns the path of model.safetensors, by which hwnpu opens either.
func (s *Store) Weights(ctx context.Context, repo string) (string, error) {
	return chat.FetchWeights(ctx, repo, s.log)
}

// Dir is the folder the files of repo are kept in here (original/OWNER/NAME under the root).
func (s *Store) Dir(repo string) string { return chat.RepoDir(repo) }

// NotFound reports whether err is File's for a file no layer has.
func NotFound(err error) bool { return chat.NotFound(err) }

// Layers are the layers in effect, in order (Hugging Face last, when it is read).
func (s *Store) Layers() []Layer {
	ls := []Layer{{Name: fmt.Sprintf("this machine's cache (%s)", cache.Root()), Writeback: true}}
	for _, src := range shared.Sources() {
		ls = append(ls, Layer{Name: src.String(), Writeback: src.Writeback, Bundled: src.From == "bundle"})
	}
	if shared.HuggingFace() {
		ls = append(ls, Layer{Name: "Hugging Face"})
	}
	return ls
}

// Usage walks this machine's cache and counts what it holds, by unit.
func (s *Store) Usage() (Usage, error) { return cache.Scan() }

// Limits are the limits in effect (Options.Limits, else the variables').
func (s *Store) Limits() (Limits, error) { return cache.CurrentLimits() }

// DiskSpace is the bytes free and the size of the disk of this machine's cache.
func (s *Store) DiskSpace() (free, total int64, err error) { return cache.DiskSpace(cache.Root()) }

// PlanPrune is what Prune(o) would remove, the least recently used first: units used in the last
// hour (cache.KeepRecent unless o.Keep), by this process, or with a writeback queued are kept.
func (s *Store) PlanPrune(o PruneOptions) ([]Unit, error) {
	u, err := cache.Scan()
	if err != nil {
		return nil, err
	}
	return cache.PlanPrune(u, o), nil
}

// Prune removes the units o says, and returns them.
func (s *Store) Prune(o PruneOptions) ([]Unit, error) {
	units, err := s.PlanPrune(o)
	if err != nil {
		return nil, err
	}
	return s.Remove(units)
}

// SourceClean is what a writeback cut off left in a source written back to (PlanClean): the
// folders that are not whole and the temporary files of uploads.
type SourceClean = shared.Clean

// CleanFolder is a folder of a source SourceClean removes files of.
type CleanFolder = shared.CleanFolder

// PlanClean works out what a writeback cut off left in each source written back to (not this
// machine's cache, not the bundled models): an export's folder with no entry.json (all its
// files), a repository's files its hwnpu-lock.json does not name, the temporary files of
// uploads. A folder a writeback marked in the last hour (store.WritingStale) is being written and
// is left. Nothing is removed until Clean.
func (s *Store) PlanClean(ctx context.Context) ([]SourceClean, error) {
	var out []SourceClean
	for _, src := range shared.Writebacks() {
		if src.From == "bundle" || src.From == cache.RootEnv {
			continue
		}
		c, err := shared.PlanClean(ctx, src)
		if err != nil {
			return out, fmt.Errorf("%s: %w", src, err)
		}
		out = append(out, c)
	}
	return out, nil
}

// Clean removes what plans name, each folder looked at again first (one a writeback has started
// in since is left), and returns how many files it removed.
func (s *Store) Clean(ctx context.Context, plans []SourceClean) (int, error) {
	var errs []error
	n := 0
	for _, c := range plans {
		m, err := c.Apply(ctx)
		n += m
		if err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", c.Source, err))
		}
	}
	return n, errors.Join(errs...)
}

// Remove removes units (of Usage), and returns those it removed. A unit another process holds
// open may stay (Windows).
func (s *Store) Remove(units []Unit) ([]Unit, error) {
	var done []Unit
	var errs []error
	for _, u := range units {
		if err := u.Remove(); err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", u.Name, err))
			continue
		}
		done = append(done, u)
	}
	return done, errors.Join(errs...)
}

// Memory is the memory a model runs as on this machine (Store.Memory).
type Memory struct {
	Model   string // the model's name in the definitions
	GB      int    // the memory it runs as: Chosen, else Machine
	Chosen  int    // chosen for it (SetMemory), 0 for none
	Machine int    // this machine's (HWNPU_MEMORY, else the physical memory)
}

// Memory is the memory model name (a name of the definitions, its repository, or a family or a
// use: the model it picks here) runs as on this machine. Every hwnpu command and a Store's
// application run a model as on a machine of that memory: its programs are those models.yaml
// defines for it (variants), converted or fetched for that alone.
func (s *Store) Memory(name string) (Memory, error) {
	m, err := chat.Named(name)
	if err != nil {
		return Memory{}, err
	}
	mem := Memory{Model: m.Name, Chosen: chat.ChosenMemory(m), Machine: chat.MachineMemory()}
	mem.GB = cmp.Or(mem.Chosen, mem.Machine)
	return mem, nil
}

// SetMemory chooses gb as the memory model name runs as on this machine, in every hwnpu command
// (a 16 GB Mac that runs a model the 8 GB way, beside other programs); 0 forgets it. It is kept
// in hwnpu's directory of settings (run.json beside the user's models.yaml), not in the cache:
// pruning the cache keeps it, and another machine reading the same cache does not get it.
func (s *Store) SetMemory(name string, gb int) (Memory, error) {
	m, err := chat.Named(name)
	if err != nil {
		return Memory{}, err
	}
	if err := chat.ChooseMemory(m, gb); err != nil {
		return Memory{}, err
	}
	return s.Memory(m.Name)
}

// Memories are the models with a memory chosen (SetMemory), by name.
func (s *Store) Memories() map[string]int { return chat.ChosenMemories() }
