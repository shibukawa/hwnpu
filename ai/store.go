package ai

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/modelstore"
)

// Store is where an app's models come from and where they are kept prepared.
type Store struct {
	// Models are where the models' files come from and this machine's cache they are kept in,
	// as hwnpu's commands read them (modelstore: this machine's cache, HWNPU_MODEL_SOURCE*, the
	// models bundled with the app, Hugging Face), and the app's own definition files
	// (Models.Files). The zero value is the commands': what they downloaded and converted is
	// used as it is, and what the app prepares they use.
	Models modelstore.Options
	// Prepared are caches prepared beforehand and shipped with the app, read only and used where
	// they are: directories with a StoreFile whose paths are relative to them (a copy of the
	// prepared parts of a cache). They are looked in before this machine's cache. Unlike the
	// bundled models of Models.Bundle, which preparing fetches the files of, a use found in one
	// runs without being prepared here.
	Prepared []string
}

// openMu keeps the process's layers and definitions (modelstore.Open) from changing while a
// Plan or an Open reads them.
var openMu sync.Mutex

// open sets the process's layers and definitions as s says (modelstore.Open). Prepared models
// and the compile caches are then under cache.Root.
func (s Store) open() error {
	if _, err := modelstore.Open(s.Models); err != nil {
		return fmt.Errorf("ai: %w", err)
	}
	return nil
}

// dir is the directory of this machine's cache the StoreFile is in and its paths are relative
// to: cache.Root, which holds the models' programs (models/, entries/) and their checkpoints
// (original/).
func (s Store) dir() string { return cache.Root() }

// cacheDir is the compile cache of the NPU's programs. Apple's compiled programs belong to the
// machine and its OS, so they are not shipped.
func (s Store) cacheDir() string { return cache.Root() }

// StoreFile is the file in a store (this machine's cache, or a directory of Store.Prepared) that
// says what each use was prepared as.
const StoreFile = "store.json"

// storeFormat is the version of StoreFile's format this package writes and reads.
const storeFormat = 2

var (
	// ErrNotPrepared is the error of a use the store has not prepared: Plan.Run prepares it.
	ErrNotPrepared = errors.New("ai: not prepared")
	// ErrBundleTarget is the error of a store prepared for another NPU alone (a bundle for
	// Apple's Neural Engine on a machine with AMD's NPU), or of a format this package does not
	// read.
	ErrBundleTarget = errors.New("ai: prepared for another target")
)

// manifest is StoreFile: the uses prepared, by the NPU their programs are for (targetKey).
type manifest struct {
	Format  int                           `json:"format"`
	Targets map[string]map[string]*record `json:"targets"`
}

// record is what a use was prepared as. Its paths are relative to the store's directory, or
// absolute for files outside it.
type record struct {
	Model string `json:"model"`
	Role  string `json:"role"`
	// Run is the model's definition for the memory it runs as (Model.On): the variant its
	// programs were converted for.
	Run      chat.Run  `json:"run"`
	MemoryGB float64   `json:"memory_gb,omitempty"`
	Prepared time.Time `json:"prepared"`
	// Compiled is when the model was first loaded on this machine's NPU (the zero time: not
	// yet): its programs are compiled for it, so that the next load does not wait for that.
	Compiled time.Time    `json:"compiled,omitzero"`
	Files    airole.Files `json:"files"`
}

// each calls fn with a pointer to every path of the record.
func (r *record) each(fn func(*string)) {
	for _, ps := range r.Files.Paths {
		for i := range ps {
			fn(&ps[i])
		}
	}
}

// relative makes the record's paths under root relative to it.
func (r *record) relative(root string) {
	r.each(func(p *string) {
		if !filepath.IsAbs(*p) {
			return
		}
		if rel, err := filepath.Rel(root, *p); err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			*p = filepath.ToSlash(rel)
		}
	})
}

// absolute makes the record's relative paths absolute under root.
func (r *record) absolute(root string) {
	r.each(func(p *string) {
		if !filepath.IsAbs(*p) {
			*p = filepath.Join(root, filepath.FromSlash(*p))
		}
	})
}

// missing is the first of the record's files (absolute) that is not there, "" when all are.
func (r *record) missing() string {
	miss := ""
	r.each(func(p *string) {
		if miss == "" {
			if _, err := os.Stat(*p); err != nil {
				miss = *p
			}
		}
	})
	return miss
}

// clone is a deep copy of the record.
func (r *record) clone() *record {
	raw, _ := json.Marshal(r)
	var c record
	_ = json.Unmarshal(raw, &c)
	return &c
}

// readManifest reads the StoreFile of root; an empty one when there is none, or when it is of
// an older format in this machine's cache (mine: what it names is prepared again). A format
// that is not read in a prepared cache is ErrBundleTarget.
func readManifest(root string, mine bool) (*manifest, error) {
	empty := &manifest{Format: storeFormat, Targets: map[string]map[string]*record{}}
	file := filepath.Join(root, StoreFile)
	raw, err := os.ReadFile(file)
	if errors.Is(err, os.ErrNotExist) {
		return empty, nil
	}
	if err != nil {
		return nil, err
	}
	m := &manifest{}
	if err := json.Unmarshal(raw, m); err != nil {
		return nil, fmt.Errorf("ai: %s: %w", file, err)
	}
	if m.Format != storeFormat {
		if mine && m.Format < storeFormat {
			return empty, nil
		}
		return nil, fmt.Errorf("%w: %s is of format %d, this hwnpu reads %d", ErrBundleTarget, file, m.Format, storeFormat)
	}
	if m.Targets == nil {
		m.Targets = map[string]map[string]*record{}
	}
	return m, nil
}

// write writes the manifest as root's StoreFile, through a temporary file.
func (m *manifest) write(root string) error {
	raw, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(root, 0o755); err != nil {
		return err
	}
	tmp := filepath.Join(root, StoreFile+".tmp")
	if err := os.WriteFile(tmp, append(raw, '\n'), 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, filepath.Join(root, StoreFile))
}

// targetKey is the name the uses prepared for npu are kept under in a StoreFile: the NPU's
// target (coreai, intel, amd-xdna2), else the program format converted for without an NPU.
func targetKey(npu airole.NPU) string { return cmp.Or(npu.Target, npu.Format, "none") }

// located is a use's record found in a store, its paths absolute.
type located struct {
	*record
	root   string // the store's directory
	bundle bool   // of Store.Prepared
}

// find looks for key's record for npu in the prepared caches, then in this machine's, with its
// files all there, prepared as the model want and the way it runs (its definition's run for the
// memory it runs as): what was prepared as another model, or for another memory, is not what the
// app runs now (its line-up changed, or the machine's memory did), so it is as good as not there.
// A prepared cache that has the use for other NPUs alone is reported as ErrBundleTarget when
// nothing else is found.
func (s Store) find(key string, npu airole.NPU, want chat.Model) (located, error) {
	var mismatch error
	stale := ""
	target := targetKey(npu)
	roots := append(append([]string(nil), s.Prepared...), s.dir())
	for i, root := range roots {
		bundle := i < len(s.Prepared)
		m, err := readManifest(root, !bundle)
		if err != nil {
			return located{}, err
		}
		r, ok := m.Targets[target][key]
		if !ok {
			for other, uses := range m.Targets {
				if _, ok := uses[key]; ok && bundle {
					mismatch = fmt.Errorf("%w: %s in %s is for %s, this machine is %s", ErrBundleTarget, key, root, other, target)
				}
			}
			continue
		}
		if r.Model != want.Name || r.Run != want.Run {
			stale = r.Model
			continue
		}
		r = r.clone()
		r.absolute(root)
		if f := r.missing(); f != "" {
			continue
		}
		return located{record: r, root: root, bundle: bundle}, nil
	}
	switch {
	case mismatch != nil:
		return located{}, mismatch
	case stale != "" && stale != want.Name:
		return located{}, fmt.Errorf("%w: %s is prepared as %s, and is %s now (the app's Plan and its Run prepare it)", ErrNotPrepared, key, stale, want.Name)
	case stale != "":
		return located{}, fmt.Errorf("%w: %s (%s) is prepared for another memory (the app's Plan and its Run prepare it)", ErrNotPrepared, key, want.Name)
	}
	return located{}, fmt.Errorf("%w: %s (the app's Plan and its Run prepare it)", ErrNotPrepared, key)
}

// put writes key's record for npu into this machine's StoreFile, under the lock of the file
// (other processes prepare other uses in the same cache).
func (s Store) put(ctx context.Context, key string, npu airole.NPU, r *record) error {
	dir := s.dir()
	file := filepath.Join(dir, StoreFile)
	unlock, err := cache.Lock(ctx, file, file, io.Discard)
	if err != nil {
		return err
	}
	defer unlock()
	m, err := readManifest(dir, true)
	if err != nil {
		return err
	}
	target := targetKey(npu)
	if m.Targets[target] == nil {
		m.Targets[target] = map[string]*record{}
	}
	r = r.clone()
	r.relative(dir)
	m.Targets[target][key] = r
	return m.write(dir)
}
