// Package modelstore finds and fetches the files of the models hwnpu runs, as hwnpu's commands
// do (docs/cache-sharing.md 7). A file is looked for in layers, in order:
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
	"context"
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
	// Log is where downloads and copies are shown (nil for nowhere).
	Log io.Writer
}

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
// the model definitions (hwnpu's models.yaml, the machine's and the user's) too.
func Open(o Options) (*Store, error) {
	cache.SetRoot(o.Root)
	if err := chat.LoadModels(nil, true, o.Log); err != nil {
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
