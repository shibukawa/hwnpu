package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/internal/catalog"
	"github.com/shibukawa/hwnpu/internal/imagewire"
)

// hwnpu-image runs no engine of its own: its models run in hwnpud through the ai package, and
// what it asks that is not a model (its list, why a family picked a model, an export's family,
// an export the definitions pin, Anima's conversion) goes to hwnpud's engine on a session of its
// own (internal/imagewire.Host). With HWNPU_DAEMON=off it says that it needs hwnpud.

// hwnpud runs fn on a session with hwnpud, whose notes went to the log of ai's session before.
func hwnpud(ctx context.Context, fn func(*imagewire.Session) error) error {
	s, err := imagewire.Dial(ctx, io.Discard)
	if err != nil {
		return err
	}
	defer s.Close()
	return fn(s)
}

// npuVendor is the vendor of this machine's NPU ("amd", "apple", "intel"), "" without one:
// HWNPU_BACKEND's (openvino is an Intel machine), else the one hwnpud found (cat's).
func npuVendor(cat *ai.Catalog) string {
	if v := imagewire.Vendor(os.Getenv(imagewire.BackendEnv)); v != "" {
		return v
	}
	return cat.Spec().Vendor
}

// listModels writes the list of the models of kinds with, in place of the download, what each
// runs from on this machine (hwnpud's: the CPU, an AMD export, converted on the first run, or
// "-"). An error is written in its place.
func listModels(w io.Writer, cat *ai.Catalog, kinds ...catalog.Kind) {
	ks := make([]string, len(kinds))
	for i, k := range kinds {
		ks[i] = string(k)
	}
	err := hwnpud(context.Background(), func(s *imagewire.Session) error { return s.List(context.Background(), w, npuVendor(cat), ks...) })
	if err != nil {
		fmt.Fprintf(w, "(the list: %v)\n", err)
	}
}

// picked says on the standard error why name, a family or a use, picked the model it picked.
func picked(ctx context.Context, name string) error {
	return hwnpud(ctx, func(s *imagewire.Session) error { return s.Picked(ctx, name, os.Stderr) })
}

// imageExport is the model an export's directory runs as (ai/imagegen.Export, in hwnpud).
func imageExport(ctx context.Context, dir string) (d ai.Definition, err error) {
	if dir, err = filepath.Abs(dir); err != nil { // hwnpud's working directory is not ours
		return d, err
	}
	err = hwnpud(ctx, func(s *imagewire.Session) error {
		d, err = s.Export(ctx, dir)
		return err
	})
	return d, err
}

// pinnedExport is the export the definitions pin by name for AMD's NPU, here or fetched from
// hwnpud's sources; "" for none.
func pinnedExport(ctx context.Context, name string, log io.Writer) (dir string, err error) {
	err = hwnpud(ctx, func(s *imagewire.Session) error {
		dir, err = s.Pinned(ctx, name, log)
		return err
	})
	return dir, err
}
