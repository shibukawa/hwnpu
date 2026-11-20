package chat

import (
	"context"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu/internal/cache"
	shared "github.com/shibukawa/hwnpu/internal/store"
)

// CodesKind is the kind of entry of the lookup tables chosen against the activations (GPTQ's
// codes, models.yaml's codes) that a model's programs are converted with.
const CodesKind = "codes"

// codesRecipe is what the codes of hash for the model whose directory is dir are, their key in
// the shared layers: the model's part (modelRecipe) and the hash of their index. The programs
// converted with them can be taken from the sources, but only while the sources hold the
// conversion of this hwnpu: when what it writes changes (ailower.FormatVersion), a conversion
// needs the codes again, and the machine that chose them may not have them any more.
func codesRecipe(ctx context.Context, dir, hash string) (Model, map[string]string) {
	// the model of the directory the codes lie in (modelOfDir looks from a path's parent up)
	m, r := modelRecipe(ctx, CodesDir(dir, hash))
	if r == nil {
		fmt.Fprintf(sharedLog, "%s: the model of %s is not known; the codes are not shared\n", filepath.Base(CodesDir(dir, hash)), dir)
		return Model{}, nil
	}
	r["kind"], r["codes"] = CodesKind, hash
	return m, r
}

// fetchCodes makes the codes of hash in dst (CodesDir) from the shared layers and reports whether
// one had them, with an index of that hash.
func fetchCodes(ctx context.Context, dir, hash, dst string) bool {
	m, r := codesRecipe(ctx, dir, hash)
	if r == nil {
		return false
	}
	tmp := dst + ".shared"
	os.RemoveAll(tmp)
	defer os.RemoveAll(tmp)
	if !fetchShared(ctx, CodesKind, m, r, tmp, filepath.Base(dst)) {
		return false
	}
	if h, err := codesHash(tmp); err != nil || h != hash {
		fmt.Fprintf(sharedLog, "%s: the codes from the sources are %s, not %s; not taken\n", filepath.Base(dst), h, hash)
		return false
	}
	os.RemoveAll(dst)
	return os.Rename(tmp, dst) == nil
}

// shareCodes puts the codes of hash in src in the shared layers that are written (putCodes)
// unless they were put or fetched before: their entry.json, written when they are put, says so.
func shareCodes(ctx context.Context, dir, hash, src string) {
	if _, err := os.Stat(filepath.Join(src, shared.EntryFile)); err == nil {
		return
	}
	if work := putCodes(ctx, dir, hash, src); work != "" {
		copyStream(filepath.Join(src, shared.EntryFile), filepath.Join(work, shared.EntryFile))
	}
}

// putCodes puts the codes of hash in src, which a conversion here reads, in the shared layers
// that are written: a directory of the work area holds them (hard links, else copies) until the
// queue has put them. It returns that directory, "" when nothing is put.
func putCodes(ctx context.Context, dir, hash, src string) string {
	m, r := codesRecipe(ctx, dir, hash)
	if r == nil || len(shared.Writebacks()) == 0 {
		return ""
	}
	work := cache.Sub("work", "shared", shared.Fingerprint(r))
	os.RemoveAll(work)
	err := filepath.WalkDir(src, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(src, p)
		if err != nil {
			return err
		}
		if rel == shared.EntryFile {
			return nil // the description of where they came from: written again for here
		}
		dst := filepath.Join(work, rel)
		if d.IsDir() {
			return os.MkdirAll(dst, 0o755)
		}
		if !d.Type().IsRegular() {
			return nil
		}
		if os.Link(p, dst) == nil {
			return nil
		}
		return copyStream(dst, p)
	})
	if err != nil {
		fmt.Fprintf(sharedLog, "%s: %v; not put in the shared layers\n", filepath.Base(src), err)
		os.RemoveAll(work)
		return ""
	}
	putShared(CodesKind, m, r, work, "codes-"+hash)
	if _, err := os.Stat(filepath.Join(work, shared.EntryFile)); err != nil {
		return "" // not described: not put
	}
	return work
}
