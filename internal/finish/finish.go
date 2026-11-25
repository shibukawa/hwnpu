// Package finish does to a finished export what its recipe's finish says (model-recipes.yaml
// models.NAME.amd.finish, models.NAME.apple.finish): the steps after the script or the
// conversion that made it, before it is moved into place and described (entry.json).
package finish

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/shibukawa/hwnpu/internal/recipe"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

// Run does steps to the export in dir, then checks that it holds no tokenizer file the steps do
// not make binary (recipe.IsTokenizerFile): a model whose recipe forgets one stops its build
// rather than ship the text the runtime would parse. A step done before (its binary file there,
// its text gone: an export finished already) is done again by nothing.
func Run(dir string, steps []recipe.Step) error {
	made := map[string]string{}
	for _, s := range steps {
		for _, f := range s.Tokenizer {
			src := filepath.Join(dir, filepath.FromSlash(f))
			dst := tokenizer.BinaryPath(src)
			if other, ok := made[dst]; ok {
				return fmt.Errorf("finish: %s and %s make the same %s", other, f, filepath.Base(dst))
			}
			made[dst] = f
			if _, err := os.Stat(src); errors.Is(err, fs.ErrNotExist) {
				if _, err := os.Stat(dst); err == nil {
					continue // finished before
				}
				return fmt.Errorf("finish: the export has no %s (finish: tokenizer)", f)
			}
			if err := tokenizer.WriteBinary(src, dst); err != nil {
				return fmt.Errorf("finish: %s: %w", f, err)
			}
			if err := os.Remove(src); err != nil {
				return err
			}
		}
	}
	left, err := Leftover(dir)
	if err != nil {
		return err
	}
	if len(left) > 0 {
		return fmt.Errorf("finish: the export holds %s, which its recipe's finish does not make binary (add finish: [{tokenizer: [%s]}])", strings.Join(left, ", "), strings.Join(left, ", "))
	}
	return nil
}

// Check reports what keeps the export in dir, made by hand (hwnpu-dev sources import, its first
// use here: chat.ShareMadeHere) or converted (chat.FinishCoreAI), out of the sources: a tokenizer
// file not made binary yet (Leftover), or the data of a run of its tool with --check (Checks),
// which only the tools of tools/amd read. how is what makes the tokenizer binary, for the
// message ("": nothing in hand does, the recipe lacks a finish for it).
func Check(dir, how string) error {
	left, err := Leftover(dir)
	if err != nil {
		return err
	}
	if len(left) > 0 {
		if how == "" {
			how = fmt.Sprintf("its recipe's finish (add finish: [{tokenizer: [%s]}] and raise its version)", strings.Join(left, ", "))
		}
		return fmt.Errorf("the export holds %s, which %s makes binary", strings.Join(left, ", "), how)
	}
	checks, err := Checks(dir)
	if err != nil {
		return err
	}
	if len(checks) > 0 {
		return fmt.Errorf("the export holds %s, the data of --check (read by tools/amd's checks, not by hwnpu): remove it first", strings.Join(checks, ", "))
	}
	return nil
}

// Checks are the folders of the data of a run of a tools/amd script with --check in the export
// in dir (check/ beside a manifest.json or model.json: iron_graph.write_check, iron_export_lm.py
// --check), as paths in it.
func Checks(dir string) ([]string, error) {
	var out []string
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() || d.Name() != "check" {
			return nil
		}
		for _, f := range []string{"manifest.json", "model.json"} {
			if _, err := os.Stat(filepath.Join(filepath.Dir(p), f)); err == nil {
				rel, _ := filepath.Rel(dir, p)
				out = append(out, filepath.ToSlash(rel)+"/")
				return filepath.SkipDir
			}
		}
		return nil
	})
	sort.Strings(out)
	return out, err
}

// Leftover are the tokenizer files (recipe.IsTokenizerFile) in the export in dir, as paths in it.
func Leftover(dir string) ([]string, error) {
	var out []string
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(dir, p)
		if !d.IsDir() && recipe.IsTokenizerFile(filepath.ToSlash(rel)) {
			out = append(out, filepath.ToSlash(rel))
		}
		return nil
	})
	sort.Strings(out)
	return out, err
}
