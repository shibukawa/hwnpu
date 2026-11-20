package coreai

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// The prebuilt programs of an export (a directory NAME.coreai of .aimodel programs) for an
// architecture are kept packed beside it, in NAME.coreai.prebuilt/ARCH/ with the export's
// layout: each X.ARCH.aimodelc without its weights (resources.bin, 99 % of it), which are the
// bytes of X.aimodel's main.mlirb in another order, with a map of that order in their place
// (resources.bin.map.json, Pack). That folder is what the shared layers carry for an
// architecture (docs/cache-sharing-design.md 7.9): a few percent of the export. Core AI is given the
// program assembled again from the export's bytecode (Assemble) and fills its cache entry from it.

// PackedSuffix is the folder beside an export that holds its packed prebuilt programs.
const PackedSuffix = ".prebuilt"

// MapSuffix ends the name of a map that stands for a file of a packed prebuilt program.
const MapSuffix = ".map.json"

// FetchPrebuilt, when set (the shared layers, chat), makes the packed prebuilt programs of the
// export for arch from the shared layers (PackedRoot) when one has them. The backend calls it
// once a process for an export with a program it would compile (no entry in Core AI's cache
// and no prebuilt program here).
var FetchPrebuilt func(ctx context.Context, export, arch string)

// ExportOf is the export (the nearest directory NAME.coreai) that holds the program at asset.
func ExportOf(asset string) (string, bool) {
	d := filepath.Dir(filepath.Clean(asset))
	for {
		if strings.EqualFold(filepath.Ext(d), ".coreai") {
			return d, true
		}
		p := filepath.Dir(d)
		if p == d {
			return "", false
		}
		d = p
	}
}

// PackedRoot is the folder of the export's packed prebuilt programs for arch.
func PackedRoot(export, arch string) string {
	return filepath.Clean(export) + PackedSuffix + string(filepath.Separator) + arch
}

// PackedPath is where the packed prebuilt program of the asset at dir for arch lies, "" when no
// export holds the asset.
func PackedPath(dir, arch string) string {
	export, ok := ExportOf(dir)
	if !ok {
		return ""
	}
	rel, err := filepath.Rel(export, PrebuiltPath(dir, arch))
	if err != nil {
		return ""
	}
	return filepath.Join(PackedRoot(export, arch), rel)
}

// PrebuiltPath is where a prebuilt program of the asset at dir for arch lies: coreai-build's
// name, the asset's base name with the architecture and .aimodelc.
func PrebuiltPath(dir, arch string) string {
	base := strings.TrimSuffix(dir, "/")
	if strings.EqualFold(filepath.Ext(base), ".aimodel") {
		base = base[:len(base)-len(".aimodel")]
	}
	return base + "." + arch + ".aimodelc"
}

// Assets lists the Core AI programs (asset directories, X.aimodel with main.mlirb) under paths: a
// path that is one, or every one below a directory (an export), prebuilt programs left out.
func Assets(paths []string) ([]string, error) {
	var out []string
	seen := map[string]bool{}
	add := func(p string) {
		p = filepath.Clean(p)
		if !seen[p] {
			seen[p] = true
			out = append(out, p)
		}
	}
	for _, p := range paths {
		if isAsset(p) {
			add(p)
			continue
		}
		fi, err := os.Stat(p)
		if err != nil {
			return nil, err
		}
		if !fi.IsDir() {
			return nil, fmt.Errorf("%s is not a Core AI program (an .aimodel directory) or a directory of them", p)
		}
		n := len(out)
		err = filepath.WalkDir(p, func(q string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if !d.IsDir() {
				return nil
			}
			if strings.HasSuffix(q, ".aimodelc") || strings.HasSuffix(q, PackedSuffix) {
				return filepath.SkipDir
			}
			if isAsset(q) {
				add(q)
				return filepath.SkipDir
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
		if len(out) == n {
			return nil, fmt.Errorf("%s holds no Core AI program", p)
		}
	}
	sort.Strings(out)
	return out, nil
}

// LeftToDevice reports whether the program at dir is compiled on the device even where a prebuilt
// one could be had: a VAE's (vae*.aimodel) or a text encoder's (text*.aimodel). coreai-build's VAE programs come out wrong on the
// device: FLUX.2 klein's at 512 x 512 (the 512 function of the 512+1024 export, prebuilt for
// h13g) drew a band of noise every 112 rows on M1, and the 1024 x 768 export's ended in
// "Request cancelled" at their first stage; the same programs compiled on the device draw the
// picture right. Nor do they gain anything: the device finishes a prebuilt program no faster than
// it compiles one this small (the ten stages at 1024 x 1024 104 s either way: a finish waits on
// the respawn of ANELargeModelCompilerService, 10 s a program). Build leaves them out and Prebuilt
// passes over them in what the shared layers hold already (design.md 16.261).
//
// The text encoders likewise: Chroma1-Flash's T5-XXL (text_0 ... text_21, prebuilt for h13g) ended
// in "Request cancelled" at text_0 on M1 four times running (fresh Core AI caches, nothing else on
// the Neural Engine), and the same programs compiled on the device ran (T5-XXL 220 to 230 s); FLUX.2
// klein's Qwen3 ran from its prebuilts but slower than compiled on the device (112 s against 81 s,
// 16.252), so a prebuilt text encoder gains nothing either.
func LeftToDevice(dir string) bool {
	base := strings.ToLower(filepath.Base(filepath.Clean(dir)))
	return strings.HasPrefix(base, "vae") || strings.HasPrefix(base, "text")
}

// isAsset reports whether dir is a Core AI program hwnpu lowered or converted: its bytecode and
// hash are there.
func isAsset(dir string) bool {
	if !strings.EqualFold(filepath.Ext(dir), ".aimodel") {
		return false
	}
	if fi, err := os.Stat(filepath.Join(dir, "main.mlirb")); err != nil || !fi.Mode().IsRegular() || fi.Size() == 0 {
		return false
	}
	_, err := os.Stat(filepath.Join(dir, "main.hash"))
	return err == nil
}

// ProgramsDigest is the SHA-256 of what the programs of the export are (each one's path in it
// and main.hash, the hash of its bytecode): the prebuilt programs of an export are of the
// programs, whatever made them. "" when the export holds none.
func ProgramsDigest(export string) string {
	assets, err := Assets([]string{export})
	if err != nil || len(assets) == 0 {
		return ""
	}
	h := sha256.New()
	for _, a := range assets {
		rel, err := filepath.Rel(export, a)
		if err != nil {
			return ""
		}
		sum, err := os.ReadFile(filepath.Join(a, "main.hash"))
		if err != nil {
			return ""
		}
		fmt.Fprintf(h, "%s\x00%x\n", filepath.ToSlash(rel), sum)
	}
	return hex.EncodeToString(h.Sum(nil))
}
