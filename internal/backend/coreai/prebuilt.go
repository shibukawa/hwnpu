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

// Archs are the architectures coreai-build compiles for on macOS (Metal Toolchain 27.2): a name
// for each chip variant, as AIModel's deviceArchitectureName. The chips, as coreai-build inspect
// names them for each (seen on a device: h13c, h14g): g the chip (h13g M1 ... h17g M5, h18g M6),
// s its Pro, c its Max, d its Ultra (h17d none named); h15m an M3 Max too, h17p the A18 and A18
// Pro.
var Archs = []string{
	"h13c", "h13d", "h13g", "h13s",
	"h14c", "h14d", "h14g", "h14s",
	"h15c", "h15d", "h15g", "h15m", "h15s",
	"h16c", "h16g", "h16s",
	"h17c", "h17d", "h17g", "h17p", "h17s",
	"h18g",
}

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

// LeftToDevice reports whether the program at dir is compiled on the device on every chip even
// where a prebuilt one could be had: a program whose name ends in _device (NAME_device.aimodel),
// one coreai-build leaves to the GPU where the device's compiler takes it, as FLUX.2 klein's edits
// of 8192 keys or more (flux2klein.PrebuiltKeys). What is left to the device on some chips alone,
// or for one model, is the recipe's (model-recipes.yaml apple prebuild, do: device): the VAEs and
// the text encoders on M1 and M2 (docs/design/image-apple.md).
//
// The text encoders were left to the device on every chip until 2026-11-26: Chroma1-Flash's T5-XXL
// (removed) ended in "Request cancelled" from its h13g prebuilts, and FLUX.2 klein's Qwen3 was
// slower from prebuilts. They are prebuilt now for the memory: on an M3 the nine Qwen3 programs
// took 94 to 190 s from prebuilts against 46 s compiled on the device, but the device's compiler
// (ANECompilerService) grew by 1.5 GB doing it, the prebuilts' finishing service by 0.5 GB, and
// both left the same 7.9 GB in the caches.
func LeftToDevice(dir string) bool {
	base := strings.ToLower(filepath.Base(filepath.Clean(dir)))
	return strings.HasSuffix(strings.TrimSuffix(base, ".aimodel"), "_device")
}

// NothingToPrebuild reports whether every program under export is LeftToDevice (FLUX.2 klein's
// DeviceExport): no prebuilt of it is made, or looked for.
func NothingToPrebuild(export string) bool {
	all, err := Assets([]string{export})
	if err != nil {
		return false
	}
	for _, a := range all {
		if !LeftToDevice(a) {
			return false
		}
	}
	return true
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
