package cache

import (
	"encoding/hex"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// Core AI keeps what it compiled of an export (.aimodel) in a cache of its own, outside the root:
// ~/Library/Caches/coreai-cache/BUILD/EXECUTABLE/HASH, HASH the export's main.hash, one per OS
// build and per name of the executable that loaded it. An entry holds the program's weights
// again, about as many bytes as the export. Removing a unit removes the entries of the exports in
// it too (CoreAIEntries): only the export's main.hash finds them, and the disk is what removing
// is for. The export downloaded again is compiled again (minutes on the Neural Engine compiler,
// seconds from a prebuilt program).

// coreAICache is Core AI's cache of every OS build, "" off macOS and in tests (a test
// that wants one replaces it): a test's unit never removes the machine's entries.
var coreAICache = func() string {
	if runtime.GOOS != "darwin" || testing.Testing() {
		return ""
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, "Library", "Caches", "coreai-cache")
}

// CoreAIEntries are the entries of Core AI's cache (of every OS build and executable) of the
// exports in x, which removing x removes too: none of a main.hash an export left in the root has
// too, none this process uses (Use), none of a unit of prebuilt programs (their exports keep
// them).
func CoreAIEntries(x Unit) []string {
	base := coreAICache()
	if base == "" || x.Kind == KindPrebuilt {
		return nil
	}
	mine := map[string]bool{}
	for _, p := range x.Paths {
		aimodelHashes(p, nil, mine)
	}
	if len(mine) == 0 {
		return nil
	}
	skip := map[string]bool{}
	for _, p := range x.Paths {
		skip[p] = true
	}
	others := map[string]bool{}
	aimodelHashes(Root(), skip, others)
	var out []string
	for h := range mine {
		if others[h] {
			continue
		}
		found, _ := filepath.Glob(filepath.Join(base, "*", "*", h))
		for _, e := range found {
			usedMu.Lock()
			here := usedHere[e]
			usedMu.Unlock()
			if fi, err := os.Stat(e); err == nil && fi.IsDir() && !here {
				out = append(out, e)
			}
		}
	}
	return out
}

// CoreAIBytes are the bytes of the entries (CoreAIEntries).
func CoreAIBytes(entries []string) int64 {
	var n int64
	seen := map[fileKey]bool{}
	for _, e := range entries {
		b, _, _ := tree(e, seen)
		n += b
	}
	return n
}

// aimodelHashes adds to hashes the main.hash (in hex) of each export (.aimodel) under dir, not
// looking into the paths of skip nor into what holds no export (repositories, IRON's builds,
// Core ML's packages, prebuilt programs).
func aimodelHashes(dir string, skip map[string]bool, hashes map[string]bool) {
	filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil || !d.IsDir() {
			return nil
		}
		if skip[p] {
			return filepath.SkipDir
		}
		name := d.Name()
		if p == filepath.Join(Root(), "original") || p == filepath.Join(Root(), "iron") {
			return filepath.SkipDir
		}
		switch {
		case strings.HasSuffix(name, ".aimodel"):
			if b, err := os.ReadFile(filepath.Join(p, "main.hash")); err == nil && len(b) > 0 {
				hashes[hex.EncodeToString(b)] = true
			}
			return filepath.SkipDir
		case strings.HasSuffix(name, ".aimodelc"), strings.HasSuffix(name, ".mlpackage"),
			strings.HasSuffix(name, ".mlmodelc"), strings.HasSuffix(name, PrebuiltSuffix):
			return filepath.SkipDir
		}
		return nil
	})
}
