package llm

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/shibukawa/hwnpu/internal/cache"
)

// WeightFile is the file of a program's weights: weight.bin in a package, the .bin next to an IR's
// .xml.
func WeightFile(program string) string {
	if strings.EqualFold(filepath.Ext(program), ".xml") {
		return strings.TrimSuffix(program, filepath.Ext(program)) + ".bin"
	}
	return filepath.Join(program, "Data", "com.apple.CoreML", "weights", "weight.bin")
}

// weightFiles lists the weight files of the programs in dir (packages and IR).
func weightFiles(dir string) []string {
	pkgs, _ := filepath.Glob(filepath.Join(dir, "*.mlpackage", "Data", "com.apple.CoreML", "weights", "weight.bin"))
	irs, _ := filepath.Glob(filepath.Join(dir, "*.bin"))
	var out []string
	for _, f := range irs {
		if _, err := os.Stat(strings.TrimSuffix(f, ".bin") + ".xml"); err == nil {
			out = append(out, f)
		}
	}
	return append(pkgs, out...)
}

// shareWeights makes the weights of the program at path a clone of the same bytes of another
// program in its directory, if there is one (cache.Clone): the programs of a model that differ only
// in their tokens per call (hwnpu-chat's 8 and 64) have the same weights, which would otherwise
// take the disk twice. Best effort.
func shareWeights(path string) {
	w := WeightFile(path)
	info, err := os.Stat(w)
	if err != nil {
		return
	}
	for _, f := range weightFiles(filepath.Dir(path)) {
		if f == w {
			continue
		}
		if fi, err := os.Stat(f); err != nil || fi.Size() != info.Size() {
			continue
		}
		if ok, _ := cache.Clone(w, f); ok {
			return
		}
	}
}

// ShareWeights makes the programs in dir whose weights have the same bytes share them on the disk
// (cache.Clone), as conversions do as they write (shareWeights), for programs written before they
// did. It reads the files of each size that more than one has.
func ShareWeights(dir string) {
	bySize := map[int64][]string{}
	for _, f := range weightFiles(dir) {
		if fi, err := os.Stat(f); err == nil {
			bySize[fi.Size()] = append(bySize[fi.Size()], f)
		}
	}
	for _, fs := range bySize {
		// each file joins the first earlier one with its bytes
		var firsts []string
		for _, f := range fs {
			shared := false
			for _, r := range firsts {
				if ok, _ := cache.Clone(f, r); ok {
					shared = true
					break
				}
			}
			if !shared {
				firsts = append(firsts, f)
			}
		}
	}
}
