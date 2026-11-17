package flux2klein

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/imagegen/aspect"
)

// AMDTool is the IRON exporter of FLUX.2 klein (tools/amd, built into hwnpu).
const AMDTool = "iron_flux2klein.py"

// AMDExportVersion is the version of PrepareAMD's exports (a key of the shared layers:
// chat.PinnedRecipe), raised by hand when what PrepareAMD passes AMDTool changes.
const AMDExportVersion = 1

// AMDSizes are the pictures' sizes (width, height) PrepareAMD exports for: the eight aspect
// ratios of --size (aspect.All), 1024 x 1024 first (laid out as an export of its own). A step
// whose rows are no whole rounds of the attention's 512 runs padded, the text's rows first.
var AMDSizes = aspect.Sizes(16)

// amdSquares are the sizes of the exports of one size each that PrepareAMD made before
// AMDSizes (AMDExportName) and that MergeAMD makes one.
var amdSquares = [][2]int{{1024, 1024}, {512, 512}}

// AMDFiles are the repository's files AMDTool reads: Files and model_index.json (whether the
// checkpoint is distilled).
var AMDFiles = append(slices.Clone(Files), "model_index.json")

// AMDExportName is the name of the export for AMD's NPU of the model named name for pictures of
// size x size: NAME.amd-npu2 at 1024, else NAME-SIZE.amd-npu2, NAME in lower case (the names
// tools/amd's guide gives and the shared layers hold the exports made by hand under).
func AMDExportName(name string, size int) string {
	n := strings.ToLower(name)
	if size != 1024 {
		n += fmt.Sprintf("-%d", size)
	}
	return n + ".amd-npu2"
}

// AMDMergedName is the name of the one export for AMD's NPU of the model named name for pictures
// of every one of sizes (tools/amd/iron_flux2klein.py --size 1024 512, or MergeAMD of the exports
// of each), NAME in lower case: NAME-aspects.amd-npu2 for AMDSizes, else NAME-512+1024.amd-npu2,
// the sizes in increasing order (a square by its side, else WxH).
func AMDMergedName(name string, sizes ...[2]int) string {
	n := strings.ToLower(name)
	if sameSizes(sizes, AMDSizes) {
		return n + "-aspects.amd-npu2"
	}
	s := slices.Clone(sizes)
	slices.SortFunc(s, func(a, b [2]int) int { return a[0]*a[1] - b[0]*b[1] })
	parts := make([]string, len(s))
	for i, x := range s {
		parts[i] = sizeArg(x)
	}
	return n + "-" + strings.Join(parts, "+") + ".amd-npu2"
}

// sameSizes reports whether a and b hold the same sizes.
func sameSizes(a, b [][2]int) bool {
	if len(a) != len(b) {
		return false
	}
	for _, x := range a {
		if !slices.Contains(b, x) {
			return false
		}
	}
	return true
}

// sizeArg is a size as AMDTool's --size takes it: N for N x N, else WxH.
func sizeArg(s [2]int) string {
	if s[0] == s[1] {
		return fmt.Sprint(s[0])
	}
	return fmt.Sprintf("%dx%d", s[0], s[1])
}

// AMDRecipe is what PrepareAMD's export of m at sizes (several: one export of them all, the
// first laid out as an export of its own) is made from (a key of the shared layers):
// chat.PinnedRecipe and the exporter's arguments. nil when the definitions pin none of m's files.
func AMDRecipe(m chat.Model, sizes ...[2]int) map[string]string {
	r := chat.PinnedRecipe(m, AMDTool, AMDExportVersion)
	if r != nil {
		r["args"] = strings.Join(amdArgs(sizes...), " ")
	}
	return r
}

// amdArgs are AMDTool's options for pictures of each of sizes: its defaults otherwise (the
// blocks for every mix of up to 4 references of the picture's size and of half its sides, and
// the VAE's encoders of those sizes).
func amdArgs(sizes ...[2]int) []string {
	args := []string{"--size"}
	for _, s := range sizes {
		args = append(args, sizeArg(s))
	}
	return args
}

// PrepareAMD makes an export of FLUX.2 klein (m) for AMD's NPU that draws pictures of w x h
// ready, as hwnpu-chat makes its models' (chat.PrepareAMD), and returns its directory, in this
// order:
//
//   - the export of every AMDSizes here (AMDMergedName in cache.ExportRoots; one made by hand
//     goes in the sources written back to: chat.ShareMadeByHand)
//   - for 512 x 512 and 1024 x 1024: the export of both here (MergeAMD's), else of the size
//     alone (AMDExportName; NAME-1024.amd-npu2 too at 1024; one made by hand goes in the sources
//     written back to: chat.ShareMadeByHand). When the cache's entries hold the exports of both
//     sizes alone, they are made one (MergeAMD) and removed: 11 GB less
//   - from the shared layers: the export of every AMDSizes (AMDRecipe, or one made by hand under
//     AMDMergedName), else for a square of the
//     size alone (the export a first run made elsewhere, or one made by hand that hwnpu-dev
//     cache import put there under its name)
//   - else it downloads the checkpoint (AMDFiles) and exports every AMDSizes in one with IRON in
//     IRON's environment (chat.IRONEnv), in the cache's entries/amd-xdna2, and puts that in the
//     sources written back to
//
// An export of every AMDSizes, once here, replaces the cache's exports of fewer (removed when
// they may go: mergeHere's terms).
func PrepareAMD(ctx context.Context, m chat.Model, w, h int, log io.Writer) (string, error) {
	size := [2]int{w, h}
	if !slices.Contains(AMDSizes, size) {
		return "", fmt.Errorf("--size %dx%d: FLUX.2 klein's exports for AMD's NPU are of %s (tools/amd/%s --size exports another, run with -m DIR)",
			w, h, aspect.Names(), AMDTool)
	}
	if log == nil {
		log = io.Discard
	}
	draws := func(d string) bool { return drawsSize(d, w, h) }
	mergedName := AMDMergedName(m.Name, AMDSizes...)
	if d := cache.FindExport(mergedName, draws); d != "" {
		chat.ShareMadeByHand(m, d, log) // one made by hand here goes in the sources written back to
		dropSubsets(m, log)
		return d, nil
	}
	square := slices.Contains(amdSquares, size)
	if square {
		if d := cache.FindExport(AMDMergedName(m.Name, amdSquares...), draws); d != "" {
			return d, nil
		}
		if d := findSingle(m.Name, w); d != "" {
			chat.ShareMadeByHand(m, d, log) // one made by hand here goes in the sources written back to
			if merged := mergeHere(ctx, m, amdSquares, d, log); merged != "" {
				return merged, nil
			}
			return d, nil
		}
	}
	merged := filepath.Join(chat.AMDRoot(), mergedName)
	mergedImported := []chat.SharedRecipe{{Kind: chat.ImportKind, Recipe: chat.ImportedRecipe(m, cache.AMDTarget, mergedName)}}
	if ok, err := chat.FetchShared(ctx, chat.ExportKind(AMDTool), AMDRecipe(m, AMDSizes...), mergedImported, merged, log); err != nil {
		return "", err
	} else if ok && draws(merged) {
		dropSubsets(m, log)
		return merged, nil
	}
	if square {
		name := AMDExportName(m.Name, w)
		single := filepath.Join(chat.AMDRoot(), name)
		imported := []chat.SharedRecipe{{Kind: chat.ImportKind, Recipe: chat.ImportedRecipe(m, cache.AMDTarget, name)}}
		if ok, err := chat.FetchShared(ctx, chat.ExportKind(AMDTool), AMDRecipe(m, size), imported, single, log); err != nil {
			return "", err
		} else if ok && isIronExport(single) {
			if d := mergeHere(ctx, m, amdSquares, single, log); d != "" {
				return d, nil
			}
			return single, nil
		}
	}
	err := chat.SharedExportOr(ctx, chat.ExportKind(AMDTool), AMDRecipe(m, AMDSizes...), mergedImported, merged, log, func() error {
		python, err := chat.IRONPython(ctx)
		if err != nil {
			return fmt.Errorf("%s has no export for AMD's NPU here (%s) or in the sources, and making one needs IRON (amd/IRON with mlir-aie and Peano, Python 3.12): %w. Install it as INSTALL_AMD.md says and set %s to its virtual environment or to a script that sets it up, or put an export made elsewhere in a source (hwnpu-dev cache import)",
				m.Name, merged, err, chat.IRONEnv)
		}
		fmt.Fprintf(log, "%s: no export for AMD's NPU here or in the sources; downloading the checkpoint and exporting it with IRON (%s in one, the blocks for up to 4 references too; %s on a Ryzen AI 9 365)\n",
			m.Name, aspect.Names(), amdExportTime)
		want := make([]chat.File, len(AMDFiles))
		for i, f := range AMDFiles {
			want[i] = chat.File{Repo: m.Repo, Name: f}
		}
		if err := chat.FetchFiles(ctx, want, log); err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		args := append([]string{chat.RepoDir(m.Repo), chat.ExportOut}, amdArgs(AMDSizes...)...)
		return chat.RunIRON(ctx, python, m.Name, AMDTool, args, merged, log)
	})
	if err != nil {
		return "", err
	}
	if !draws(merged) {
		return "", fmt.Errorf("flux2klein: %s is no whole export of %s at %dx%d", merged, AMDTool, w, h)
	}
	dropSubsets(m, log)
	return merged, nil
}

// amdExportTime is how long the export of every AMDSizes in one took (IRON's kernel cache empty).
const amdExportTime = "about 30 minutes"

// drawsSize reports whether dir is an IRON export that draws pictures of w x h.
func drawsSize(dir string, w, h int) bool {
	return isIronExport(dir) && slices.Contains(IronSizes(dir), [2]int{w, h})
}

// findSingle is the export of the model named name of size alone here (AMDExportName, and the
// older NAME-1024.amd-npu2 at 1024), "" for none.
func findSingle(name string, size int) string {
	names := []string{AMDExportName(name, size)}
	if size == 1024 {
		names = append(names, strings.TrimSuffix(names[0], ".amd-npu2")+"-1024.amd-npu2")
	}
	for _, n := range names {
		if d := cache.FindExport(n, isIronExport); d != "" {
			return d
		}
	}
	return ""
}

// mergeHere makes the exports of each of sizes alone in the cache's entries (made by PrepareAMD
// or fetched: not those of ~/models) one export (MergeAMD) and removes them, when there is one of
// each that may go: no writeback queued for it, and, but for used (the one this process is to
// run), none used in the last cache.KeepRecent (another process may read it). It returns the
// export of them all, "" when there is none (then the exports stay as they are: a size missing,
// one in use, or the merge failed).
func mergeHere(ctx context.Context, m chat.Model, sizes [][2]int, used string, log io.Writer) string {
	root := chat.AMDRoot()
	dirs := make([]string, len(sizes))
	for i, s := range sizes {
		d := filepath.Join(root, AMDExportName(m.Name, s[0]))
		if !isIronExport(d) || len(IronSizes(d)) != 1 || (d != used && !mayGo(d)) || cache.Queued(d) {
			return ""
		}
		dirs[i] = d
	}
	out := filepath.Join(root, AMDMergedName(m.Name, sizes...))
	unlock, err := cache.Lock(ctx, out+":layers", "the export "+filepath.Base(out), log)
	if err != nil {
		return ""
	}
	defer unlock()
	t := time.Now()
	if err := MergeAMD(out, dirs[0], dirs[1:]...); err != nil {
		fmt.Fprintf(log, "%v; the exports stay as they are\n", err)
		return ""
	}
	os.WriteFile(out+".done", nil, 0o644)
	for _, d := range dirs {
		removeExport(d)
	}
	fmt.Fprintf(log, "%s: the exports of %s made one in %.1f s (the blocks' weights, the text encoder and the encoders they share stored once)\n",
		filepath.Base(out), sizeNames(sizes), time.Since(t).Seconds())
	cache.Touch(out)
	return out
}

// mayGo reports whether the cache's export d may be removed: not used in the last
// cache.KeepRecent (another process may read it) and no writeback queued for it.
func mayGo(d string) bool {
	fi, err := os.Stat(d)
	return err == nil && !cache.Queued(d) && cache.Removable(cache.Unit{Path: d, Used: fi.ModTime()}, time.Now(), cache.KeepRecent)
}

// removeExport removes the cache's export d.
func removeExport(d string) {
	os.Remove(filepath.Join(d, "entry.json")) // a folder without it is not whole
	os.Remove(d + ".done")
	os.RemoveAll(d)
}

// dropSubsets removes the cache's exports of m of fewer sizes than AMDSizes (MergeAMD's of the
// squares, and those of one square each) that may go (mayGo): the export of every AMDSizes
// draws theirs.
func dropSubsets(m chat.Model, log io.Writer) {
	root := chat.AMDRoot()
	names := []string{AMDMergedName(m.Name, amdSquares...)}
	for _, s := range amdSquares {
		names = append(names, AMDExportName(m.Name, s[0]))
	}
	for _, n := range names {
		d := filepath.Join(root, n)
		if isIronExport(d) && mayGo(d) {
			removeExport(d)
			fmt.Fprintf(log, "%s: removed (%s draws its sizes)\n", n, AMDMergedName(m.Name, AMDSizes...))
		}
	}
}

// sizeNames are sizes as "512 x 512 and 1024 x 1024".
func sizeNames(sizes [][2]int) string {
	var s []string
	for _, n := range sizes {
		s = append(s, fmt.Sprintf("%d x %d", n[0], n[1]))
	}
	return strings.Join(s, " and ")
}
