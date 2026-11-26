package imagegen

import (
	"context"
	"fmt"
	"image"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/internal/airole"
	hcache "github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/imagegen/anima"
	"github.com/shibukawa/hwnpu/internal/imagegen/aspect"
	"github.com/shibukawa/hwnpu/internal/imagegen/flux2klein"
	"github.com/shibukawa/hwnpu/internal/imagegen/flux3klein"
	"github.com/shibukawa/hwnpu/internal/llm"
)

// Export is the model an export's directory runs as (an ai.Definition the app defines and uses
// by its ID): an export of tools/amd/iron_flux2klein.py, iron_flux3klein.py or iron_anima.py
// for AMD's NPU, or of hwnpu-image convert (Anima) and the library's conversions for Apple's
// Neural Engine or Intel's NPU. It is like the family's model of the definitions
// (FLUX.2-klein-4B, flux-3-image-klein, Anima-Turbo), which names
// it and its license; the export's own files say how it draws.
func Export(dir string) (ai.Definition, error) {
	abs, err := filepath.Abs(dir)
	if err != nil {
		return ai.Definition{}, err
	}
	like := map[string]string{famKlein: "FLUX.2-klein-4B", famFlux3: "flux-3-image-klein", famAnima: "Anima-Turbo"}[exportFamily(abs)]
	if like == "" {
		return ai.Definition{}, fmt.Errorf("ai/imagegen: %s is no export of a picture model (FLUX.2 klein, FLUX.3 klein, Anima)", dir)
	}
	return ai.Definition{Like: like, Programs: map[string][]string{ProgramExport: {abs}}}, nil
}

// exportFamily is the family of the export dir, "" for none.
func exportFamily(dir string) string {
	switch {
	case flux3klein.IsExport(dir):
		return famFlux3
	case flux2klein.IsExport(dir):
		return famKlein
	case anima.IsProgramExport(dir) || len(anima.IronSizes(dir)) > 0:
		return famAnima
	}
	return ""
}

// The helpers below are hwnpu-image's (cmd/hwnpu-image/backend.go and anima.go), for the
// devices and the exports the families run on.

// exportDevice is the device backend to open for an export of the program format: Core AI's for
// an export for Apple's Neural Engine (an older Core ML export is lowered to Core AI as it
// loads), else llm.ExportDevice's for the NPU's device.
func exportDevice(format string, n airole.NPU) string {
	if llm.AppleFormat(format) {
		return llm.BackendCoreAI
	}
	return llm.ExportDevice(format, n.Device)
}

// placement is the placement the families' programs are loaded with: strict, so that a program
// off the NPU, or one whose placement is not verified, says why, unless the app allows them.
func placement(e airole.LoadEnv) hwnpu.Option {
	if e.AllowOffNPU {
		return hwnpu.WithPlacement(hwnpu.PlacementPrefer)
	}
	return hwnpu.WithPlacement(hwnpu.PlacementStrict)
}

// openAMD opens AMD's NPU for an export of IRON.
func openAMD(ctx context.Context, opts ...hwnpu.Option) (*hwnpu.Device, error) {
	return hwnpu.Open(ctx, append(opts, hwnpu.WithVendor(hwnpu.VendorAMD))...)
}

// logfOf is a log's printf, a line each.
func logfOf(w io.Writer) func(format string, args ...any) {
	return func(format string, args ...any) {
		if w != nil {
			fmt.Fprintf(w, format+"\n", args...)
		}
	}
}

// aspectsAMD is the export of model m for AMD's NPU that draws w x h, of every aspect ratio by
// the name NAME.amd-npu2 (an older one of 1024 x 1024 alone by that name draws that size only),
// here or in the sources (chat.AMDNamed): made by hand with tool, which the error tells how to
// run for every aspect ratio when there is none. (hwnpu-image's, cmd/hwnpu-image/anima.go.)
func aspectsAMD(ctx context.Context, m chat.Model, w, h int, tool string, sizes func(dir string) [][2]int, log io.Writer) (string, error) {
	names := []string{strings.ToLower(m.Name) + ".amd-npu2"}
	draws := func(d string) bool { return isDir(d) && slices.Contains(sizes(d), [2]int{w, h}) }
	if d := chat.AMDNamed(ctx, m, names, draws, log); d != "" {
		return d, nil
	}
	return "", noAMDExport(m, w, h, tool)
}

// noAMDExport is the error of no export of model m for AMD's NPU that draws w x h: how to make one.
func noAMDExport(m chat.Model, w, h int, tool string) error {
	name := strings.ToLower(m.Name) + ".amd-npu2"
	flags := ""
	if m.File != "" {
		flags = " --dit " + m.File
	}
	var args []string
	for _, a := range aspect.All {
		if a.Width == a.Height {
			args = append(args, fmt.Sprint(a.Width))
		} else {
			args = append(args, fmt.Sprintf("%dx%d", a.Width, a.Height))
		}
	}
	return fmt.Errorf("AMD's NPU runs exports of %s, and neither this machine (%s) nor the sources have one of %s that draws %dx%d: download %s and run\n  python %s ~/models/%s %s --size %s%s\n(one export of all eight aspect ratios; its first run here puts it in the sources written back to; see docs/amd-npu-models.md)",
		tool, strings.Join(hcache.ExportRoots(), ", "), m.Name, w, h, strings.TrimPrefix(m.Source(), "hf:"), tool, filepath.Base(m.Repo),
		shortHome(filepath.Join(chat.AMDRoot(), name)), strings.Join(args, " "), flags)
}

// shortHome is path with the home directory written ~.
func shortHome(path string) string {
	if home, err := os.UserHomeDir(); err == nil {
		if rel, err := filepath.Rel(home, path); err == nil && filepath.IsLocal(rel) {
			return "~/" + filepath.ToSlash(rel)
		}
	}
	return path
}

func isDir(path string) bool {
	fi, err := os.Stat(path)
	return err == nil && fi.IsDir()
}

// points are sizes as points.
func points(sizes [][2]int) []image.Point {
	out := make([]image.Point, len(sizes))
	for i, s := range sizes {
		out[i] = image.Pt(s[0], s[1])
	}
	return out
}

// setting is the model's setting name, def when it has none.
func setting(m chat.Model, name, def string) string {
	if v := m.Settings[name]; v != "" {
		return v
	}
	return def
}
