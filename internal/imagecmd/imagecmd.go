// Package imagecmd is the engine's side of what hwnpu-image asks hwnpud that is not a model
// (internal/imagewire.Host, served by internal/aihost): the list of the image models and what
// each runs from, why a name picks its model, an export's family, an export the definitions pin,
// and Anima's conversion. It runs in hwnpud's engine, with hwnpud's definitions and cache.
package imagecmd

import (
	"cmp"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/imagegen"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/imagedev"
	"github.com/shibukawa/hwnpu/internal/imagegen/anima"
	"github.com/shibukawa/hwnpu/internal/imagewire"
	"github.com/shibukawa/hwnpu/internal/llm"

	// imagegen.Export reads the export with the picture models' engine
	_ "github.com/shibukawa/hwnpu/ai/imagegen/engine"
)

// definitions reads the definitions as a client's zero ai.Store does (the built-in, the
// machine's and the user's), which the client's Catalog asked for before.
func definitions() error {
	_, err := ai.HostCatalog(ai.StoreSpec{})
	return err
}

// List writes chat's list of the models of kinds with, in place of the download, what each runs
// from on a machine whose NPU is vendor's: the CPU where models.yaml's device (or HWNPU_DEVICE)
// puts it for this machine's NPU, else the AMD export (~/models/NAME.amd-npu2) or "-".
func List(w io.Writer, vendor string, kinds ...string) error {
	if err := definitions(); err != nil {
		return err
	}
	var ks []chat.Kind
	for _, k := range kinds {
		ks = append(ks, chat.Kind(k))
	}
	chat.ListHave(w, "RUNS FROM", func(m chat.Model) string {
		if d, _, err := chat.ChooseDevice(m, vendor, ""); err == nil && d == "cpu" {
			return "CPU"
		}
		if (m.Kind == chat.KindDetect || m.Kind == chat.KindPose || m.Kind == chat.KindSegment) && vendor == "amd" { // an export of iron_onnxcnn.py or iron_efficientsam.py, made on the first run
			if d := chat.AMDPictureExportHere(m); d != "" {
				return shortHome(d)
			}
			return "exported on first run (IRON)"
		}
		if m.Kind == chat.KindSegment { // the encoder: converted on Apple's and Intel's, else the CPU
			if imagedev.ProgramDevice(m, vendor, "") {
				return "converted on first run"
			}
			return "CPU"
		}
		if m.Kind == chat.KindPose || m.Kind == chat.KindDetect || m.Kind == chat.KindOCR { // converted on the first run, for Core AI or OpenVINO
			if d, _, err := chat.ConvertedDevice(m, vendor, ""); err == nil && d == "cpu" {
				return "CPU"
			}
			return "converted on first run"
		}
		if d := chat.AMDPictureExportHere(m); d != "" {
			return shortHome(d)
		}
		if m.Kind == chat.KindImage && (strings.HasPrefix(m.Family, "flux2-klein") || strings.HasPrefix(m.Family, "flux3-klein")) && vendor == "amd" { // flux2klein.PrepareAMD, flux3klein.PrepareAMD
			return "exported on first run (IRON)"
		}
		if m.Kind == chat.KindLabel && strings.EqualFold(vendor, hwnpu.VendorApple.String()) { // siglip.Prepare
			return "converted on first run"
		}
		if m.Kind == chat.KindImage && convertedImage(m.Family, vendor) {
			return "converted on first run"
		}
		return "-"
	}, ks...)
	return nil
}

// convertedImage reports whether the image family converts for this machine's NPU (vendor, as
// hwnpu.Vendor's String) on its first run: Anima on Apple's (Core AI) and Intel's, FLUX.2 and
// FLUX.3 klein on Apple's.
func convertedImage(family, vendor string) bool {
	switch v := strings.ToLower(vendor); {
	case strings.HasPrefix(family, "anima"):
		return v == strings.ToLower(hwnpu.VendorApple.String()) || v == strings.ToLower(hwnpu.VendorIntel.String())
	case strings.HasPrefix(family, "flux2-klein"), strings.HasPrefix(family, "flux3-klein"):
		return v == strings.ToLower(hwnpu.VendorApple.String())
	}
	return false
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

// Picked writes why name, a family or a use, picks its model on this machine, as the commands
// say it (chat.LookupOn); a name the definitions refuse or do not have writes what it writes.
func Picked(name string, log io.Writer) error {
	if err := definitions(); err != nil {
		return err
	}
	chat.LookupOn(name, 0, log)
	return nil
}

// Export is the model an export's directory runs as (imagegen.Export).
func Export(dir string) (ai.Definition, error) { return imagegen.Export(dir) }

// Pinned is the directory of the export the definitions pin by name for AMD's NPU, here or
// fetched from the sources; "" for none.
func Pinned(ctx context.Context, name string, log io.Writer) (string, error) {
	if err := definitions(); err != nil {
		return "", err
	}
	m, ok := chat.PinnedAMDModel(name)
	if !ok {
		return "", nil
	}
	return chat.AMDNamed(ctx, m, []string{name}, func(d string) bool { _, err := os.Stat(d); return err == nil }, log), nil
}

// Convert converts an Anima checkpoint into an export, as hwnpu-image convert does.
func Convert(c imagewire.Convert, log io.Writer) error {
	w, h, err := anima.ParseAspect(c.Size)
	if err != nil {
		return err
	}
	o := anima.ConvertOptions{Backend: cmp.Or(c.Backend, llm.BackendCoreAI), Width: w, Height: h, Dit: c.Dit,
		TextFormat: c.TextFormat, DitFormat: c.DitFormat, Group: c.Group, QChunk: c.QChunk}
	text, dit := anima.Formats(o.Backend)
	o.TextFormat, o.DitFormat = cmp.Or(o.TextFormat, text), cmp.Or(o.DitFormat, dit)
	if !filepath.IsAbs(c.Dir) || !filepath.IsAbs(c.Out) {
		return fmt.Errorf("imagecmd: the checkpoint %q and the export %q must be absolute paths", c.Dir, c.Out)
	}
	return anima.Convert(anima.LayoutCheckpoint(c.Dir), c.Out, o, log)
}
