package main

import (
	"context"
	"errors"
	"fmt"
	"github.com/shibukawa/hwnpu"
	hcache "github.com/shibukawa/hwnpu/internal/cache"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"

	"github.com/shibukawa/hwnpu/internal/chat"
)

// acceptLicenses is --accept-license.
var acceptLicenses bool

// isTerminal reports whether f is a terminal (a character device).
func isTerminal(f *os.File) bool {
	fi, err := f.Stat()
	return err == nil && fi.Mode()&os.ModeCharDevice != 0
}

// expandHome is path with a leading ~ made the home directory.
func expandHome(path string) string {
	if path == "~" || strings.HasPrefix(path, "~/") || strings.HasPrefix(path, "~"+string(filepath.Separator)) {
		home, _ := os.UserHomeDir()
		return filepath.Join(home, path[1:])
	}
	return path
}

// isPath reports whether --model names a directory (an export) rather than a model of the
// definitions: a name with a path separator or a leading . or ~ (./export for one here).
func isPath(name string) bool {
	return strings.ContainsAny(name, `/\`) || strings.HasPrefix(name, ".") || strings.HasPrefix(name, "~")
}

// resolve is the model of the definitions (models.yaml) a name of --model means for kind: its
// name, a family (sana, sana:1.6b: picked for the machine's memory) or a use (image, depth).
// What was picked is logged.
func resolve(name string, kind chat.Kind) (chat.Model, error) {
	if err := chat.LoadUserModels("", os.Stderr); err != nil {
		return chat.Model{}, err
	}
	m, err := chat.ResolveFor(name, kind, os.Stderr)
	if e := (*chat.NoModelError)(nil); errors.As(err, &e) {
		err = fmt.Errorf("no %s model, family or use %q (hwnpu-image list shows them)", kind, name)
	}
	if err == nil {
		// a license that restricts the use: accepted once, before the first download or run
		err = chat.AcceptLicense(m, acceptLicenses, isTerminal(os.Stdin) && isTerminal(os.Stderr), os.Stdin, os.Stderr)
		if err != nil {
			return m, err
		}
	}
	if err != nil {
		// what there is, then the error (kong prints it last)
		fmt.Fprintf(os.Stderr, "no %s model %q; the %s models of the definitions:\n\n", kind, name, kind)
		listModels(os.Stderr, kind)
		fmt.Fprintln(os.Stderr)
	}
	return m, err
}

// ListCmd is "hwnpu-image list": the image, depth, face-detect, pose, detect, ocr and label models of the definitions (models.yaml),
// whether their exports are here, and what each use and family picks for each machine's memory.
type ListCmd struct {
	Kind string `arg:"" optional:"" enum:"image,depth,face-detect,pose,detect,ocr,label," default:"" help:"Only image, depth, face-detect, pose, detect, ocr or label models."`
}

// Run lists the models.
func (c *ListCmd) Run() error {
	if err := chat.LoadUserModels("", os.Stderr); err != nil {
		return err
	}
	if c.Kind != "" {
		listModels(os.Stdout, chat.Kind(c.Kind))
	} else {
		listModels(os.Stdout, chat.KindImage, chat.KindDepth, chat.KindFace, chat.KindPose, chat.KindDetect, chat.KindOCR, chat.KindLabel, chat.KindSegment)
	}
	return nil
}

// listModels writes chat's list of the models of kinds with, in place of the download, what
// each runs from on this machine: the CPU where models.yaml's device (or HWNPU_DEVICE) puts it
// for this machine's NPU, else the AMD export (~/models/NAME.amd-npu2) or "-".
func listModels(w io.Writer, kinds ...chat.Kind) {
	vendor := npuVendor(context.Background())
	chat.ListHave(w, "RUNS FROM", func(m chat.Model) string {
		if d, _, err := chat.ChooseDevice(m, vendor, ""); err == nil && d == "cpu" {
			return "CPU"
		}
		if (m.Kind == chat.KindDetect || m.Kind == chat.KindPose || m.Kind == chat.KindSegment) && vendor == "amd" { // an export of iron_onnxcnn.py or iron_efficientsam.py, made on the first run
			if d, err := amdExport(m, ""); err == nil {
				return shortHome(d)
			}
			return "exported on first run (IRON)"
		}
		if m.Kind == chat.KindSegment { // the encoder: converted on Apple's and Intel's, else the CPU
			if programDevice(m, vendor, "") {
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
		if d, err := amdExport(m, ""); err == nil {
			return shortHome(d)
		}
		if m.Kind == chat.KindLabel && strings.EqualFold(vendor, hwnpu.VendorApple.String()) { // siglip.Prepare
			return "converted on first run"
		}
		if m.Kind == chat.KindImage && convertedImage(m.Family, vendor) {
			return "converted on first run"
		}
		return "-"
	}, kinds...)
}

// convertedImage reports whether the image family converts for this machine's NPU (vendor, as
// hwnpu.Vendor's String) on its first run: Anima and Sana on Apple's (Core AI) and Intel's,
// Qwen-Image and Chroma on Apple's.
func convertedImage(family, vendor string) bool {
	switch v := strings.ToLower(vendor); {
	case strings.HasPrefix(family, "anima"), strings.HasPrefix(family, "sana"):
		return v == strings.ToLower(hwnpu.VendorApple.String()) || v == strings.ToLower(hwnpu.VendorIntel.String())
	case strings.HasPrefix(family, "qwen-image"), strings.HasPrefix(family, "chroma"):
		return v == strings.ToLower(hwnpu.VendorApple.String())
	}
	return false
}

// amdExport is the export of model m on AMD's NPU: NAME.amd-npu2 or NAME-1024.amd-npu2, NAME
// in lower case (what tools/amd/iron_*.py are told to write), in entries/amd-xdna2 under the
// cache's root or in ~/models (hcache.ExportRoots). tool is the script that writes it, for the
// error when there is none.
func amdExport(m chat.Model, tool string) (string, error) {
	root := hcache.LegacyExports()
	name := strings.ToLower(m.Name)
	for _, n := range []string{name, name + "-1024"} {
		if d := hcache.FindExport(n+".amd-npu2", isDir); d != "" {
			return d, nil
		}
	}
	file := ""
	if m.File != "" {
		file = " --dit " + m.File
	}
	return "", fmt.Errorf("AMD's NPU runs exports of %s, and %s has no %s.amd-npu2 for %s: download %s and run\n  python %s ~/models/%s ~/models/%s.amd-npu2%s\n(see docs/amd-npu-models.md)",
		tool, root, name, m.Name, strings.TrimPrefix(m.Source(), "hf:"), tool, filepath.Base(m.Repo), name, file)
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

// freeName is the first of base, "stem (2).ext", "stem (3).ext", ... that does not exist and
// is not one of taken (names given out already for this run).
func freeName(base string, taken ...string) string {
	free := func(p string) bool {
		_, err := os.Lstat(p)
		return errors.Is(err, os.ErrNotExist) && !slices.Contains(taken, p)
	}
	if free(base) {
		return base
	}
	ext := filepath.Ext(base)
	stem := strings.TrimSuffix(base, ext)
	for i := 2; ; i++ {
		p := fmt.Sprintf("%s (%d)%s", stem, i, ext)
		if free(p) {
			return p
		}
	}
}

// openFile shows path in the system's viewer (open on macOS, the file's association on
// Windows, xdg-open elsewhere when there is a display). It does not wait for the viewer.
func openFile(path string) error {
	var c *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		c = exec.Command("open", path)
	case "windows":
		c = exec.Command("rundll32", "url.dll,FileProtocolHandler", path)
	default:
		if os.Getenv("DISPLAY") == "" && os.Getenv("WAYLAND_DISPLAY") == "" {
			return errors.New("no display")
		}
		c = exec.Command("xdg-open", path)
	}
	if err := c.Start(); err != nil {
		return err
	}
	go c.Wait() // xdg-open returns at once; the process may end first
	return nil
}
