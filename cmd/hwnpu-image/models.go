package main

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/internal/catalog"
)

// acceptLicenses is --accept-license.
var acceptLicenses bool

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

// ListCmd is "hwnpu-image list": the image, depth, face-detect, pose, detect, ocr and label models of the definitions (models.yaml),
// whether their exports are here, and what each use and family picks for each machine's memory.
type ListCmd struct {
	Kind string `arg:"" optional:"" enum:"image,depth,face-detect,pose,detect,ocr,label," default:"" help:"Only image, depth, face-detect, pose, detect, ocr or label models."`
}

// Run lists the models.
func (c *ListCmd) Run() error {
	// the definitions as ai reads them (its catalog: the built-in, the machine's and the user's)
	cat, err := (ai.Store{}).Catalog()
	if err != nil {
		return err
	}
	if c.Kind != "" {
		listModels(os.Stdout, cat, catalog.Kind(c.Kind))
	} else {
		listModels(os.Stdout, cat, catalog.KindImage, catalog.KindDepth, catalog.KindFace, catalog.KindPose, catalog.KindDetect, catalog.KindOCR,
			catalog.KindLabel, catalog.KindSegment)
	}
	return nil
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
