package main

import (
	"cmp"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu/internal/chat"
)

// modelsDir is --dir, or where the models are kept by default (as hwnpu-chat's --dir).
func modelsDir(dir string) string { return cmp.Or(dir, chat.ModelsDir()) }

// cacheDir is --cache, or hwnpu in the user cache directory.
func cacheDir(dir string) string {
	if dir != "" {
		return dir
	}
	if d, err := os.UserCacheDir(); err == nil {
		return filepath.Join(d, "hwnpu")
	}
	return ""
}

// placement is how a log line says where a model runs.
func placement(cpu, onNPU bool) string {
	switch {
	case cpu:
		return "on the CPU interpreter"
	case onNPU:
		return "all on the NPU"
	}
	return "NOT all on the NPU"
}

// resolveKind is the model of the definitions --model means for kind (with --models' files read
// first); when there is none, what there is is listed before the error.
func resolveKind(name string, kind chat.Kind, models string, lw io.Writer) (chat.Model, error) {
	if err := chat.LoadUserModels(models, lw); err != nil {
		return chat.Model{}, err
	}
	m, err := chat.ResolveFor(name, kind, lw)
	if e := (*chat.NoModelError)(nil); errors.As(err, &e) {
		err = fmt.Errorf("no %s model, family or use %q (hwnpu-audio list %s shows them)", kind, name, kind)
		fmt.Fprintf(os.Stderr, "no %s model %q; the %s models of the definitions:\n\n", kind, name, kind)
		chat.List(os.Stderr, chat.ModelsDir(), kind)
		fmt.Fprintln(os.Stderr)
	}
	return m, err
}
