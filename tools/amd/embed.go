// Package amdtools holds the IRON exporters of tools/amd (iron_export_lm.py and the modules and
// kernels it reads), so that a hwnpu built with go install can export a model for AMD's NPU on
// its first run without a checkout of hwnpu (internal/chat.PrepareAMD).
package amdtools

import (
	"crypto/sha256"
	"embed"
	"encoding/hex"
	"io/fs"
	"os"
	"path/filepath"
)

//go:embed *.py kernels
var files embed.FS

// Extract writes the exporters into a directory of dir named by their hash (once: a directory
// there is complete, as it is renamed into place after writing) and returns it.
func Extract(dir string) (string, error) {
	h := sha256.New()
	err := fs.WalkDir(files, ".", func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		b, err := files.ReadFile(p)
		h.Write([]byte(p + "\x00"))
		h.Write(b)
		return err
	})
	if err != nil {
		return "", err
	}
	out := filepath.Join(dir, "amd-tools-"+hex.EncodeToString(h.Sum(nil))[:12])
	if _, err := os.Stat(out); err == nil {
		return out, nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	tmp, err := os.MkdirTemp(dir, ".amd-tools-")
	if err != nil {
		return "", err
	}
	err = fs.WalkDir(files, ".", func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		dst := filepath.Join(tmp, filepath.FromSlash(p))
		if d.IsDir() {
			return os.MkdirAll(dst, 0o755)
		}
		b, err := files.ReadFile(p)
		if err != nil {
			return err
		}
		return os.WriteFile(dst, b, 0o644)
	})
	if err == nil {
		err = os.Rename(tmp, out)
	}
	if err != nil {
		os.RemoveAll(tmp)
		if _, e := os.Stat(out); e == nil { // another run put it there first
			return out, nil
		}
		return "", err
	}
	return out, nil
}
