package hfcmd

import (
	"archive/tar"
	"io/fs"
	"os"
	"path/filepath"
	"testing"

	"github.com/klauspost/compress/zstd"

	"github.com/shibukawa/hwnpu/internal/backend/coreai"
	shared "github.com/shibukawa/hwnpu/internal/store"
)

// The digest of a Core AI export's programs read from its pack is coreai.ProgramsDigest's of
// the export unpacked: the programs of the export, not those in prebuilt ones or in programs.
func TestProgramsDigestOfPack(t *testing.T) {
	export := filepath.Join(t.TempDir(), "x.coreai")
	for p, b := range map[string]string{
		"decoder.aimodel/main.mlirb":                 "code",
		"decoder.aimodel/main.hash":                  "h1",
		"enc/encoder_0.aimodel/main.mlirb":           "code2",
		"enc/encoder_0.aimodel/main.hash":            "h2",
		"enc-b.aimodel/main.mlirb":                   "code3",
		"enc-b.aimodel/main.hash":                    "h3",
		"empty.aimodel/main.mlirb":                   "",
		"empty.aimodel/main.hash":                    "h4",
		"decoder.h14g.aimodelc/x.aimodel/main.hash":  "h5",
		"decoder.h14g.aimodelc/x.aimodel/main.mlirb": "code5",
		"tables.safetensors":                         "t",
	} {
		f := filepath.Join(export, filepath.FromSlash(p))
		os.MkdirAll(filepath.Dir(f), 0o755)
		os.WriteFile(f, []byte(b), 0o644)
	}
	want := coreai.ProgramsDigest(export)
	if want == "" {
		t.Fatal("no programs")
	}
	stageDir := t.TempDir()
	f := found{want: shared.Entry{Path: "converted/m/coreai/x.coreai/v1"}, got: shared.Entry{Pack: &shared.File{Path: shared.PackFile}}}
	out := filepath.Join(stageDir, "coreai/x.coreai/v1")
	os.MkdirAll(out, 0o755)
	pf, _ := os.Create(filepath.Join(out, shared.PackFile))
	zw, _ := zstd.NewWriter(pf)
	tw := tar.NewWriter(zw)
	filepath.WalkDir(export, func(p string, d fs.DirEntry, err error) error {
		if d.IsDir() {
			return nil
		}
		rel, _ := filepath.Rel(export, p)
		b, _ := os.ReadFile(p)
		tw.WriteHeader(&tar.Header{Typeflag: tar.TypeReg, Name: filepath.ToSlash(rel), Mode: 0o644, Size: int64(len(b))})
		tw.Write(b)
		return nil
	})
	tw.Close()
	zw.Close()
	pf.Close()
	got, err := programsDigest(f, stageDir)
	if err != nil || got != want {
		t.Errorf("%s %v, want %s", got, err, want)
	}
}
