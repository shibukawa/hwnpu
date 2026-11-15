package envfile

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadFile(t *testing.T) {
	p := filepath.Join(t.TempDir(), ".env")
	os.WriteFile(p, []byte(`# settings
HWNPU_MODEL_SOURCE=file:///mnt/llm/hwnpu?writeback=true  # the NAS
export HWNPU_MODEL_SOURCE_1='https://me:p#ss@nas/dav'
HF_X="a \"b\"\nc"
HWNPU_SET=from-file
PATH=/nothing

`), 0o644)
	t.Setenv("HWNPU_SET", "from-env")
	for _, k := range []string{"HWNPU_MODEL_SOURCE", "HWNPU_MODEL_SOURCE_1", "HF_X"} {
		t.Setenv(k, "")
		os.Unsetenv(k)
	}
	path := os.Getenv("PATH")
	f, ok := LoadFile(p)
	if !ok || f.Err != nil || len(f.Set) != 3 {
		t.Fatalf("%+v", f)
	}
	for k, want := range map[string]string{
		"HWNPU_MODEL_SOURCE":   "file:///mnt/llm/hwnpu?writeback=true",
		"HWNPU_MODEL_SOURCE_1": "https://me:p#ss@nas/dav",
		"HF_X":                 "a \"b\"\nc",
		"HWNPU_SET":            "from-env",
		"PATH":                 path,
	} {
		if got := os.Getenv(k); got != want {
			t.Errorf("%s=%q, want %q", k, got, want)
		}
	}
	if _, ok := LoadFile(filepath.Join(t.TempDir(), "none")); ok {
		t.Error("a file that is not there")
	}
	os.WriteFile(p, []byte("HWNPU_BAD\n"), 0o644)
	if f, _ := LoadFile(p); f.Err == nil {
		t.Error("a line without =")
	}
}
