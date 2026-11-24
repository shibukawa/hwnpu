package envfile

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestLoadFile(t *testing.T) {
	p := filepath.Join(t.TempDir(), ".env")
	os.WriteFile(p, []byte(`# settings
BONSAIAI_MODEL_SOURCE=file:///mnt/llm/bonsaiai?writeback=true  # the NAS
export BONSAIAI_MODEL_SOURCE_1='https://me:p#ss@nas/dav'
HF_X="a \"b\"\nc"
BONSAIAI_SET=from-file
PATH=/nothing

`), 0o644)
	t.Setenv("BONSAIAI_SET", "from-env")
	for _, k := range []string{"BONSAIAI_MODEL_SOURCE", "BONSAIAI_MODEL_SOURCE_1", "HF_X"} {
		t.Setenv(k, "")
		os.Unsetenv(k)
	}
	path := os.Getenv("PATH")
	f, ok := LoadFile(p)
	if !ok || f.Err != nil || len(f.Set) != 3 {
		t.Fatalf("%+v", f)
	}
	for k, want := range map[string]string{
		"BONSAIAI_MODEL_SOURCE":   "file:///mnt/llm/bonsaiai?writeback=true",
		"BONSAIAI_MODEL_SOURCE_1": "https://me:p#ss@nas/dav",
		"HF_X":                    "a \"b\"\nc",
		"BONSAIAI_SET":            "from-env",
		"PATH":                    path,
	} {
		if got := os.Getenv(k); got != want {
			t.Errorf("%s=%q, want %q", k, got, want)
		}
	}
	if _, ok := LoadFile(filepath.Join(t.TempDir(), "none")); ok {
		t.Error("a file that is not there")
	}
	os.WriteFile(p, []byte("BONSAIAI_BAD\n"), 0o644)
	if f, _ := LoadFile(p); f.Err == nil {
		t.Error("a line without =")
	}
}

func TestEnviron(t *testing.T) {
	user, machine := t.TempDir(), t.TempDir()
	os.WriteFile(filepath.Join(user, ".env"), []byte("BONSAIAI_CACHE=/user\nBONSAIAI_MODEL_SOURCE=file:///nas\nOTHER=x\n"), 0o644)
	os.WriteFile(filepath.Join(machine, ".env"), []byte("BONSAIAI_CACHE=/machine\nBONSAIAI_DEVICE=npu\n"), 0o644)
	env, files, err := environIn([]string{"PATH=/bin", "BONSAIAI_MODEL_SOURCE=file:///shell"}, []string{user, machine})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"PATH=/bin", "BONSAIAI_MODEL_SOURCE=file:///shell", "BONSAIAI_CACHE=/user", "BONSAIAI_DEVICE=npu"}
	if !slices.Equal(env, want) {
		t.Errorf("env = %q, want %q", env, want)
	}
	if len(files) != 2 {
		t.Errorf("files = %q", files)
	}
}
