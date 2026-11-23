package envfile

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSetUnset(t *testing.T) {
	p := filepath.Join(t.TempDir(), "hwnpu", ".env")
	if err := Set(p, "HWNPU_CACHE", "/a"); err != nil {
		t.Fatal(err)
	}
	if fi, err := os.Stat(p); err != nil || fi.Mode().Perm() != 0o600 {
		t.Fatalf("a new file: %v, %v", fi, err)
	}
	os.WriteFile(p, []byte("# mine\nHWNPU_CACHE=/a\nexport HWNPU_CACHE=/old\nOTHER=1\n"), 0o640)
	os.Chmod(p, 0o640)
	for _, kv := range []Var{{"HWNPU_CACHE", "/b"}, {"HWNPU_MODEL_ALLOWLIST", "+nsfw,-license:non-commercial"}, {"HF_X", `a "b" #c` + "\nd"}} {
		if err := Set(p, kv.Name, kv.Value); err != nil {
			t.Fatal(err)
		}
	}
	b, _ := os.ReadFile(p)
	want := "# mine\nHWNPU_CACHE=/b\nOTHER=1\nHWNPU_MODEL_ALLOWLIST=+nsfw,-license:non-commercial\nHF_X=\"a \\\"b\\\" #c\\nd\"\n"
	if string(b) != want {
		t.Fatalf("file:\n%s\nwant:\n%s", b, want)
	}
	if fi, _ := os.Stat(p); fi.Mode().Perm() != 0o640 {
		t.Errorf("mode %v, want the file's 0640", fi.Mode().Perm())
	}
	if v, set, err := Lookup(p, "HF_X"); !set || err != nil || v != "a \"b\" #c\nd" {
		t.Errorf("HF_X read back as %q, %v, %v", v, set, err)
	}
	if removed, err := Unset(p, "HWNPU_CACHE"); !removed || err != nil {
		t.Fatalf("unset: %v, %v", removed, err)
	}
	if removed, _ := Unset(p, "HWNPU_CACHE"); removed {
		t.Error("unset twice")
	}
	if _, set, _ := Lookup(p, "HWNPU_CACHE"); set {
		t.Error("still set")
	}
	if err := Set(p, "PATH", "/x"); err == nil {
		t.Error("PATH is not hwnpu's")
	}
	if err := Set(p, "HWNPU X", "1"); err == nil {
		t.Error("a name with a space")
	}
}
