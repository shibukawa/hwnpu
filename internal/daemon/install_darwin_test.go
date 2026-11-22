package daemon

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestPlist(t *testing.T) {
	p := filepath.Join(t.TempDir(), "a.plist")
	exe := "/Users/me/go/bin/hwnpud & co"
	if err := os.WriteFile(p, Plist(exe, "/Users/me/Library/Caches/hwnpu/hwnpud.sock", "/Users/me/Library/Logs/hwnpud.log"), 0o644); err != nil {
		t.Fatal(err)
	}
	if b, err := exec.Command("plutil", "-lint", p).CombinedOutput(); err != nil {
		t.Fatalf("plutil: %v: %s", err, b)
	}
	b, err := exec.Command("plutil", "-convert", "json", "-o", "-", p).Output()
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"Label":"` + Label + `"`, `"SockPathMode":384`, `"` + SocketName + `":{`, `hwnpud & co`, `"serve"`} {
		if !strings.Contains(string(b), want) {
			t.Errorf("plist has no %s: %s", want, b)
		}
	}
}
