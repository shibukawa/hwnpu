package daemon

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestUnits(t *testing.T) {
	exe := `/home/me/go/bin/hwnpud "a" 100%\x`
	svc := string(ServiceUnit(exe))
	if !strings.Contains(svc, `ExecStart="/home/me/go/bin/hwnpud \"a\" 100%%\\x" serve`) {
		t.Errorf("service:\n%s", svc)
	}
	sock := string(SocketUnit("/run/user/1000/hwnpud.sock"))
	for _, want := range []string{"ListenStream=/run/user/1000/hwnpud.sock\n", "SocketMode=0600\n", "FileDescriptorName=" + SocketName + "\n"} {
		if !strings.Contains(sock, want) {
			t.Errorf("socket has no %q:\n%s", want, sock)
		}
	}
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	if Installed() != "" {
		t.Fatal("installed without a unit")
	}
	os.MkdirAll(filepath.Dir(ServiceFile()), 0o755)
	if err := os.WriteFile(ServiceFile(), []byte(svc), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := Installed(); got != exe {
		t.Errorf("Installed() = %q, want %q", got, exe)
	}
}

func TestUnquoteShell(t *testing.T) {
	for in, want := range map[string]string{
		`plain`:                      "plain",
		`$'https://u:p@h/l?a=1&b=2'`: "https://u:p@h/l?a=1&b=2",
		`$'it\'s\x41\n'`:             "it's" + "A\n",
	} {
		if got := unquoteShell(in); got != want {
			t.Errorf("unquoteShell(%s) = %q, want %q", in, got, want)
		}
	}
}
