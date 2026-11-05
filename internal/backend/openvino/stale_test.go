//go:build unix

package openvino

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestRemoveStale(t *testing.T) {
	cmd := exec.Command("true")
	if err := cmd.Run(); err != nil {
		t.Skip(err)
	}
	dead := cmd.Process.Pid // ended and waited for
	if processAlive(dead) {
		t.Skipf("pid %d is taken again", dead)
	}
	stale, err := os.MkdirTemp(shmDir(), fmt.Sprintf("%s%d-", dirPrefix, dead))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(stale, "buf1"), []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	own, err := makeDir()
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(own)
	removeStale()
	if _, err := os.Stat(stale); !os.IsNotExist(err) {
		os.RemoveAll(stale)
		t.Errorf("%s of ended process %d is still there", stale, dead)
	}
	if _, err := os.Stat(own); err != nil {
		t.Errorf("this process's own directory: %v", err)
	}
}
