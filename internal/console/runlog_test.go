package console

import (
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/cache"
)

// TestRunLog keeps a run's arguments, the lines of its log (a bar's last state, no colors) and
// its error, and keeps RunsKept runs of a command.
func TestRunLog(t *testing.T) {
	cache.SetRoot(t.TempDir())
	defer cache.SetRoot("")
	t.Setenv("HWNPU_NO_RUN_LOG", "")
	args := os.Args
	defer func() { os.Args = args }()
	os.Args = []string{"/bin/hwnpu-test", "-m", "x", "a prompt"}
	RecordRun()
	w := Log(os.Stderr)
	fmt.Fprint(w, "loading\n")
	fmt.Fprint(w, "\x1b[90mdownloading 10%\rdownloading 100%\x1b[0m\n")
	fmt.Fprint(w, "no new line")
	EndInterrupted(errors.New("it broke\nhere"))
	runs := Runs()
	if len(runs) != 1 {
		t.Fatalf("%d runs", len(runs))
	}
	r := runs[0]
	if strings.Join(r.Args, "|") != "hwnpu-test|-m|x|a prompt" || r.Error != "it broke here" || r.End.IsZero() || !r.Failed() {
		t.Errorf("run %+v", r)
	}
	if strings.Join(r.Lines, "|") != "loading|downloading 100%|no new line" {
		t.Errorf("lines %q", r.Lines)
	}
	for range RunsKept + 2 {
		RecordRun()
		EndInterrupted(nil)
	}
	if n := len(Runs()); n > RunsKept {
		t.Errorf("%d runs kept", n)
	}
}
