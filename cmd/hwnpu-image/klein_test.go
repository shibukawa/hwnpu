package main

import (
	"os"
	"path/filepath"
	"testing"
)

// An --image argument: FILE, FILE@half, FILE@full; a file whose own name ends in @half is read as
// it is.
func TestReferenceArg(t *testing.T) {
	dir := t.TempDir()
	plain, odd := filepath.Join(dir, "a.png"), filepath.Join(dir, "b@half")
	for _, f := range []string{plain, odd} {
		if err := os.WriteFile(f, nil, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	for _, c := range []struct {
		arg  string
		file string
		div  int
	}{
		{plain, plain, 1},
		{plain + "@half", plain, 2},
		{plain + "@HALF", plain, 2},
		{plain + "@full", plain, 1},
		{odd, odd, 1},
		{odd + "@half", odd, 2},
	} {
		file, div, err := referenceArg(c.arg)
		if err != nil || file != c.file || div != c.div {
			t.Errorf("%s: %s, %d, %v; want %s, %d", c.arg, file, div, err, c.file, c.div)
		}
	}
	for _, arg := range []string{filepath.Join(dir, "none.png"), plain + "@third", filepath.Join(dir, "none.png@half")} {
		if _, _, err := referenceArg(arg); err == nil {
			t.Errorf("%s: no error", arg)
		}
	}
	if got := referenceNames([]int{2, 1, 2}); got != "1+2@half" {
		t.Errorf("names: %q", got)
	}
}
