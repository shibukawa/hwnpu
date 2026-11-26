package llir

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// testdata/share.mlir.bc and share-flat.mlir.bc are coreai-build 3605.6.3's h13g LLIR of the two
// smallest programs that differ by the cause (internal/llm/qwen35 TestWriteMLPProbe, 1024x256,
// share and share-flat): a sigmoid over (1, 8, 16, 64) then the silu of a Linear's output, and
// the same with the sigmoid over (1, 8, 1024). In the first, the sigmoid's two steps (tasks 6 and
// 7) read 8 sets of coefficients of the kernels they share with the silu's, which are sized for
// the silu's one set.
func TestOverruns(t *testing.T) {
	for _, c := range []struct {
		file  string
		tasks int
		over  []Overrun
	}{
		{"share.mlir.bc", 16, []Overrun{{Task: 6, Reads: 1024, Kernel: 128}, {Task: 7, Reads: 1024, Kernel: 128}}},
		{"share-flat.mlir.bc", 15, nil},
	} {
		b, err := os.ReadFile(filepath.Join("testdata", c.file))
		if err != nil {
			t.Fatal(err)
		}
		tasks, over, err := Overruns(b)
		if err != nil {
			t.Fatalf("%s: %v", c.file, err)
		}
		if tasks != c.tasks {
			t.Errorf("%s: %d tasks, want %d", c.file, tasks, c.tasks)
		}
		if len(over) != len(c.over) {
			t.Fatalf("%s: tasks past their kernel %v, want %v", c.file, over, c.over)
		}
		for i := range over {
			if over[i] != c.over[i] {
				t.Errorf("%s: %v, want %v", c.file, over[i], c.over[i])
			}
		}
	}
	if _, _, err := Overruns([]byte("not bytecode")); err == nil {
		t.Error("what is not bytecode was read")
	}
}

// CheckKernels names the functions of a prebuilt program whose tasks read past their kernel, by
// the names of the LLIR files (FUNCTION_HASH_N_ANE_region_I_J.mlir.bc), and says which files it
// could not read.
func TestCheckKernels(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "x.h13g.aimodelc")
	put := func(fn, from string) {
		t.Helper()
		b, err := os.ReadFile(filepath.Join("testdata", from))
		if err != nil {
			t.Fatal(err)
		}
		region := fn + "_9wemocz7YmEwKSHhsD5JpQ_0_ANE_region_0_0"
		d := filepath.Join(dir, "main-h13g-delegates", "MPSGraph", "mpsExecutable.mpsgraphpackage", "binary_0.llir.bundle", region+".bc", "h13g")
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(d, region+".mlir.bc"), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	put("q128-last", "share-flat.mlir.bc")
	if unread, err := CheckKernels(dir); err != nil || len(unread) != 0 {
		t.Fatalf("a program without such tasks: %v, unread %v", err, unread)
	}
	put("sr_q8", "share.mlir.bc")
	_, err := CheckKernels(dir)
	if !errors.Is(err, ErrKernelOverrun) {
		t.Fatalf("a program with such tasks: %v", err)
	}
	if msg := err.Error(); !strings.Contains(msg, "sr_q8: 2 (task 6 reads 1024 bytes of a kernel of 128)") || strings.Contains(msg, "q128-last") {
		t.Errorf("the error names %q", msg)
	}
	// a file that is not LLIR tells nothing: listed, not an error of its own
	bad := filepath.Join(dir, "other_9wemocz7YmEwKSHhsD5JpQ_0_ANE_region_0_0.mlir.bc")
	if err := os.WriteFile(bad, []byte("not bytecode"), 0o644); err != nil {
		t.Fatal(err)
	}
	unread, err := CheckKernels(dir)
	if !errors.Is(err, ErrKernelOverrun) || len(unread) != 1 || unread[0] != bad {
		t.Errorf("with a file that is not LLIR: %v, unread %v", err, unread)
	}
}
