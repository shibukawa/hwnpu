package probe

import (
	"encoding/json"
	"strings"
	"testing"
)

// fakeNPU holds programs until its capacity, then fails as Level Zero does; leaky does not give
// a closed program's memory back.
type fakeNPU struct {
	capacity, held int
	leaky          bool
	loads          int
}

type fakeProgram struct{ n *fakeNPU }

func (f fakeProgram) run() error { return nil }
func (f fakeProgram) close() {
	if !f.n.leaky {
		f.n.held--
	}
}

func (n *fakeNPU) load(int) (residentProgram, result) {
	n.loads++
	if n.held >= n.capacity {
		return nil, result{Error: "levelzero: pfnGraphInitialize: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY (0x70000002)"}
	}
	n.held++
	return fakeProgram{n}, result{Verified: true, NPUOps: 3}
}

func TestResidentOpen(t *testing.T) {
	n := &fakeNPU{capacity: 5}
	r := residentRun("fp16", 2, "open", 16<<10, true, n.load, func() bool { return false }, nil)
	if r.Held != 5 || !r.OutOfMemory || len(r.Loads) != 6 {
		t.Fatalf("held %d, out of memory %v, loads %d; want 5, true, 6", r.Held, r.OutOfMemory, len(r.Loads))
	}
	if r.Retry == nil || r.Retry.Error != "" {
		t.Fatalf("closing the oldest should make room: %+v", r.Retry)
	}
	if r.AllRun != "ok" || !r.Loads[0].OnNPU {
		t.Errorf("all run %q, on NPU %v", r.AllRun, r.Loads[0].OnNPU)
	}
	if n.held != 0 {
		t.Errorf("%d programs left open", n.held)
	}

	leaky := &fakeNPU{capacity: 3, leaky: true}
	r = residentRun("fp16", 2, "open", 16<<10, true, leaky.load, func() bool { return false }, nil)
	if r.Retry == nil || r.Retry.Error == "" || !r.Retry.OutOfMemory {
		t.Fatalf("a driver that keeps closed memory fails the retry: %+v", r.Retry)
	}
	var sb strings.Builder
	residentMarkdown(&sb, []residentResult{r})
	if md := sb.String(); !strings.Contains(md, "failed: ") || !strings.Contains(md, "out of memory: ") || !strings.Contains(md, "(again, oldest closed)") {
		t.Errorf("report:\n%s", md)
	}
}

func TestResidentClose(t *testing.T) {
	n := &fakeNPU{capacity: 2}
	// 2 layers of fp16 are 0.23 GiB: 1 GiB holds 4
	r := residentRun("fp16", 2, "close", 1<<10, true, n.load, func() bool { return false }, nil)
	if r.Stop != "max" || r.Held != 4 || r.Retry != nil {
		t.Fatalf("stop %q, loaded %d, retry %v; want max, 4, none", r.Stop, r.Held, r.Retry)
	}
	if !strings.Contains(r.summary(), "loaded in turn") {
		t.Errorf("summary %q", r.summary())
	}
}

// TestResidentCrash: a child that dies keeps what it printed as it went.
func TestResidentCrash(t *testing.T) {
	n := &fakeNPU{capacity: 100}
	var lines []string
	emit := func(kind string, v any) {
		if kind == "step" {
			return
		}
		b, _ := json.Marshal(v)
		lines = append(lines, kind+" "+string(b))
	}
	residentRun("fp16", 2, "open", 1<<10, true, n.load, func() bool { return false }, emit)
	r := residentResult{Format: "fp16", Layers: 2, ProgramBytes: 2 * 61 << 20, Hold: "open", Error: "exit status 0xc0000005"}
	r.fromProgress(lines[:3], "exit status 0xc0000005: program 3: running")
	if !r.Crashed || r.Held != 3 || len(r.Loads) != 3 || !strings.Contains(r.summary(), "program 3: running") {
		t.Fatalf("crashed %v, held %d, loads %d, summary %q", r.Crashed, r.Held, len(r.Loads), r.summary())
	}
	var sb strings.Builder
	residentMarkdown(&sb, []residentResult{r})
	if !strings.Contains(sb.String(), "**crashed**") {
		t.Errorf("report:\n%s", sb.String())
	}
}
