package probe

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
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
	r := residentRun(residentCfg{format: "fp16", layers: 2, hold: "open", maxMiB: 16 << 10, run: true, cycle: true}, n.load, func() bool { return false }, nil)
	if r.Held != 5 || !r.OutOfMemory || len(r.Loads) != 6 {
		t.Fatalf("held %d, out of memory %v, loads %d; want 5, true, 6", r.Held, r.OutOfMemory, len(r.Loads))
	}
	if r.Retry == nil || r.Retry.Error != "" {
		t.Fatalf("closing the oldest should make room: %+v", r.Retry)
	}
	if r.AllRun != "ok" || !r.Loads[0].OnNPU {
		t.Errorf("all run %q, on NPU %v", r.AllRun, r.Loads[0].OnNPU)
	}
	if !r.Cycled || r.SlowBytes != 0 || r.FastBytes != 5*r.ProgramBytes {
		t.Errorf("calls in turn that do not slow down: cycled %v, fast %d, slow %d", r.Cycled, r.FastBytes, r.SlowBytes)
	}
	if n.held != 0 {
		t.Errorf("%d programs left open", n.held)
	}

	leaky := &fakeNPU{capacity: 3, leaky: true}
	r = residentRun(residentCfg{format: "fp16", layers: 2, hold: "open", maxMiB: 16 << 10, run: true}, leaky.load, func() bool { return false }, nil)
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
	r := residentRun(residentCfg{format: "fp16", layers: 2, hold: "close", maxMiB: 1 << 10, run: true, cycle: true}, n.load, func() bool { return false }, nil)
	if r.Stop != "max" || r.Held != 4 || r.Retry != nil {
		t.Fatalf("stop %q, loaded %d, retry %v; want max, 4, none", r.Stop, r.Held, r.Retry)
	}
	if r.Cycled || r.Loads[3].CycleMs != 0 {
		t.Errorf("close holds nothing to run in turn")
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
	residentRun(residentCfg{format: "fp16", layers: 2, hold: "open", maxMiB: 1 << 10, run: true}, n.load, func() bool { return false }, emit)
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

// evictingNPU loads everything without an error, as Core ML does, but keeps only the onChip
// programs run last on the NPU: a call to another loads it again, slowly.
type evictingNPU struct {
	onChip int
	recent []int // ids, the last run first
	loads  int
	open   int
}

type evictingProgram struct {
	n  *evictingNPU
	id int
}

func (p evictingProgram) run() error {
	n := p.n
	slow := true
	for i, id := range n.recent {
		if id == p.id {
			n.recent = append(n.recent[:i], n.recent[i+1:]...)
			slow = false
			break
		}
	}
	n.recent = append([]int{p.id}, n.recent...)
	if len(n.recent) > n.onChip {
		n.recent = n.recent[:n.onChip]
	}
	if slow {
		time.Sleep(6 * time.Millisecond)
	} else {
		time.Sleep(time.Millisecond)
	}
	return nil
}

func (p evictingProgram) close() { p.n.open-- }

func (n *evictingNPU) load(int) (residentProgram, result) {
	n.loads++
	n.open++
	return evictingProgram{n, n.loads}, result{Verified: true, NPUOps: 3}
}

// TestResidentCycle: no load fails, but running every held program in turn finds where the NPU
// starts evicting, and stops running in turn after 3 slow steps.
func TestResidentCycle(t *testing.T) {
	n := &evictingNPU{onChip: 4}
	per := int64(2 * 61 << 20 * 2) // what probe.WeightBytes gives 2 layers of fp16, near enough
	wired := func() (int64, bool) { return int64(min(n.open, n.onChip)) * per, true }
	r := residentRun(residentCfg{format: "fp16", layers: 2, hold: "open", maxMiB: 10 * 233, run: true, cycle: true, wired: wired}, n.load, func() bool { return false }, nil)
	if len(r.Loads) != 10 || r.Stop != "max" {
		t.Fatalf("loads %d, stop %q; want 10, max", len(r.Loads), r.Stop)
	}
	if r.FastBytes != 4*r.ProgramBytes || r.SlowBytes != 5*r.ProgramBytes {
		t.Fatalf("fast up to %d programs, slow from %d; want 4 and 5", r.FastBytes/r.ProgramBytes, r.SlowBytes/r.ProgramBytes)
	}
	for i, l := range r.Loads {
		ran := l.CycleMs > 0
		if want := i < 7; ran != want { // slow with 5, 6 and 7 held: no more after
			t.Errorf("program %d: ran in turn %v, want %v", i, ran, want)
		}
		if ran && l.Slow != (i >= 4) {
			t.Errorf("program %d: slow %v at %.1f ms (base %.1f)", i, l.Slow, l.CycleMs, r.BaseMs)
		}
	}
	if !r.NPUWired || r.Loads[2].NPUBytes != 3*per || r.Loads[9].NPUBytes != 4*per {
		t.Errorf("Neural Engine memory: %v, %d, %d", r.NPUWired, r.Loads[2].NPUBytes, r.Loads[9].NPUBytes)
	}
	s := r.summary()
	if !strings.Contains(s, "times slower from") {
		t.Errorf("summary %q", s)
	}
	var sb strings.Builder
	residentMarkdown(&sb, []residentResult{r})
	if md := sb.String(); !strings.Contains(md, "| full speed in turn |") || !strings.Contains(md, "**") || !strings.Contains(md, "Neural Engine |") {
		t.Errorf("report:\n%s", md)
	}
	if n.open != 0 {
		t.Errorf("%d left open", n.open)
	}
}
