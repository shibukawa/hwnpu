package anima

import (
	"context"
	"fmt"
	"io"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm/runner"
	"github.com/shibukawa/hwnpu/internal/types"
)

// fakeNPU loads programs while their sizes fit in room, and fails as Intel's driver does past it.
type fakeNPU struct {
	room   int
	size   map[string]int
	loaded map[string]bool
	loads  []string
}

func (n *fakeNPU) used() int {
	u := 0
	for name := range n.loaded {
		u += n.size[name]
	}
	return u
}

func (n *fakeNPU) open(_ context.Context, name string, _ bool) (*fakeProgram, error) {
	if n.loaded[name] {
		return nil, fmt.Errorf("%s loaded twice", name)
	}
	if n.used()+n.size[name] > n.room {
		return nil, fmt.Errorf("%s: %w", name, &types.DriverError{Backend: "levelzero", Op: "pfnGraphInitialize", Code: 0x70000002, Msg: "ERROR_OUT_OF_HOST_MEMORY"})
	}
	n.loaded[name] = true
	n.loads = append(n.loads, name)
	return &fakeProgram{npu: n, name: name}, nil
}

type fakeProgram struct {
	npu  *fakeNPU
	name string
}

func (p *fakeProgram) Inputs() map[string][]int    { return nil }
func (p *fakeProgram) Output() []int               { return nil }
func (p *fakeProgram) Metadata() map[string]string { return nil }
func (p *fakeProgram) OnNPU() bool                 { return true }
func (p *fakeProgram) Run(context.Context, map[string][]float32) ([]float32, error) {
	if !p.npu.loaded[p.name] {
		return nil, fmt.Errorf("%s runs closed", p.name)
	}
	return []float32{1}, nil
}
func (p *fakeProgram) Close() error { delete(p.npu.loaded, p.name); return nil }

// newFakeEngine is a progEngine of Anima's parts (sizes as fp16 at 512 x 512, in 10 MB) on npu.
func newFakeEngine(npu *fakeNPU, log *strings.Builder) *progEngine {
	var w io.Writer
	if log != nil {
		w = log
	}
	npu.size = map[string]int{"text": 120, "time": 2, "final": 1, "vae": 10}
	npu.loaded = map[string]bool{}
	e := &progEngine{timeRows: 1, log: w}
	e.open = func(ctx context.Context, name string, show bool) (runner.Program, error) {
		p, err := npu.open(ctx, name, show)
		if err != nil {
			return nil, err
		}
		return p, nil
	}
	e.textP = &slot{name: "text", kind: slotText}
	e.time = &slot{name: "time", kind: slotSmall}
	e.fin = &slot{name: "final", kind: slotSmall}
	for i := 0; i < 7; i++ {
		name := fmt.Sprintf("dit_blocks_%d", 4*i)
		npu.size[name] = 57
		e.blocks = append(e.blocks, &slot{name: name, kind: slotBlock})
	}
	return e
}

func loadAll(t *testing.T, e *progEngine) {
	t.Helper()
	for _, s := range e.slots() {
		if _, err := e.get(context.Background(), s); err != nil {
			t.Fatal(err)
		}
	}
}

func generate(t *testing.T, e *progEngine, steps int) {
	t.Helper()
	ctx := context.Background()
	if _, err := e.text(ctx, nil, nil, 0, 0); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < steps; i++ {
		if _, err := e.velocity(ctx, nil, []float32{0}); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := e.get(ctx, &slot{name: "vae", kind: slotStage}); err != nil {
		t.Fatal(err)
	}
}

func TestProgEngineAllFit(t *testing.T) {
	npu := &fakeNPU{room: 1000}
	var log strings.Builder
	e := newFakeEngine(npu, &log)
	loadAll(t, e)
	generate(t, e, 8)
	if len(npu.loads) != 11 || e.unloaded != 0 {
		t.Fatalf("%d loads, %d unloaded: %v", len(npu.loads), e.unloaded, npu.loads)
	}
}

// The case of the Core Ultra 5 135U: the last group does not fit beside the text encoder.
func TestProgEngineTextMakesRoom(t *testing.T) {
	npu := &fakeNPU{room: 123 + 6*57 + 30}
	var log strings.Builder
	e := newFakeEngine(npu, &log)
	loadAll(t, e)
	if e.textP.p != nil || e.blocks[6].p == nil {
		t.Fatalf("text loaded %v, last group loaded %v", e.textP.p != nil, e.blocks[6].p != nil)
	}
	n := len(npu.loads)
	generate(t, e, 8)
	// the text loads again (closing a group), then that group once: 2 loads for the image
	reloads := npu.loads[n:]
	if len(reloads) != 3 || reloads[0] != "text" { // + the VAE's stage
		t.Fatalf("reloads %v", reloads)
	}
	if !strings.Contains(log.String(), "text unloaded for dit_blocks_24") {
		t.Fatalf("log %q", log.String())
	}
}

// When the groups themselves do not fit (4 places for 7 groups), closing the group used last
// loads 7 - 4 + 1 = 4 groups a step (closing the one used first, LRU, would load all 7).
func TestProgEngineGroupsCycle(t *testing.T) {
	npu := &fakeNPU{room: 3 + 4*57 + 20}
	var log strings.Builder
	e := newFakeEngine(npu, &log)
	loadAll(t, e)
	n := len(npu.loads)
	const steps = 8
	generate(t, e, steps)
	blocks := 0
	for _, name := range npu.loads[n:] {
		if strings.HasPrefix(name, "dit_blocks_") {
			blocks++
		}
	}
	// + the groups the text encoder closed at the start
	if want := 4*steps + 2; blocks > want {
		t.Fatalf("%d groups loaded again in %d steps, want at most %d: %v", blocks, steps, want, npu.loads[n:])
	}
}

func TestProgEngineNothingToClose(t *testing.T) {
	npu := &fakeNPU{room: 50}
	e := newFakeEngine(npu, nil)
	var err error
	for _, s := range e.slots() {
		if _, err = e.get(context.Background(), s); err != nil {
			break
		}
	}
	if err == nil || !types.IsOutOfMemory(err) || !strings.Contains(err.Error(), "no other program") {
		t.Fatalf("err %v", err)
	}
}
