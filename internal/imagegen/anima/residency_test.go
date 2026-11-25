package anima

import (
	"context"
	"fmt"
	"io"
	"maps"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/shibukawa/bonsaiai/internal/llm/runner"
	"github.com/shibukawa/bonsaiai/internal/types"
)

// fakeNPU loads programs while their sizes fit in room, and fails as Intel's driver does past it.
type fakeNPU struct {
	mu     sync.Mutex // a load ahead (swap) runs beside a group
	room   int
	size   map[string]int
	loaded map[string]bool
	loads  []string
	groups int       // the most groups of the DiT's blocks loaded at once
	out    []float32 // what a run gives ({1} when nil)
	fail   string    // the program whose runs fail
}

func (n *fakeNPU) used() int {
	u := 0
	for name := range n.loaded {
		u += n.size[name]
	}
	return u
}

// held are the programs loaded now, sorted.
func (n *fakeNPU) held() []string {
	n.mu.Lock()
	defer n.mu.Unlock()
	return slices.Sorted(maps.Keys(n.loaded))
}

// groupLoads are the loads of the DiT's groups from load from on, in order, and the most groups
// loaded at once since reset.
func (n *fakeNPU) groupLoads(from int) (names []string, atOnce int) {
	n.mu.Lock()
	defer n.mu.Unlock()
	for _, name := range n.loads[from:] {
		if strings.HasPrefix(name, "dit_blocks_") {
			names = append(names, name)
		}
	}
	return names, n.groups
}

func (n *fakeNPU) open(_ context.Context, name string, _ bool) (*fakeProgram, error) {
	n.mu.Lock()
	defer n.mu.Unlock()
	if n.loaded[name] {
		return nil, fmt.Errorf("%s loaded twice", name)
	}
	if n.used()+n.size[name] > n.room {
		return nil, fmt.Errorf("%s: %w", name, &types.DriverError{Backend: "levelzero", Op: "pfnGraphInitialize", Code: 0x70000002, Msg: "ERROR_OUT_OF_HOST_MEMORY"})
	}
	n.loaded[name] = true
	n.loads = append(n.loads, name)
	groups := 0
	for name := range n.loaded {
		if strings.HasPrefix(name, "dit_blocks_") {
			groups++
		}
	}
	n.groups = max(n.groups, groups)
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
	p.npu.mu.Lock()
	defer p.npu.mu.Unlock()
	if !p.npu.loaded[p.name] {
		return nil, fmt.Errorf("%s runs closed", p.name)
	}
	if p.name == p.npu.fail {
		return nil, fmt.Errorf("%s fails", p.name)
	}
	if p.npu.out != nil {
		return p.npu.out, nil
	}
	return []float32{1}, nil
}
func (p *fakeProgram) Close() error {
	p.npu.mu.Lock()
	defer p.npu.mu.Unlock()
	delete(p.npu.loaded, p.name)
	return nil
}

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
		if _, err := e.velocity(ctx, nil, []float32{0}, [][]float32{nil}); err != nil {
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

// On Core AI the DiT's groups load for their turns (swap): one is held, with a load ahead two; the
// text encoder is not held beside them, the group loaded for a step that does not come is not held
// beside the VAE, and a run that fails leaves no group loaded.
func TestProgEngineSwap(t *testing.T) {
	ctx := context.Background()
	for _, ahead := range []bool{false, true} {
		npu := &fakeNPU{room: 1000}
		e := newFakeEngine(npu, nil)
		e.dit.Width, e.dit.Height = 1, 1
		e.vae = []string{"vae"}
		loadAll(t, e) // as openPrograms: each loaded once, the groups then closed
		for _, b := range e.blocks {
			b.p.Close()
			b.p = nil
		}
		e.swapGroups(ahead)
		n := len(npu.loads)
		npu.groups = 0
		if _, err := e.text(ctx, nil, nil, 0, 0); err != nil {
			t.Fatal(err)
		}
		if got := npu.held(); !slices.Equal(got, []string{"final", "time"}) {
			t.Fatalf("ahead %v: after the prompt %v are loaded", ahead, got)
		}
		const steps = 8
		for i := 0; i < steps; i++ {
			if _, err := e.velocity(ctx, nil, []float32{0}, [][]float32{nil, nil}); err != nil {
				t.Fatal(err)
			}
		}
		npu.mu.Lock()
		npu.out = make([]float32, 3)
		npu.mu.Unlock()
		if _, err := e.decode(ctx, nil, nil); err != nil {
			t.Fatal(err)
		}
		if got := npu.held(); !slices.Equal(got, []string{"final", "time"}) {
			t.Errorf("ahead %v: after the picture %v are loaded", ahead, got)
		}
		// the groups in their order each step; with a load ahead the first once more, for the step
		// that did not come
		groups, atOnce := npu.groupLoads(n)
		most := 1
		if ahead { // the next one, when its load started before this one was closed
			most = 2
		}
		if atOnce < 1 || atOnce > most {
			t.Errorf("ahead %v: %d groups loaded at once, want at most %d", ahead, atOnce, most)
		}
		if want := steps * len(e.blocks); len(groups) != want && !(ahead && len(groups) == want+1) {
			t.Errorf("ahead %v: %d loads of groups in %d steps", ahead, len(groups), steps)
		}
		for i, name := range groups {
			if name != e.blocks[i%len(e.blocks)].name {
				t.Fatalf("ahead %v: load %d is %s: %v", ahead, i, name, groups)
			}
		}
		// a second picture after the first, then a group that fails
		if _, err := e.velocity(ctx, nil, []float32{0}, [][]float32{nil}); err != nil {
			t.Fatal(err)
		}
		npu.mu.Lock()
		npu.out, npu.fail = nil, e.blocks[3].name
		npu.mu.Unlock()
		if _, err := e.velocity(ctx, nil, []float32{0}, [][]float32{nil}); err == nil || !strings.Contains(err.Error(), "blocks group 3") {
			t.Fatalf("ahead %v: err %v", ahead, err)
		}
		if err := e.close(); err != nil {
			t.Fatal(err)
		}
		if got := npu.held(); len(got) != 0 {
			t.Errorf("ahead %v: after close %v are loaded", ahead, got)
		}
	}
}
