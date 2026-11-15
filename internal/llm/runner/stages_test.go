package runner

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
)

// stageLog loads fake programs for Stages and writes down what happens: "+name" a load, "-name" a
// close, "name(v)" a run on an input that starts with v.
type stageLog struct {
	mu  sync.Mutex // a load ahead runs beside a stage
	log []string
}

func (l *stageLog) add(s string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.log = append(l.log, s)
}

func (l *stageLog) open(_ context.Context, name string) (Program, error) {
	l.add("+" + name)
	return &stageProgram{l: l, name: name}, nil
}

type stageProgram struct {
	Program
	l    *stageLog
	name string
}

func (p *stageProgram) Close() error { p.l.add("-" + p.name); return nil }

// Narrow stages run every picture together; from the first stage whose outputs for all of them
// pass the budget, as many as fit go on to the end and the others start again from that stage.
// Each picture comes out as it does alone.
func TestStages(t *testing.T) {
	ctx := context.Background()
	names := []string{"a", "b", "c"}
	// a stage gives size values of its input's first plus 1: 1 value (4 bytes) from a, 4 (16
	// bytes) from b, 2 from c
	size := map[string]int{"a": 1, "b": 4, "c": 2}
	for _, c := range []struct {
		budget int
		want   string
		told   string
	}{
		// all together
		{1 << 20, "+a a(1) a(2) a(3) -a +b b(2) b(3) b(4) -b +c c(3) c(4) c(5) -c", "2/9 5/9 8/9"},
		// b's outputs: two pictures' fit (32 bytes), not three
		{40, "+a a(1) a(2) a(3) -a +b b(2) b(3) -b +c c(3) c(4) -c +b b(4) -b +c c(5) -c", "2/9 4/9 6/9 7/9 8/9"},
		// one picture at a time from b on
		{16, "+a a(1) a(2) a(3) -a +b b(2) -b +c c(3) -c +b b(3) -b +c c(4) -c +b b(4) -b +c c(5) -c", "2/9 3/9 4/9 5/9 6/9 7/9 8/9"},
		// no room at all: one picture at a time from the first stage
		{0, "+a a(1) -a +b b(2) -b +c c(3) -c +a a(2) -a +b b(3) -b +c c(4) -c +a a(3) -a +b b(4) -b +c c(5) -c", "0/9 1/9 2/9 3/9 4/9 5/9 6/9 7/9 8/9"},
	} {
		l := &stageLog{}
		var told []string
		ys, err := Stages(ctx, names, false, l.open, [][]float32{{1}, {2}, {3}}, c.budget, func(i int, p Program, x []float32) ([]float32, error) {
			name := p.(*stageProgram).name
			if name != names[i] {
				t.Errorf("stage %d is given %s", i, name)
			}
			l.add(fmt.Sprintf("%s(%g)", name, x[0]))
			y := make([]float32, size[name])
			for j := range y {
				y[j] = x[0] + 1
			}
			return y, nil
		}, func(done, total int, _ time.Duration) { told = append(told, fmt.Sprintf("%d/%d", done, total)) })
		if err != nil {
			t.Fatal(err)
		}
		if got := strings.Join(l.log, " "); got != c.want {
			t.Errorf("budget %d: ran\n%s\nwant\n%s", c.budget, got, c.want)
		}
		if got := strings.Join(told, " "); got != c.told {
			t.Errorf("budget %d: told %s, want %s", c.budget, got, c.told)
		}
		if len(ys) != 3 || !slices.Equal(ys[0], []float32{4, 4}) || !slices.Equal(ys[1], []float32{5, 5}) || !slices.Equal(ys[2], []float32{6, 6}) {
			t.Errorf("budget %d: outputs %v", c.budget, ys)
		}
	}
}

// A run that fails ends the stages with every program closed, and so does a load that fails.
func TestStagesErrors(t *testing.T) {
	ctx := context.Background()
	fail := errors.New("fails")
	for _, ahead := range []bool{false, true} {
		l := &stageLog{}
		_, err := Stages(ctx, []string{"a", "b", "c"}, ahead, l.open, [][]float32{{1}, {2}}, 1<<20, func(i int, p Program, x []float32) ([]float32, error) {
			if i == 1 && x[0] == 2 {
				return nil, fail
			}
			return x, nil
		}, nil)
		if !errors.Is(err, fail) || !strings.Contains(err.Error(), "b: ") {
			t.Errorf("ahead %v: err %v", ahead, err)
		}
		open := map[string]int{}
		for _, e := range l.log {
			if e[0] == '+' {
				open[e[1:]]++
			} else {
				open[e[1:]]--
			}
		}
		for name, n := range open {
			if n != 0 {
				t.Errorf("ahead %v: %s left open (%v)", ahead, name, l.log)
			}
		}
		_, err = Stages(ctx, []string{"a", "b"}, ahead, func(ctx context.Context, name string) (Program, error) {
			if name == "b" {
				return nil, fail
			}
			return l.open(ctx, name)
		}, [][]float32{{1}}, 1<<20, func(i int, p Program, x []float32) ([]float32, error) { return x, nil }, nil)
		if !errors.Is(err, fail) {
			t.Errorf("ahead %v: a load that fails: %v", ahead, err)
		}
	}
	if ys, err := Stages(ctx, []string{"a"}, false, nil, nil, 0, nil, nil); err != nil || len(ys) != 0 {
		t.Errorf("no pictures: %v, %v", ys, err)
	}
}
