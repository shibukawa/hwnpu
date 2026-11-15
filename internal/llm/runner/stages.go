package runner

import (
	"context"
	"fmt"
	"time"
)

// StageBudget is how much of one stage's outputs Stages lets the host hold: the float32 values of
// every picture that goes through the stage together. A VAE's last stages give 0.5 to 1.2 GB a
// picture at 1024 x 1024 (128 to 288 channels); past about this much, pictures decoded together
// take as long as one after another, or longer (Sana's two together 2.4 s of 2.7, Qwen-Image's two
// 7.3 s for 6.5; design.md 16.216).
const StageBudget = 1280 << 20

// StageRun runs stage i's program p on one picture's input x.
type StageRun func(i int, p Program, x []float32) ([]float32, error)

// Stages runs the programs names (a VAE's stages), each on the output of the one before, over
// the inputs xs, one of each picture: the last stage's outputs. A stage is loaded once and runs
// every picture before the next stage loads (with ahead the next one loads meanwhile, as InTurn),
// so that only its first run follows a load: a stage's first call is its slow one. Where a
// stage's outputs for all the pictures would pass budget (StageBudget), as many pictures as fit
// go on to the end, and the others then start again from that stage: the narrow stages run every
// picture together, the wide ones as many as the host holds.
//
// told, when not nil, is told each time pictures are through a stage, in d: i is the last of the
// total (len(xs) x len(names)) stages of pictures done so far, so that one picture is told of its
// stage i of total as that ends.
func Stages(ctx context.Context, names []string, ahead bool, open OpenFunc, xs [][]float32, budget int, run StageRun,
	told func(i, total int, d time.Duration)) ([][]float32, error) {
	s := &stages{ctx: ctx, names: names, ring: NewRing(names, ahead, open), budget: budget, run: run, told: told, total: len(xs) * len(names)}
	defer s.ring.Close()
	return s.from(0, xs)
}

type stages struct {
	ctx         context.Context
	names       []string
	ring        *Ring
	budget      int
	run         StageRun
	told        func(i, total int, d time.Duration)
	done, total int
}

// from runs the stages first.. on xs.
func (s *stages) from(first int, xs [][]float32) ([][]float32, error) {
	for i := first; i < len(s.names) && len(xs) > 0; i++ {
		t0 := time.Now()
		p, err := s.ring.take(s.ctx, i, i+1 < len(s.names))
		if err != nil {
			return nil, err
		}
		n := len(xs) // the pictures this stage runs together
		ys := make([][]float32, 0, n)
		for k := 0; k < n; k++ {
			y, err := s.run(i, p, xs[k])
			if err != nil {
				p.Close()
				return nil, fmt.Errorf("%s: %w", s.names[i], err)
			}
			ys = append(ys, y)
			if b := 4 * len(y); k == 0 && n*b > s.budget {
				n = max(1, s.budget/b)
			}
		}
		p.Close()
		s.done += n
		if s.told != nil {
			s.told(s.done-1, s.total, time.Since(t0))
		}
		if n < len(xs) {
			head, err := s.from(i+1, ys)
			if err != nil {
				return nil, err
			}
			tail, err := s.from(i, xs[n:])
			if err != nil {
				return nil, err
			}
			return append(head, tail...), nil
		}
		xs = ys
	}
	return xs, nil
}
