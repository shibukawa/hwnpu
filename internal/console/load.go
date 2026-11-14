package console

import (
	"io"
	"math"
	"sync"
	"time"
)

// How long a program takes to load is not known beforehand: one that the NPU's compiler has
// kept loads in milliseconds whatever its size, one that is compiled takes seconds to minutes,
// and the runtime (Core AI, Core ML, Level Zero) tells nothing while it does. What is known is
// how long the programs loaded before took for their size: a load's bar goes by the time since
// it started over the time a program of that size took, an estimate (design.md 16.204).

const (
	// slowLoad is the shortest load that tells how long a compile takes: a shorter one was kept
	// by the compiler's cache, and tells nothing of the next one that is not.
	slowLoad = time.Second
	// loadFloor is the least size a load counts as: the time of a program with few weights (an
	// attention program, a stage of a VAE) goes with its tensors, not its file.
	loadFloor = 1 << 20
)

type loadSample struct {
	bytes int64
	d     time.Duration
}

var loads struct {
	sync.Mutex
	samples []loadSample // the slow loads so far
	bars    []*Bar       // the load bars going on, the last started last
}

// observe keeps how long a load of bytes took, when it was a slow one.
func observe(bytes int64, d time.Duration) {
	if d < slowLoad {
		return
	}
	loads.Lock()
	loads.samples = append(loads.samples, loadSample{max(bytes, loadFloor), d})
	loads.Unlock()
}

// estimate is how long a load of bytes takes when it is compiled, 0 when no slow load tells:
// the time of the slow load nearest in size (the last of them), in proportion to the sizes.
// Programs of one size (the groups of a DiT's blocks, the parts of a language model) take the
// same time; between kinds of programs the time per byte differs severalfold.
func estimate(bytes int64) time.Duration {
	bytes = max(bytes, loadFloor)
	loads.Lock()
	defer loads.Unlock()
	best, dist := -1, 0.0
	for i, s := range loads.samples {
		if d := math.Abs(math.Log(float64(bytes) / float64(s.bytes))); best < 0 || d <= dist {
			best, dist = i, d
		}
	}
	if best < 0 {
		return 0
	}
	s := loads.samples[best]
	return time.Duration(float64(s.d) * float64(bytes) / float64(s.bytes))
}

// pseudo is the share shown of a load after x of its estimated time: the time itself up to 0.8,
// then ever slower toward 1 (0.9 when the estimate is over, 0.97 at twice, 0.99 at five times),
// so that a load that takes longer than the estimate still moves and never shows as done.
func pseudo(x float64) float64 {
	if x <= 0.8 {
		return max(x, 0)
	}
	return 1 - 0.04/(x-0.6)
}

// loadState is what a load bar knows: the bytes of its programs, those loaded, and the one
// being loaded with its estimate. Guarded by the bar's mu.
type loadState struct {
	total, done int64
	parts       bool // LoadingPart told of a part: else the bar's load is one part
	start       time.Time
	cur         int64 // the part being loaded: its bytes, 0 between parts
	curStart    time.Time
	est         time.Duration // its estimate, 0 when not known
}

// frac is the share to show at now, negative when nothing tells (the spinner alone): the bytes
// loaded and, of the part being loaded, the share of its estimated time gone.
func (l *loadState) frac(now time.Time) float64 {
	if l.total <= 0 {
		return -1
	}
	n := float64(l.done)
	switch {
	case l.cur > 0 && l.est >= slowLoad:
		n += pseudo(now.Sub(l.curStart).Seconds()/l.est.Seconds()) * float64(l.cur)
	case l.done == 0:
		return -1
	}
	return min(n/float64(l.total), 0.99)
}

// NewLoadBar is NewBar for loading a program (or a model's programs) of bytes on disk. Once
// loads before it told how long a compile takes for a size, it shows a bar that goes by the
// time, an estimate; and by the parts loaded, when what loads them tells (LoadingPart). Without
// either (the first program, loads from the compiler's cache) it is the spinner of NewBar.
// End it with Done, or with Failed when the load failed (its time tells nothing).
func NewLoadBar(w io.Writer, label string, bytes int64) *Bar {
	return newBar(w, label, &loadState{total: bytes, cur: bytes, est: estimate(bytes)})
}

// Failed is Done for a load that failed.
func (b *Bar) Failed(result string) {
	if b.load != nil {
		popLoad(b, false)
	}
	b.end(result)
}

func pushLoad(b *Bar) {
	loads.Lock()
	loads.bars = append(loads.bars, b)
	loads.Unlock()
}

// popLoad takes b out of the load bars; a load whose parts were not told is one sample.
func popLoad(b *Bar, ok bool) {
	loads.Lock()
	for i, x := range loads.bars {
		if x == b {
			loads.bars = append(loads.bars[:i], loads.bars[i+1:]...)
			break
		}
	}
	loads.Unlock()
	b.mu.Lock()
	whole := !b.load.parts
	b.mu.Unlock()
	if ok && whole {
		observe(b.load.total, time.Since(b.load.start))
	}
}

// LoadingPart says that one program of bytes on disk is being loaded (compiled, or taken from
// the compiler's cache) until done, which is told whether it loaded: the load bar going on, if
// any, counts it, and its time tells how long the next ones take.
func LoadingPart(bytes int64) (done func(ok bool)) {
	loads.Lock()
	var b *Bar
	if n := len(loads.bars); n > 0 {
		b = loads.bars[n-1]
	}
	loads.Unlock()
	start := time.Now()
	if b != nil {
		est := estimate(bytes)
		b.mu.Lock()
		l := b.load
		if !l.parts {
			l.parts, l.done = true, 0
		}
		l.cur, l.curStart, l.est = bytes, start, est
		b.mu.Unlock()
	}
	var once sync.Once
	return func(ok bool) {
		once.Do(func() {
			if ok {
				observe(bytes, time.Since(start))
			}
			if b != nil {
				b.mu.Lock()
				b.load.cur = 0
				if ok {
					b.load.done += bytes
				}
				b.mu.Unlock()
			}
		})
	}
}
