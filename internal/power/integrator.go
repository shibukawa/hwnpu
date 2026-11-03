package power

import (
	"sync"
	"time"

	"github.com/shibukawa/hwnpu/internal/types"
)

// Interval is how often an Integrator samples the power.
const Interval = 50 * time.Millisecond

// Reader reads the power now: watts, what they cover and where they come from.
type Reader func() (watts float64, scope types.PowerScope, source string, err error)

// Integrator turns a meter of the power now (a sensor, AMD's PPT) into energy: its first
// Sample starts a sampler that integrates every Interval, by the trapezoid rule, until Stop.
// From two samples alone, energy would be the mean of two moments.
type Integrator struct {
	read Reader

	mu      sync.Mutex
	energyJ float64
	lastW   float64
	lastT   time.Time
	stop    chan struct{}
	done    chan struct{}
}

// NewIntegrator returns an Integrator over read.
func NewIntegrator(read Reader) *Integrator { return &Integrator{read: read} }

// Sample returns the power now and the energy integrated since the first call.
func (g *Integrator) Sample() (types.PowerSample, error) {
	w, scope, source, err := g.read()
	if err != nil {
		return types.PowerSample{}, err
	}
	now := time.Now()
	g.mu.Lock()
	defer g.mu.Unlock()
	g.add(w, now)
	if g.stop == nil {
		g.stop, g.done = make(chan struct{}), make(chan struct{})
		go g.sample(g.stop, g.done)
	}
	return types.PowerSample{Time: now, EnergyJ: g.energyJ, Watts: w, Scope: scope, Source: source + " (integrated)"}, nil
}

// add adds the energy since the last reading. mu is held.
func (g *Integrator) add(w float64, now time.Time) {
	if !g.lastT.IsZero() {
		g.energyJ += (g.lastW + w) / 2 * now.Sub(g.lastT).Seconds()
	}
	g.lastW, g.lastT = w, now
}

func (g *Integrator) sample(stop <-chan struct{}, done chan<- struct{}) {
	defer close(done)
	t := time.NewTicker(Interval)
	defer t.Stop()
	for {
		select {
		case <-stop:
			return
		case <-t.C:
			if w, _, _, err := g.read(); err == nil {
				g.mu.Lock()
				g.add(w, time.Now())
				g.mu.Unlock()
			}
		}
	}
}

// Stop ends the sampler; a later Sample starts it again.
func (g *Integrator) Stop() {
	g.mu.Lock()
	stop, done := g.stop, g.done
	g.stop = nil
	g.mu.Unlock()
	if stop != nil {
		close(stop)
		<-done
	}
}
