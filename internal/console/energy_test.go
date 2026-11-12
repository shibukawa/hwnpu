package console

import (
	"math"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/internal/power"
	"github.com/shibukawa/hwnpu/internal/types"
)

// TestEnergy splits a meter's energy at the changes of phase: 10 W while preparing (and while
// reading weights again), 30 W while running, with an idle power of 5 W.
func TestEnergy(t *testing.T) {
	resetPhases()
	reads.n, reads.bytes, reads.d = 0, 0, 0
	var watts atomic.Value
	watts.Store(10.0)
	m := power.NewIntegrator(func() (float64, types.PowerScope, string, error) {
		return watts.Load().(float64), types.ScopePackage, "test", nil
	})
	s, _ := m.Sample()
	energy.meter, energy.idleW, energy.last, energy.acc = m, 5, s.EnergyJ, [2]float64{}
	defer func() { energy.meter = nil }()
	const u = 200 * time.Millisecond
	time.Sleep(u) // preparing at 10 W
	watts.Store(30.0)
	Running()
	time.Sleep(u) // running at 30 W
	watts.Store(10.0)
	done := Reading()
	time.Sleep(u) // reading at 10 W: preparing
	done(1 << 20)
	watts.Store(30.0)
	time.Sleep(u)
	_, prep, run := PhaseTimes()
	line := energyLine(prep, run)
	acc := energy.acc
	// the sampler runs every 50 ms: each change can move one interval's difference
	tol := 20.0 * power.Interval.Seconds() * 2
	if want := 10 * 2 * u.Seconds(); math.Abs(acc[0]-want) > tol {
		t.Errorf("preparing %.2f J, want about %.2f (%s)", acc[0], want, line)
	}
	if want := 30 * 2 * u.Seconds(); math.Abs(acc[1]-want) > tol {
		t.Errorf("running %.2f J, want about %.2f (%s)", acc[1], want, line)
	}
	if !strings.Contains(line, "above the idle 5.0 W") {
		t.Errorf("line %q", line)
	}
	if reads.n != 1 || reads.bytes != 1<<20 {
		t.Errorf("reads %d of %d bytes", reads.n, reads.bytes)
	}
}

func TestJoules(t *testing.T) {
	for _, c := range []struct {
		j    float64
		want string
	}{{820.4, "820 J"}, {4574, "4.57 kJ"}, {-12, "-12 J"}} {
		if got := Joules(c.j); got != c.want {
			t.Errorf("Joules(%g) = %q, want %q", c.j, got, c.want)
		}
	}
}
