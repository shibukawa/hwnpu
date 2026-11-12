package console

import (
	"fmt"
	"io"
	"os"
	"strconv"
	"time"

	"github.com/shibukawa/hwnpu/internal/power"
)

// The energy of a command's phases (HWNPU_ENERGY): the package power integrated (power.Integrator,
// every 50 ms) and split at each change of phase as the times are, with the idle power measured
// before the command so that the energy above idle can be told too.
var energy struct {
	meter *power.Integrator
	idleW float64
	last  float64    // the meter's energy at the last change
	acc   [2]float64 // joules in each phase (preparing, running)
}

// MeasureEnergy starts the energy accounting when HWNPU_ENERGY is set (its value: the seconds of
// idle to measure first, 5 when it is not a number): the idle power over that time, then each
// phase's energy from now, told by Times. Call it first in a command's main: the idle seconds
// are not counted in the times. Without a meter (only AMD's APUs on Linux have one now) it says
// so and does nothing.
func MeasureEnergy(w io.Writer) {
	v := os.Getenv("HWNPU_ENERGY")
	if v == "" {
		return
	}
	idle, err := strconv.ParseFloat(v, 64)
	if err != nil || idle <= 0 {
		idle = 5
	}
	read, err := power.PackagePower()
	if err != nil {
		fmt.Fprintf(w, "HWNPU_ENERGY: %v\n", err)
		return
	}
	m := power.NewIntegrator(read)
	a, err := m.Sample()
	if err != nil {
		fmt.Fprintf(w, "HWNPU_ENERGY: %v\n", err)
		return
	}
	fmt.Fprintf(w, "measuring the idle power for %s (HWNPU_ENERGY)\n", Duration(time.Duration(idle*float64(time.Second))))
	time.Sleep(time.Duration(idle * float64(time.Second)))
	b, err := m.Sample()
	if err != nil {
		m.Stop()
		return
	}
	energy.meter, energy.idleW, energy.last = m, (b.EnergyJ-a.EnergyJ)/b.Time.Sub(a.Time).Seconds(), b.EnergyJ
	phases.mu.Lock()
	phases.since, phases.dlSince = time.Now(), time.Now() // the idle time is not the command's
	phases.mu.Unlock()
}

// energyChange adds the energy since the last change to phase i. phases.mu is held.
func energyChange(i int) {
	if energy.meter == nil {
		return
	}
	if s, err := energy.meter.Sample(); err == nil {
		energy.acc[i] += s.EnergyJ - energy.last
		energy.last = s.EnergyJ
	}
}

// energyLine is Times' line of the phases' energy, "" without HWNPU_ENERGY: the package's joules
// while preparing and running, and the same above the idle power (each phase's time, downloads
// included, at the idle power taken off).
func energyLine(prep, run time.Duration) string {
	if energy.meter == nil {
		return ""
	}
	p := &phases
	p.mu.Lock()
	energyChange(phaseNow())
	acc := energy.acc
	p.mu.Unlock()
	energy.meter.Stop()
	idle := energy.idleW
	netP, netR := acc[0]-idle*prep.Seconds(), acc[1]-idle*run.Seconds()
	return fmt.Sprintf("energy %s: converting, loading and compiling %s, running %s (the package, amdgpu PPT); above the idle %.1f W: %s and %s",
		Joules(acc[0]+acc[1]), Joules(acc[0]), Joules(acc[1]), idle, Joules(netP), Joules(netR))
}

// Joules formats an energy: "820 J", "4.57 kJ".
func Joules(j float64) string {
	if j >= 1000 || j <= -1000 {
		return fmt.Sprintf("%.2f kJ", j/1000)
	}
	return fmt.Sprintf("%.0f J", j)
}
