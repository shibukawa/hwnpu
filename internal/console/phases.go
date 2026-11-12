package console

import (
	"fmt"
	"io"
	"slices"
	"sync"
	"time"
)

// The time of a command goes to three phases for its last line (Times): downloading the models,
// preparing them (converting, loading, compiling: everything before Running, and Loading), and
// running (from Running until Preparing again or the end). A download is taken out of the phase
// it falls in; downloads at the same time count once.
var phases struct {
	mu      sync.Mutex
	since   time.Time // the start of the current phase's span
	running bool      // after Running (until Preparing)
	loading int       // Loading going on: preparing while running
	loads   []string  // what they load
	ran     bool      // Running was called
	wall    [2]time.Duration
	// downloads now going on, the start of their span, and the time taken in each phase
	active  int
	dlSince time.Time
	dl      [2]time.Duration
}

func init() { phases.since = time.Now() }

// phaseNow is the index of the phase now: 0 preparing, 1 running.
func phaseNow() int {
	if phases.running && phases.loading == 0 {
		return 1
	}
	return 0
}

// change ends the current phase's span (and the downloads' span) at now and then applies f,
// which changes the phase.
func change(f func()) {
	p := &phases
	p.mu.Lock()
	defer p.mu.Unlock()
	now := time.Now()
	i := phaseNow()
	p.wall[i] += now.Sub(p.since)
	energyChange(i)
	if p.active > 0 {
		p.dl[i] += now.Sub(p.dlSince)
	}
	p.since, p.dlSince = now, now
	f()
}

// Running says that the models are ready and the command's own work begins.
func Running() { change(func() { phases.running, phases.ran = true, true }) }

// Preparing says that the command prepares models again (another model is loaded).
func Preparing() { change(func() { phases.running = false }) }

// Loading counts the time until done as preparing, then goes back to the phase before: for a
// program converted, compiled or loaded while the work goes on (a picture of a new size, a
// stage of a pipeline). Time of the work going on at the same time counts as preparing. what
// (a program's path) names it in the profile (HWNPU_PROFILE).
func Loading(what string) (done func()) {
	var depth int
	var again bool // inside a load of the same (runner.Open of a chain.Open): listed once
	change(func() {
		depth, again = phases.loading, slices.Contains(phases.loads, what)
		phases.loading++
		phases.loads = append(phases.loads, what)
	})
	start := time.Now()
	var once sync.Once
	return func() {
		once.Do(func() {
			change(func() {
				phases.loading--
				if i := slices.Index(phases.loads, what); i >= 0 {
					phases.loads = slices.Delete(phases.loads, i, i+1)
				}
			})
			if !again {
				record(kindLoad, what, depth, time.Since(start), -1)
			}
		})
	}
}

// Reading counts the time until done as preparing, as Loading does, for weights read again
// while the work goes on (a pipeline whose groups of layers are read into one bank in turn:
// iron.GraphOptions.SwapVariants, every step), without a line each in the profile: the profile
// tells their count, bytes (done's, the bytes read) and time once.
func Reading() (done func(bytes int64)) {
	change(func() { phases.loading++ })
	start := time.Now()
	var once sync.Once
	return func(bytes int64) {
		once.Do(func() {
			change(func() { phases.loading-- })
			reads.mu.Lock()
			reads.n++
			reads.bytes += bytes
			reads.d += time.Since(start)
			reads.mu.Unlock()
		})
	}
}

var reads struct {
	mu    sync.Mutex
	n     int
	bytes int64
	d     time.Duration
}

// Downloading counts the time until stop as downloading. what (REPOSITORY/FILE) names it in the
// profile (HWNPU_PROFILE), with the bytes stop is given (negative: none).
func Downloading(what string) (stop func(bytes int64)) {
	change(func() { phases.active++ })
	start := time.Now()
	var once sync.Once
	return func(bytes int64) {
		once.Do(func() {
			change(func() { phases.active-- })
			if bytes >= 0 {
				record(kindDownload, what, 0, time.Since(start), bytes)
			}
		})
	}
}

// PhaseTimes are the time spent downloading, preparing (converting, loading, compiling) and
// running so far.
func PhaseTimes() (download, prepare, run time.Duration) {
	p := &phases
	p.mu.Lock()
	defer p.mu.Unlock()
	now := time.Now()
	wall, dl := p.wall, p.dl
	wall[phaseNow()] += now.Sub(p.since)
	if p.active > 0 {
		dl[phaseNow()] += now.Sub(p.dlSince)
	}
	return dl[0] + dl[1], max(0, wall[0]-dl[0]), max(0, wall[1]-dl[1])
}

// Times writes the line of the phases' times to w ("took 3m12s: downloading 2m40s, converting,
// loading and compiling 25s, running 6.2s"), when the command said Running; a phase with no
// time is left out.
func Times(w io.Writer) {
	phases.mu.Lock()
	ran := phases.ran
	phases.mu.Unlock()
	if w == nil || !ran {
		return
	}
	dl, prep, run := PhaseTimes()
	line := fmt.Sprintf("took %s: ", Duration(dl+prep+run))
	if dl > 0 {
		line += "downloading " + Duration(dl) + ", "
	}
	line += "converting, loading and compiling " + Duration(prep) + ", running " + Duration(run)
	fmt.Fprintln(w, line)
	if e := energyLine(dl+prep, run); e != "" {
		fmt.Fprintln(w, e)
	}
}

// Done writes the command's last lines: Times, and the profile with HWNPU_PROFILE.
func Done(w io.Writer) {
	Times(w)
	WriteProfile(w)
}

// Prepared writes the line of the time spent so far before the work ("ready after 1m02s:
// downloading 25s, converting, loading and compiling 37s"): for a conversation or a server,
// whose running time is not the command's work.
func Prepared(w io.Writer) {
	if w == nil {
		return
	}
	dl, prep, _ := PhaseTimes()
	line := fmt.Sprintf("ready after %s: ", Duration(dl+prep))
	if dl > 0 {
		line += "downloading " + Duration(dl) + ", "
	}
	fmt.Fprintln(w, line+"converting, loading and compiling "+Duration(prep))
}
