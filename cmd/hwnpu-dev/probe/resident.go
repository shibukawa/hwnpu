package probe

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/mem"
	"github.com/shibukawa/hwnpu/internal/probe"
	"github.com/shibukawa/hwnpu/internal/types"
)

// The resident mode measures how much the NPU holds at once: programs of one size are loaded one
// after another and kept open (hold open) until a load fails or -resident-max-gb is reached;
// each runs once after its load, and every held one again at the end. Intel's NPU has failed
// with ERROR_OUT_OF_HOST_MEMORY at about 4.6 GB of Anima's fp16 programs held at once (design.md
// 16.138), so after such a failure the oldest program is closed and the failed one loaded again:
// whether closing a program gives its memory back is what Anima's reloading (progEngine) counts
// on. Hold close loads, runs and closes each in turn, the same total, to tell a limit on what is
// held from one on what has been loaded.
//
// A load that does not fail is not a program that stays on the NPU: Core ML evicts without an
// error, and a program evicted is loaded onto the Neural Engine again at its next call. The M1
// ran parts in turn about 12 times slower once they held over about 2.9 GiB together (design.md
// 16.118). So with cycle, after each load every held program runs in turn twice, the second pass
// timed: the total held where a call in turn gets slower than with few held is the most the NPU
// keeps at once. On Apple the Neural Engine driver's wired memory (mem.NPUWired) is recorded with
// each load too: what it really holds, not the weights added up.

type residentOpts struct {
	run     bool // run each program after its load (false: load only, to tell a crash in loading from one in running)
	formats []string
	sizes   []float64 // GiB of one program, each size a run of its own
	cycle   bool      // open: after each load, run every held program in turn and time it
	maxGiB  float64   // stop past this total
	holds   []string
}

// check checks the flags.
func (o *residentOpts) check() error {
	if len(o.formats) == 0 || len(o.holds) == 0 {
		return fmt.Errorf("-resident-formats and -resident-hold: at least one each")
	}
	for _, f := range o.formats {
		if _, ok := probe.BytesPerParam[f]; !ok {
			return fmt.Errorf("-resident-formats: %q is not fp16, int8, lut4 or int4", f)
		}
	}
	for _, h := range o.holds {
		if h != "open" && h != "close" {
			return fmt.Errorf("-resident-hold: %q is not open or close", h)
		}
	}
	if len(o.sizes) == 0 || o.maxGiB < 0 {
		return fmt.Errorf("-resident-size and -resident-max-gb: positive sizes")
	}
	for _, z := range o.sizes {
		if z <= 0 {
			return fmt.Errorf("-resident-size: %v is not a positive size", z)
		}
	}
	return nil
}

// residentResult is one format and way of holding.
type residentResult struct {
	Format       string         `json:"format"`
	Layers       int            `json:"layers"` // of each program
	ProgramBytes int64          `json:"program_bytes"`
	Hold         string         `json:"hold"` // open: every program kept; close: each closed before the next
	MaxBytes     int64          `json:"max_bytes"`
	Loads        []residentLoad `json:"loads"`
	Held         int            `json:"held"`       // programs open together at the most (open), or loaded in turn (close)
	HeldBytes    int64          `json:"held_bytes"` // their weights
	Stop         string         `json:"stop"`       // why it stopped: the error of the load that failed, or "max"
	OutOfMemory  bool           `json:"out_of_memory"`
	Retry        *residentLoad  `json:"retry,omitempty"`   // open, after running out: the failed program again with the oldest closed
	AllRun       string         `json:"all_run,omitempty"` // open: every held program run again after the last load: ok, or the error
	Crashed      bool           `json:"crashed,omitempty"` // the child died (a crash in the driver): Loads are what it reported before
	// cycle: BaseMs is the fastest mean call in turn (few held); FastBytes the most held with
	// every call in turn at about that speed, SlowBytes the least held where it was slower
	// (0: never); Cycled whether it ran at all
	Cycled    bool    `json:"cycled,omitempty"`
	BaseMs    float64 `json:"base_ms,omitempty"`
	FastBytes int64   `json:"fast_bytes,omitempty"`
	SlowBytes int64   `json:"slow_bytes,omitempty"`
	SlowMs    float64 `json:"slow_ms,omitempty"`   // the mean call in turn at SlowBytes
	NPUWired  bool    `json:"npu_wired,omitempty"` // NPUBytes of the loads could be read (Apple)
	Error     string  `json:"error,omitempty"`
}

// residentLoad is one program loaded.
type residentLoad struct {
	Program      int     `json:"program"`
	HeldBytes    int64   `json:"held_bytes"` // weights open with it, its own included
	LoadS        float64 `json:"load_s"`
	RunS         float64 `json:"run_s"`
	OnNPU        bool    `json:"on_npu"`
	PrivateBytes int64   `json:"private_bytes,omitempty"` // the process's private memory after it (Windows; the peak RSS elsewhere)
	// cycle: every held program run in turn, a pass after a pass to warm: the mean and the
	// slowest call of the timed pass, and whether it counted as slow
	CycleMs    float64 `json:"cycle_ms,omitempty"`
	CycleMaxMs float64 `json:"cycle_max_ms,omitempty"`
	Slow       bool    `json:"slow,omitempty"`
	CycleError string  `json:"cycle_error,omitempty"`
	// NPUBytes is what the Neural Engine driver's wired memory grew by since the first load, after
	// this load, its run and the cycle (Apple; 0 where it cannot be read)
	NPUBytes    int64  `json:"npu_bytes,omitempty"`
	OutOfMemory bool   `json:"out_of_memory,omitempty"`
	Error       string `json:"error,omitempty"`
}

// residents runs every format and way of holding, each in a child process of its own.
func (p *prober) residents(rep *report, o residentOpts, save func()) {
	for _, size := range o.sizes {
		for _, f := range o.formats {
			p.residentRuns(rep, o, f, size, save)
		}
	}
}

// residentRuns is one format and size, every way of holding.
func (p *prober) residentRuns(rep *report, o residentOpts, f string, size float64, save func()) {
	per := probe.WeightBytes(f, 1)
	layers := max(1, int(math.Round(size*(1<<30)/float64(per))))
	for _, hold := range o.holds {
		if p.ctx.Err() != nil {
			return
		}
		max := int64(o.maxGiB * (1 << 30))
		fmt.Printf("resident, %s programs of %.2f GiB, %s, up to %.1f GiB:\n", f, probe.GiB(probe.WeightBytes(f, layers)), hold, o.maxGiB)
		var r residentResult
		if err := p.child(childSpec{Kind: "resident", Format: f, Layers: layers, Layout: hold, N: int(max >> 20), Quick: !o.run, Cycle: o.cycle}, &r); err != nil {
			r = residentResult{Format: f, Layers: layers, ProgramBytes: probe.WeightBytes(f, layers), Hold: hold, MaxBytes: max, Error: err.Error()}
			r.fromProgress(p.progress, err.Error())
		}
		for _, l := range r.Loads {
			fmt.Printf("  %s\n", residentLine(l))
		}
		if r.Retry != nil {
			fmt.Printf("  oldest closed, again: %s\n", residentLine(*r.Retry))
		}
		fmt.Printf("  %s\n", r.summary())
		rep.Resident = append(rep.Resident, r)
		save()
	}
}

// fromProgress fills what a child that died reported before: its loads, and how far it got.
func (r *residentResult) fromProgress(lines []string, msg string) {
	for _, line := range lines {
		kind, data, _ := strings.Cut(line, " ")
		var l residentLoad
		if kind != "load" && kind != "retry" || json.Unmarshal([]byte(data), &l) != nil {
			continue
		}
		if kind == "retry" {
			r.Retry = &l
			continue
		}
		r.Loads = append(r.Loads, l)
		if l.Error == "" {
			r.Held++
			if r.Hold == "open" {
				r.HeldBytes = l.HeldBytes
			} else {
				r.HeldBytes += r.ProgramBytes
			}
		}
	}
	r.Crashed = true
	r.Stop = "the process died: " + msg
}

func residentLine(l residentLoad) string {
	s := fmt.Sprintf("program %3d: %6.2f GiB held, load %6.1f s, run %6.3f s", l.Program, probe.GiB(l.HeldBytes), l.LoadS, l.RunS)
	if l.PrivateBytes > 0 {
		s += fmt.Sprintf(", process %.2f GiB", probe.GiB(l.PrivateBytes))
	}
	if l.NPUBytes > 0 {
		s += fmt.Sprintf(", Neural Engine %.2f GiB", probe.GiB(l.NPUBytes))
	}
	if l.CycleMs > 0 {
		s += fmt.Sprintf(", all in turn %.1f ms a call (slowest %.1f)", l.CycleMs, l.CycleMaxMs)
		if l.Slow {
			s += " SLOW"
		}
	}
	if l.CycleError != "" {
		s += "  in turn: " + short(l.CycleError, 120)
	}
	if !l.OnNPU && l.Error == "" {
		s += ", NOT all on the NPU"
	}
	if l.Error != "" {
		s += "  error: " + short(l.Error, 160)
	}
	return s
}

// summary is what the result comes to, in a line.
func (r residentResult) summary() string {
	if r.Error != "" && !r.Crashed {
		return "error: " + short(r.Error, 200)
	}
	verb := "held at once"
	if r.Hold == "close" {
		verb = "loaded in turn"
	}
	s := fmt.Sprintf("%d programs, %.2f GiB %s", r.Held, probe.GiB(r.HeldBytes), verb)
	switch {
	case r.Crashed:
		s += ", then " + short(r.Stop, 200)
	case r.Stop == "max":
		s += fmt.Sprintf(", stopped at -resident-max-gb (%.1f GiB)", probe.GiB(r.MaxBytes))
	case r.OutOfMemory:
		s += ", then out of memory"
	default:
		s += ", then: " + short(r.Stop, 120)
	}
	if r.Retry != nil {
		if r.Retry.Error == "" {
			s += "; with the oldest closed it loaded"
		} else {
			s += "; with the oldest closed it failed again"
		}
	}
	if r.AllRun != "" && r.AllRun != "ok" {
		s += "; running the held ones again: " + short(r.AllRun, 120)
	}
	if c := r.cycleSummary(); c != "" {
		s += "; " + c
	}
	return s
}

// cycleSummary says how far running every held program in turn kept its speed.
func (r residentResult) cycleSummary() string {
	if !r.Cycled {
		return ""
	}
	if r.SlowBytes == 0 {
		return fmt.Sprintf("in turn at full speed (%.1f ms a call) up to all %.2f GiB", r.BaseMs, probe.GiB(r.FastBytes))
	}
	return fmt.Sprintf("in turn at full speed (%.1f ms a call) up to %.2f GiB, %.1f times slower from %.2f GiB", r.BaseMs, probe.GiB(r.FastBytes), r.SlowMs/r.BaseMs, probe.GiB(r.SlowBytes))
}

// residentProgram is a program loaded: run once, close.
type residentProgram interface {
	run() error
	close()
}

type loadedProgram struct {
	p *prober
	l *loaded
}

func (lp loadedProgram) run() error { return lp.l.req.Run(lp.p.ctx) }
func (lp loadedProgram) close()     { lp.l.close() }

// resident is the child side.
func (p *prober) resident(format string, layers int, hold string, maxMiB int, run, cycle bool) residentResult {
	load := func(i int) (residentProgram, result) {
		var res result
		l := p.load(format, layers, 1, "npu", uint64(5000+i), &res)
		if l == nil {
			return nil, res
		}
		return loadedProgram{p, l}, res
	}
	stop := func() bool { return p.ctx.Err() != nil }
	emit := func(kind string, v any) {
		if kind == "step" {
			fmt.Fprintln(os.Stderr, v) // the last of these says where a crash happened
			return
		}
		progress(kind, v)
	}
	return residentRun(residentCfg{format: format, layers: layers, hold: hold, maxMiB: maxMiB, run: run, cycle: cycle, wired: mem.NPUWired}, load, stop, emit)
}

// residentCfg is one run of residentRun.
type residentCfg struct {
	format     string
	layers     int
	hold       string
	maxMiB     int
	run, cycle bool
	wired      func() (int64, bool) // the NPU's memory, as mem.NPUWired (nil: not read)
}

// A call in turn is slow when its mean is over slowFactor times the fastest seen and slower by
// over slowMinMs (so that the noise of fast calls does not count); after slowSteps slow steps in
// a row the cycling stops, since every step then costs a reload of every held program.
const (
	slowFactor = 2.0
	slowMinMs  = 0.5
	slowSteps  = 3
)

// residentRun loads programs with load until one fails or the total passes maxMiB.
// run false loads without running. emit (may be nil) is told each step as it starts ("step", a
// string) and each load as it ends ("load", "retry": a residentLoad).
func residentRun(c residentCfg, loadProgram func(int) (residentProgram, result), stop func() bool, emit func(string, any)) residentResult {
	if emit == nil {
		emit = func(string, any) {}
	}
	format, layers, hold, run := c.format, c.layers, c.hold, c.run
	r := residentResult{Format: format, Layers: layers, ProgramBytes: probe.WeightBytes(format, layers), Hold: hold, MaxBytes: int64(c.maxMiB) << 20}
	var wired0 int64
	if c.wired != nil {
		wired0, r.NPUWired = c.wired()
	}
	npuBytes := func() int64 {
		if !r.NPUWired {
			return 0
		}
		w, ok := c.wired()
		if !ok {
			return 0
		}
		return max(w-wired0, 1) // 1: read, and nothing held yet
	}
	cycling := c.cycle && run && hold == "open"
	slowRun := 0
	var open []residentProgram
	defer func() {
		for _, l := range open {
			if l != nil {
				l.close()
			}
		}
	}()
	held := int64(0)
	load := func(i int) (residentProgram, residentLoad) {
		emit("step", fmt.Sprintf("program %d: loading (%.2f GiB held with it)", i, probe.GiB(held+r.ProgramBytes)))
		t := time.Now()
		l, res := loadProgram(i)
		rl := residentLoad{Program: i, HeldBytes: held + r.ProgramBytes, LoadS: time.Since(t).Seconds(), Error: res.Error}
		if l != nil && run {
			emit("step", fmt.Sprintf("program %d: running", i))
			t = time.Now()
			if err := l.run(); err != nil {
				rl.Error = "run: " + err.Error()
			}
			rl.RunS = time.Since(t).Seconds()
			rl.OnNPU = rl.Error == "" && res.Verified && res.NPUOps > 0 && res.OtherOps == 0
		}
		rl.OutOfMemory = types.IsOutOfMemory(errorString(rl.Error))
		rl.PrivateBytes = privateBytes()
		rl.NPUBytes = npuBytes()
		return l, rl
	}
	// cycle runs every held program in turn twice and times the second pass
	cycle := func(rl *residentLoad) {
		emit("step", fmt.Sprintf("program %d: running all %d held in turn", rl.Program, len(open)))
		var sum, slowest time.Duration
		for pass := 0; pass < 2; pass++ {
			for _, l := range open {
				t := time.Now()
				if err := l.run(); err != nil {
					rl.CycleError = err.Error()
					return
				}
				if d := time.Since(t); pass == 1 {
					sum += d
					slowest = max(slowest, d)
				}
			}
		}
		ms := func(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }
		rl.CycleMs, rl.CycleMaxMs = ms(sum)/float64(len(open)), ms(slowest)
		rl.NPUBytes = npuBytes()
		r.Cycled = true
		if r.BaseMs == 0 || rl.CycleMs < r.BaseMs {
			r.BaseMs = rl.CycleMs
		}
		rl.Slow = rl.CycleMs > slowFactor*r.BaseMs && rl.CycleMs-r.BaseMs > slowMinMs
		switch {
		case rl.Slow:
			if r.SlowBytes == 0 {
				r.SlowBytes, r.SlowMs = rl.HeldBytes, rl.CycleMs
			}
			slowRun++
		case r.SlowBytes == 0:
			r.FastBytes = rl.HeldBytes
			slowRun = 0
		default:
			slowRun = 0
		}
	}
	r.Stop = "max"
	for i := 0; ; i++ {
		// open: what is held; close: what has been loaded in turn
		next := held + r.ProgramBytes
		if hold == "close" {
			next = int64(i+1) * r.ProgramBytes
		}
		if next > r.MaxBytes {
			break
		}
		if stop() {
			r.Stop = "interrupted"
			break
		}
		l, rl := load(i)
		if rl.Error == "" && hold == "open" {
			open = append(open, l)
			held += r.ProgramBytes
			r.Held, r.HeldBytes = max(r.Held, len(open)), max(r.HeldBytes, held)
			if cycling {
				cycle(&rl)
				if rl.CycleError != "" || slowRun >= slowSteps {
					cycling = false
					emit("step", "no more running in turn: "+map[bool]string{true: "it failed", false: "slow for 3 steps in a row"}[rl.CycleError != ""])
				}
			}
		}
		emit("load", rl)
		r.Loads = append(r.Loads, rl)
		if rl.Error != "" {
			if l != nil {
				l.close()
			}
			r.Stop, r.OutOfMemory = rl.Error, rl.OutOfMemory
			if hold == "open" && rl.OutOfMemory && len(open) > 0 {
				// does closing one give its memory back?
				emit("step", "closing the oldest program")
				open[0].close()
				open = open[1:]
				held -= r.ProgramBytes
				l, retry := load(i)
				emit("retry", retry)
				r.Retry = &retry
				if l != nil {
					if retry.Error == "" {
						open = append(open, l)
						held += r.ProgramBytes
					} else {
						l.close()
					}
				}
			}
			break
		}
		if hold == "close" {
			emit("step", fmt.Sprintf("program %d: closing", i))
			l.close()
			r.Held++
			r.HeldBytes += r.ProgramBytes
		}
	}
	if hold == "open" && len(open) > 0 && run {
		emit("step", fmt.Sprintf("running all %d held programs again", len(open)))
		r.AllRun = "ok"
		for _, l := range open {
			if err := l.run(); err != nil {
				r.AllRun = err.Error()
				break
			}
		}
	}
	return r
}

type errorString string

func (e errorString) Error() string { return string(e) }

// residentMarkdown writes the resident section.
func residentMarkdown(sb *strings.Builder, rs []residentResult) {
	sb.WriteString("\n## What the NPU holds at once\n\n")
	sb.WriteString("Programs of one size loaded one after another. open keeps every one (each runs once after its load, all again at the end); close loads, runs and closes each in turn. After running out of memory with programs open, the oldest is closed and the failed one loaded again: does closing give the memory back?\n\n")
	sb.WriteString("A load that does not fail is not a program that stays on the NPU: Core ML evicts without an error and loads the program again at its next call. So, with open, after each load every held program runs in turn twice and the second pass is timed (in turn, ms a call). Full speed in turn is the most held with those calls about as fast as with few held; slower from is the least held where the mean call was over twice as slow. On Apple, Neural Engine is what the wired memory of its driver grew by since the first load: what it really holds.\n\n")
	sb.WriteString("| format | program | hold | programs | weights | stopped by | oldest closed, again | all run again | full speed in turn | slower from | process memory at the end | error |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n")
	for _, r := range rs {
		stop := "max (" + fmt.Sprintf("%.1f GiB", probe.GiB(r.MaxBytes)) + ")"
		if r.Crashed {
			stop = "**crashed**: " + strings.ReplaceAll(short(r.Stop, 200), "|", "/")
			r.Error = ""
		} else if r.Stop != "max" {
			stop = strings.ReplaceAll(short(r.Stop, 120), "|", "/")
			if r.OutOfMemory {
				stop = "out of memory: " + stop
			}
		}
		retry := ""
		if r.Retry != nil {
			retry = "loaded"
			if r.Retry.Error != "" {
				retry = "failed: " + strings.ReplaceAll(short(r.Retry.Error, 80), "|", "/")
			}
		}
		mem := ""
		if n := len(r.Loads); n > 0 && r.Loads[n-1].PrivateBytes > 0 {
			mem = fmt.Sprintf("%.2f GiB", probe.GiB(r.Loads[n-1].PrivateBytes))
		}
		fast, slow := "", ""
		if r.Cycled {
			fast = fmt.Sprintf("%.2f GiB (%.1f ms a call)", probe.GiB(r.FastBytes), r.BaseMs)
			slow = "never"
			if r.SlowBytes > 0 {
				slow = fmt.Sprintf("%.2f GiB (%.1f ms, %.1f×)", probe.GiB(r.SlowBytes), r.SlowMs, r.SlowMs/r.BaseMs)
			}
		}
		fmt.Fprintf(sb, "| %s | %.2f GiB | %s | %d | %.2f GiB | %s | %s | %s | %s | %s | %s | %s |\n", r.Format, probe.GiB(r.ProgramBytes), r.Hold, r.Held, probe.GiB(r.HeldBytes),
			stop, retry, strings.ReplaceAll(short(r.AllRun, 80), "|", "/"), fast, slow, mem, strings.ReplaceAll(short(r.Error, 160), "|", "/"))
	}
	for _, r := range rs {
		if len(r.Loads) == 0 {
			continue
		}
		fmt.Fprintf(sb, "\n### %s, %.2f GiB programs, %s\n\n| program | weights held | load s | run s | on the NPU | in turn, ms a call | slowest ms | Neural Engine | process memory | error |\n|---|---|---|---|---|---|---|---|---|---|\n", r.Format, probe.GiB(r.ProgramBytes), r.Hold)
		rows := r.Loads
		if r.Retry != nil {
			rows = append(append([]residentLoad(nil), rows...), *r.Retry)
		}
		for i, l := range rows {
			name := fmt.Sprint(l.Program)
			if r.Retry != nil && i == len(rows)-1 {
				name += " (again, oldest closed)"
			}
			mem := ""
			if l.PrivateBytes > 0 {
				mem = fmt.Sprintf("%.2f GiB", probe.GiB(l.PrivateBytes))
			}
			cyc, slowest, npu := "", "", ""
			if l.CycleMs > 0 {
				cyc, slowest = fmt.Sprintf("%.1f", l.CycleMs), fmt.Sprintf("%.1f", l.CycleMaxMs)
				if l.Slow {
					cyc = "**" + cyc + "**"
				}
			}
			if l.NPUBytes > 0 {
				npu = fmt.Sprintf("%.2f GiB", probe.GiB(l.NPUBytes))
			}
			msg := l.Error
			if l.CycleError != "" {
				msg = strings.TrimSpace(msg + " in turn: " + l.CycleError)
			}
			fmt.Fprintf(sb, "| %s | %.2f GiB | %.1f | %.3f | %v | %s | %s | %s | %s | %s |\n", name, probe.GiB(l.HeldBytes), l.LoadS, l.RunS, l.OnNPU, cyc, slowest, npu, mem, strings.ReplaceAll(short(msg, 160), "|", "/"))
		}
	}
}
