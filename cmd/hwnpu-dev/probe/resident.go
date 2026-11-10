package probe

import (
	"fmt"
	"math"
	"strings"
	"time"

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

type residentOpts struct {
	formats []string
	size    float64 // GiB of one program
	maxGiB  float64 // stop past this total
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
	if o.size <= 0 || o.maxGiB < 0 {
		return fmt.Errorf("-resident-size and -resident-max-gb: positive sizes")
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
	Error        string         `json:"error,omitempty"`
}

// residentLoad is one program loaded.
type residentLoad struct {
	Program      int     `json:"program"`
	HeldBytes    int64   `json:"held_bytes"` // weights open with it, its own included
	LoadS        float64 `json:"load_s"`
	RunS         float64 `json:"run_s"`
	OnNPU        bool    `json:"on_npu"`
	PrivateBytes int64   `json:"private_bytes,omitempty"` // the process's private memory after it (Windows; the peak RSS elsewhere)
	OutOfMemory  bool    `json:"out_of_memory,omitempty"`
	Error        string  `json:"error,omitempty"`
}

// residents runs every format and way of holding, each in a child process of its own.
func (p *prober) residents(rep *report, o residentOpts, save func()) {
	for _, f := range o.formats {
		per := probe.WeightBytes(f, 1)
		layers := max(1, int(math.Round(o.size*(1<<30)/float64(per))))
		for _, hold := range o.holds {
			if p.ctx.Err() != nil {
				return
			}
			max := int64(o.maxGiB * (1 << 30))
			fmt.Printf("resident, %s programs of %.2f GiB, %s, up to %.1f GiB:\n", f, probe.GiB(probe.WeightBytes(f, layers)), hold, o.maxGiB)
			var r residentResult
			if err := p.child(childSpec{Kind: "resident", Format: f, Layers: layers, Layout: hold, N: int(max >> 20)}, &r); err != nil {
				r = residentResult{Format: f, Layers: layers, ProgramBytes: probe.WeightBytes(f, layers), Hold: hold, MaxBytes: max, Error: err.Error()}
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
}

func residentLine(l residentLoad) string {
	s := fmt.Sprintf("program %3d: %6.2f GiB held, load %6.1f s, run %6.3f s", l.Program, probe.GiB(l.HeldBytes), l.LoadS, l.RunS)
	if l.PrivateBytes > 0 {
		s += fmt.Sprintf(", process %.2f GiB", probe.GiB(l.PrivateBytes))
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
	if r.Error != "" {
		return "error: " + short(r.Error, 200)
	}
	verb := "held at once"
	if r.Hold == "close" {
		verb = "loaded in turn"
	}
	s := fmt.Sprintf("%d programs, %.2f GiB %s", r.Held, probe.GiB(r.HeldBytes), verb)
	switch {
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
	return s
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
func (p *prober) resident(format string, layers int, hold string, maxMiB int) residentResult {
	load := func(i int) (residentProgram, result) {
		var res result
		l := p.load(format, layers, 1, "npu", uint64(5000+i), &res)
		if l == nil {
			return nil, res
		}
		return loadedProgram{p, l}, res
	}
	stop := func() bool { return p.ctx.Err() != nil }
	return residentRun(format, layers, hold, maxMiB, load, stop)
}

// residentRun loads programs with load until one fails or the total passes maxMiB.
func residentRun(format string, layers int, hold string, maxMiB int, loadProgram func(int) (residentProgram, result), stop func() bool) residentResult {
	r := residentResult{Format: format, Layers: layers, ProgramBytes: probe.WeightBytes(format, layers), Hold: hold, MaxBytes: int64(maxMiB) << 20}
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
		t := time.Now()
		l, res := loadProgram(i)
		rl := residentLoad{Program: i, HeldBytes: held + r.ProgramBytes, LoadS: time.Since(t).Seconds(), Error: res.Error}
		if l != nil {
			t = time.Now()
			if err := l.run(); err != nil {
				rl.Error = "run: " + err.Error()
			}
			rl.RunS = time.Since(t).Seconds()
			rl.OnNPU = rl.Error == "" && res.Verified && res.NPUOps > 0 && res.OtherOps == 0
		}
		rl.OutOfMemory = types.IsOutOfMemory(errorString(rl.Error))
		rl.PrivateBytes = privateBytes()
		return l, rl
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
		r.Loads = append(r.Loads, rl)
		if rl.Error != "" {
			if l != nil {
				l.close()
			}
			r.Stop, r.OutOfMemory = rl.Error, rl.OutOfMemory
			if hold == "open" && rl.OutOfMemory && len(open) > 0 {
				// does closing one give its memory back?
				open[0].close()
				open = open[1:]
				held -= r.ProgramBytes
				l, retry := load(i)
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
		if hold == "open" {
			open = append(open, l)
			held += r.ProgramBytes
			r.Held, r.HeldBytes = max(r.Held, len(open)), max(r.HeldBytes, held)
		} else {
			l.close()
			r.Held++
			r.HeldBytes += r.ProgramBytes
		}
	}
	if hold == "open" && len(open) > 0 {
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
	sb.WriteString("| format | program | hold | programs | weights | stopped by | oldest closed, again | all run again | process memory at the end | error |\n|---|---|---|---|---|---|---|---|---|---|\n")
	for _, r := range rs {
		stop := "max (" + fmt.Sprintf("%.1f GiB", probe.GiB(r.MaxBytes)) + ")"
		if r.Stop != "max" {
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
		fmt.Fprintf(sb, "| %s | %.2f GiB | %s | %d | %.2f GiB | %s | %s | %s | %s | %s |\n", r.Format, probe.GiB(r.ProgramBytes), r.Hold, r.Held, probe.GiB(r.HeldBytes),
			stop, retry, strings.ReplaceAll(short(r.AllRun, 80), "|", "/"), mem, strings.ReplaceAll(short(r.Error, 160), "|", "/"))
	}
	for _, r := range rs {
		if len(r.Loads) == 0 {
			continue
		}
		fmt.Fprintf(sb, "\n### %s, %.2f GiB programs, %s\n\n| program | weights held | load s | run s | on the NPU | process memory | error |\n|---|---|---|---|---|---|---|\n", r.Format, probe.GiB(r.ProgramBytes), r.Hold)
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
			fmt.Fprintf(sb, "| %s | %.2f GiB | %.1f | %.3f | %v | %s | %s |\n", name, probe.GiB(l.HeldBytes), l.LoadS, l.RunS, l.OnNPU, mem, strings.ReplaceAll(short(l.Error, 160), "|", "/"))
		}
	}
}
