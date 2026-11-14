package console

import (
	"fmt"
	"io"
	"os"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu/internal/mem"
)

// Jobs is work run side by side (programs a conversion writes), shown as a few lines: on a
// terminal, a bar of the jobs done and a line for each job running (its label and seconds),
// redrawn ten times a second and erased when all are done, so that twenty programs take no
// twenty lines; a job that fails leaves its line. Elsewhere each job writes a line when it ends,
// "[ 7/20] label result", in the order they end.
type Jobs struct {
	w    io.Writer
	list []job
}

type job struct {
	label string
	do    func() (string, error)
}

// NewJobs starts a list of jobs for w (nil: nothing is shown).
func NewJobs(w io.Writer) *Jobs { return &Jobs{w: w} }

// Add puts a job on the list: do does it and gives what its line says after the label (its time is
// put before that).
func (j *Jobs) Add(label string, do func() (string, error)) {
	j.list = append(j.list, job{label, do})
}

// Len is the jobs on the list.
func (j *Jobs) Len() int { return len(j.list) }

// Workers is the jobs run at a time when each takes up to perJob bytes at its peak (0: little):
// HWNPU_JOBS when set (1: one at a time), else the CPU's cores, and no more than three quarters of
// the memory available holds (two when that is not known).
func Workers(perJob int64) int {
	if n, err := strconv.Atoi(os.Getenv("HWNPU_JOBS")); err == nil && n > 0 {
		return n
	}
	n := runtime.NumCPU()
	if perJob > 0 {
		if avail := mem.Available(); avail > 0 {
			n = min(n, int(avail*3/4/perJob))
		} else {
			n = min(n, 2)
		}
	}
	return max(n, 1)
}

// Run does the jobs, Workers(perJob) at a time, taking them in the order they were added, and
// empties the list. perJob is the bytes a job takes at its peak, beyond what the jobs share (the
// checkpoints' tensors a job reads, the program it holds, the garbage collector's room). The first
// error stops the jobs not started; those running end, and it is returned.
func (j *Jobs) Run(perJob int64) error {
	return j.run(Workers(perJob))
}

func (j *Jobs) run(workers int) error {
	list := j.list
	j.list = nil
	if len(list) == 0 {
		return nil
	}
	workers = max(1, min(workers, len(list)))
	r := &jobRun{w: j.w, total: len(list), workers: workers, live: j.w != nil && Live(j.w),
		start: time.Now(), running: map[int]time.Time{}}
	if r.live {
		r.stop, r.exited = make(chan struct{}), make(chan struct{})
		go r.loop()
	}
	next := make(chan int)
	var wg sync.WaitGroup
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range next {
				r.begin(i, list[i].label)
				res, err := list[i].do()
				r.end(i, list[i].label, res, err)
			}
		}()
	}
	for i := range list {
		if r.failed() {
			break
		}
		next <- i
	}
	close(next)
	wg.Wait()
	if r.live {
		close(r.stop)
		<-r.exited
	}
	return r.err
}

// jobRun is the state of Run.
type jobRun struct {
	w              io.Writer
	total, workers int
	live           bool
	start          time.Time
	stop, exited   chan struct{}

	mu      sync.Mutex
	done    int
	running map[int]time.Time // by the job's index: when it started
	labels  map[int]string
	err     error
	shown   int // the lines drawn last (live)
	frame   int
}

func (r *jobRun) failed() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.err != nil
}

func (r *jobRun) begin(i int, label string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.running[i] = time.Now()
	if r.labels == nil {
		r.labels = map[int]string{}
	}
	r.labels[i] = label
}

func (r *jobRun) end(i int, label, res string, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	took := Duration(time.Since(r.running[i]))
	delete(r.running, i)
	r.done++
	line := fmt.Sprintf("  [%*d/%d] %s %s", len(fmt.Sprint(r.total)), r.done, r.total, label, took)
	if res != "" {
		line += ", " + res
	}
	if err != nil {
		line = fmt.Sprintf("  %s failed after %s", label, took)
		if r.err == nil {
			r.err = err
		}
	}
	if r.w == nil {
		return
	}
	if !r.live {
		fmt.Fprintln(r.w, line)
		return
	}
	if err != nil {
		// a failure stays: written where the lines are, which are drawn again under it
		io.WriteString(r.w, r.erase()+oneLine(r.w, line)+"\n"+r.lines())
	}
}

func (r *jobRun) loop() {
	defer close(r.exited)
	t := time.NewTicker(100 * time.Millisecond)
	defer t.Stop()
	r.mu.Lock()
	io.WriteString(r.w, r.lines())
	r.mu.Unlock()
	for {
		select {
		case <-r.stop:
			r.mu.Lock()
			io.WriteString(r.w, r.erase())
			r.mu.Unlock()
			return
		case <-t.C:
			r.mu.Lock()
			r.frame++
			io.WriteString(r.w, r.erase()+r.lines())
			r.mu.Unlock()
		}
	}
}

// erase moves the cursor back to the first of the lines drawn and clears them and what is below.
func (r *jobRun) erase() string {
	s := "\r"
	if r.shown > 1 {
		s += fmt.Sprintf("\x1b[%dA", r.shown-1)
	}
	r.shown = 0
	return s + "\x1b[J"
}

// lines draws the bar and a line for each job running, the cursor left at the end of the last
// (no newline, so that nothing scrolls when they are drawn again).
func (r *jobRun) lines() string {
	var s strings.Builder
	full := r.done * barWidth / r.total
	at := ""
	if r.workers > 1 {
		at = fmt.Sprintf(", %d at a time", r.workers)
	}
	s.WriteString(oneLine(r.w, fmt.Sprintf("  %s ▕%s%s▏ %d/%d %s%s", spinFrames[r.frame%len(spinFrames)],
		strings.Repeat("█", full), strings.Repeat("░", barWidth-full), r.done, r.total, Duration(time.Since(r.start)), at)))
	r.shown = 1
	idx := make([]int, 0, len(r.running))
	for i := range r.running {
		idx = append(idx, i)
	}
	slices.Sort(idx)
	for _, i := range idx {
		s.WriteString("\n" + oneLine(r.w, fmt.Sprintf("    %s %s", r.labels[i], Duration(time.Since(r.running[i])))))
		r.shown++
	}
	return s.String()
}
