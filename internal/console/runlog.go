package console

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu/internal/cache"
)

// What a command did is kept for hwnpu issue: RecordRun at its start writes its arguments to a
// file of logs/ under the cache's root, the writers of Log copy their lines to it (the last
// state of a line of progress, without the colors), and EndInterrupted adds how it ended. The
// last RunsKept runs of each command are kept.

// RunsKept is how many runs of each command logs/ keeps.
const RunsKept = 10

// RunsDir is where the runs are kept.
func RunsDir() string { return cache.Sub("logs") }

// The lines of a run's file that are not the log's start with these.
const (
	RunArgs  = "# args: "
	RunStart = "# start: "
	RunEnd   = "# end: "
	RunError = "# error: "
)

var (
	runMu   sync.Mutex
	runFile *os.File
	runLine []byte // the line being written (a bar's line is written again after \r)
)

// RecordRun starts keeping this run of the command (its os.Args), unless HWNPU_NO_RUN_LOG is
// set. Where logs/ cannot be written, nothing is kept.
func RecordRun() {
	if os.Getenv("HWNPU_NO_RUN_LOG") != "" {
		return
	}
	runMu.Lock()
	defer runMu.Unlock()
	if runFile != nil {
		return
	}
	dir := RunsDir()
	if os.MkdirAll(dir, 0o755) != nil {
		return
	}
	prog := strings.TrimSuffix(filepath.Base(os.Args[0]), ".exe")
	now := time.Now()
	f, err := os.Create(filepath.Join(dir, fmt.Sprintf("%s-%s-%d.log", prog, now.Format("20060102-150405.000"), os.Getpid())))
	if err != nil {
		return
	}
	runFile = f
	fmt.Fprintf(f, "%s%s\n%s%s\n", RunArgs, strings.Join(append([]string{prog}, os.Args[1:]...), "\x1f"), RunStart, now.Format(time.RFC3339))
	prune(dir, prog)
}

// prune removes the runs of prog but the last RunsKept.
func prune(dir, prog string) {
	runs, _ := filepath.Glob(filepath.Join(dir, prog+"-*.log"))
	slices.Sort(runs) // by the time in the name
	for _, r := range runs[:max(0, len(runs)-RunsKept)] {
		os.Remove(r)
	}
}

// endRun writes how the run ended (err nil: it did what it was asked) and closes its file.
func endRun(err error) {
	runMu.Lock()
	defer runMu.Unlock()
	if runFile == nil {
		return
	}
	flushLine()
	if err != nil {
		fmt.Fprintf(runFile, "%s%s\n", RunError, strings.ReplaceAll(err.Error(), "\n", " "))
	}
	fmt.Fprintf(runFile, "%s%s\n", RunEnd, time.Now().Format(time.RFC3339))
	runFile.Close()
	runFile = nil
}

func flushLine() {
	if len(runLine) > 0 {
		runFile.Write(append(runLine, '\n'))
		runLine = runLine[:0]
	}
}

// recordLog copies what a log writer wrote to the run's file: lines, a line's text after its
// last \r, no escape sequences.
func recordLog(p []byte) {
	runMu.Lock()
	defer runMu.Unlock()
	if runFile == nil {
		return
	}
	for _, c := range stripEscapes(p) {
		switch c {
		case '\n':
			runFile.Write(append(runLine, '\n'))
			runLine = runLine[:0]
		case '\r':
			runLine = runLine[:0]
		default:
			runLine = append(runLine, c)
		}
	}
}

// stripEscapes leaves out the escape sequences (colors, erasing a line) of p.
func stripEscapes(p []byte) []byte {
	out := make([]byte, 0, len(p))
	for i := 0; i < len(p); i++ {
		if p[i] != 0x1b {
			out = append(out, p[i])
			continue
		}
		if i+1 < len(p) && p[i+1] == '[' {
			i += 2
			for i < len(p) && (p[i] < 0x40 || p[i] > 0x7e) {
				i++
			}
		}
	}
	return out
}

// teeRun is a log writer without colors that copies to the run's file too.
type teeRun struct{ w *os.File }

// Terminal says whether the file is a terminal (NO_COLOR on one: lines are still redrawn).
func (t teeRun) Terminal() bool { return Terminal(t.w) }

func (t teeRun) Write(p []byte) (int, error) {
	recordLog(p)
	return t.w.Write(p)
}

// Run is a command's run kept in RunsDir.
type Run struct {
	Path  string
	Args  []string
	Start time.Time
	End   time.Time // zero: it did not end (it runs, or was killed)
	Error string    // what it ended with ("" when it did what it was asked)
	Lines []string  // its log
}

// Failed reports whether r ended with an error, or did not end.
func (r Run) Failed() bool { return r.Error != "" || r.End.IsZero() }

// ReadRun reads a run's file.
func ReadRun(path string) (Run, error) {
	f, err := os.Open(path)
	if err != nil {
		return Run{}, err
	}
	defer f.Close()
	r := Run{Path: path}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64<<10), 4<<20)
	for sc.Scan() {
		l := sc.Text()
		switch {
		case strings.HasPrefix(l, RunArgs):
			r.Args = strings.Split(strings.TrimPrefix(l, RunArgs), "\x1f")
		case strings.HasPrefix(l, RunStart):
			r.Start, _ = time.Parse(time.RFC3339, strings.TrimPrefix(l, RunStart))
		case strings.HasPrefix(l, RunEnd):
			r.End, _ = time.Parse(time.RFC3339, strings.TrimPrefix(l, RunEnd))
		case strings.HasPrefix(l, RunError):
			r.Error = strings.TrimPrefix(l, RunError)
		default:
			r.Lines = append(r.Lines, l)
		}
	}
	return r, sc.Err()
}

// Runs are the runs kept, the latest first.
func Runs() []Run {
	files, _ := filepath.Glob(filepath.Join(RunsDir(), "*.log"))
	var runs []Run
	for _, f := range files {
		if r, err := ReadRun(f); err == nil && !r.Start.IsZero() {
			runs = append(runs, r)
		}
	}
	slices.SortFunc(runs, func(a, b Run) int { return b.Start.Compare(a.Start) })
	return runs
}
