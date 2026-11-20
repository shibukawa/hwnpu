package experts

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

// DebugOpt reads an item of HWNPU_MOE_DEBUG, a list of items separated by commas, each NAME or
// NAME=VALUE: "time" (or "1") prints what the experts took; "usage=FILE" adds the counts of the
// experts each layer ran to FILE (Usage) when the experts close; "trace=FILE" writes every row's
// experts in order (a line "LAYER ROWS E E E E" a row, for trying placements on, tools/moeusage); "prefault=0" leaves their pages
// to the first runs, "prefault=wait" has every page in before the first run.
func DebugOpt(name string) (string, bool) {
	for _, it := range strings.Split(os.Getenv("HWNPU_MOE_DEBUG"), ",") {
		k, v, _ := strings.Cut(strings.TrimSpace(it), "=")
		if k == "1" {
			k = "time"
		}
		if k == name {
			return v, true
		}
	}
	return "", false
}

// DebugTime is whether HWNPU_MOE_DEBUG asks for times.
func DebugTime() bool {
	_, ok := DebugOpt("time")
	return ok
}

// traceFile is HWNPU_MOE_DEBUG's trace=FILE.
type traceFile struct {
	f *os.File
	w *bufio.Writer
}

func openTrace(path string) *traceFile {
	f, err := os.Create(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "experts: trace: %v\n", err)
		return nil
	}
	return &traceFile{f, bufio.NewWriter(f)}
}

func (t *traceFile) add(layer int, idx [][]int) {
	if t == nil {
		return
	}
	for _, row := range idx {
		fmt.Fprintf(t.w, "%d %d", layer, len(idx))
		for _, e := range row {
			fmt.Fprintf(t.w, " %d", e)
		}
		t.w.WriteByte('\n')
	}
}

func (t *traceFile) close() {
	if t == nil || t.f == nil {
		return
	}
	t.w.Flush()
	t.f.Close()
	t.f = nil
}
