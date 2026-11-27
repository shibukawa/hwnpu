package experts

import (
	"bufio"
	"fmt"
	"os"

	devenv "github.com/shibukawa/hwnpu/internal/dev"
)

// DebugOpt reads the item moe.NAME of HWNPU_DEV (internal/dev): "time" prints what the experts took; "usage=FILE" adds the counts of the
// experts each layer ran to FILE (Usage) when the experts close; "trace=FILE" writes every row's
// experts in order (a line "LAYER ROWS E E E E" a row, for trying placements on, tools/moeusage);
// "prefault=wait" has every page of experts wired whole in before the first run; "prefetch=N"
// brings in N of the next layer's guessed experts (0: none; the default is as many as a token
// runs); "row=0" runs a token's experts as the rows' three Runs instead of one piece of work each
// (Host.runRow).
func DebugOpt(name string) (string, bool) {
	return devenv.Get("moe." + name)
}

// DebugTime is whether HWNPU_DEV asks for the experts' times (moe.time).
func DebugTime() bool {
	_, ok := DebugOpt("time")
	return ok
}

// traceFile is HWNPU_DEV's moe.trace=FILE.
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
