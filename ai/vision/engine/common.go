package engine

import (
	"context"
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/ai/vision"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/imagedev"
	"github.com/shibukawa/hwnpu/internal/llm"
)

// threads is the CPU's threads of model m (its settings' threads, at least 1).
func threads(m chat.Model) int {
	n, _ := strconv.Atoi(strings.TrimSpace(m.Settings["threads"]))
	return max(1, n)
}

// interpreter reports whether model m runs its converted programs on hwnpu's CPU interpreter.
func interpreter(m chat.Model) bool { return m.Settings["cpu"] == "interpreter" }

// program is the first of the programs of model m named name, "" for none.
func program(m chat.Model, name string) string {
	if ps := m.Programs[name]; len(ps) > 0 {
		return ps[0]
	}
	return ""
}

// openDevice opens the NPU a model of pictures runs on: AMD's for an export of IRON, else the
// one of the programs' format (imagedev.Open: on a Mac Core AI's, which needs macOS 27), where
// Apple's Neural Engine is to run every operation.
func openDevice(ctx context.Context, e airole.LoadEnv) (*hwnpu.Device, error) {
	if e.NPU.AMD() {
		return hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD))
	}
	var opts []hwnpu.Option
	if llm.AppleFormat(e.NPU.Format) {
		opts = append(opts, hwnpu.WithPlacement(hwnpu.PlacementStrict))
	}
	dev, _, err := imagedev.Open(ctx, e.NPU.Device, e.Cache, opts...)
	return dev, err
}

// logf writes lines to w (nil for nowhere).
func logf(w io.Writer) func(string, ...any) {
	return func(format string, args ...any) {
		if w != nil {
			fmt.Fprintf(w, format+"\n", args...)
		}
	}
}

// quiet is a logf that writes nothing.
func quiet(string, ...any) {}

// shown is the model's name in the log: a model the app made (NAME+HASH) by the one it was
// made from.
func shown(name string) string {
	n, _, _ := strings.Cut(name, "+")
	return n
}

// named is model m by its name in the log (shown), for what logs it.
func named(m chat.Model) chat.Model {
	m.Name = shown(m.Name)
	return m
}

// box is the box of corners x1, y1, x2, y2.
func box[F float32 | float64](x1, y1, x2, y2 F) vision.Box {
	return vision.Box{Min: vision.Point{X: float64(x1), Y: float64(y1)}, Max: vision.Point{X: float64(x2), Y: float64(y2)}}
}
