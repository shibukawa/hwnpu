package vision

import (
	"context"
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/shibukawa/bonsaiai"
	"github.com/shibukawa/bonsaiai/internal/airole"
	"github.com/shibukawa/bonsaiai/internal/chat"
	"github.com/shibukawa/bonsaiai/internal/imagedev"
	"github.com/shibukawa/bonsaiai/internal/llm"
)

// The settings and programs the definition of a model of pictures may give (ai.Definition), as
// each role reads them:
//
//	settings: threads   the CPU's threads where it runs on the CPU (1, the default: the least
//	                    CPU time and energy)
//	settings: cpu       interpreter: the converted programs on bonsaiai's CPU interpreter instead
//	                    of the NPU (slow; for checking a conversion), of a model of device cpu
//	programs: export    an export's directory (tools/amd/iron_*.py, or one Prepare made): depth,
//	                    faces and labels
//	programs: onnx      faces: an ONNX file of YuNet
//
// A model of device cpu runs on the CPU: its ONNX files with ONNX Runtime where its library is
// here, else on bonsaiai's own executor. A device the definition gives for every NPU (an app's
// Definition.Device) is chosen as a command's --device is: over BONSAIAI_DEVICE, and an NPU without
// a form of the model is then an error rather than the CPU.

// threads is the CPU's threads of model m (its settings' threads, at least 1).
func threads(m chat.Model) int {
	n, _ := strconv.Atoi(strings.TrimSpace(m.Settings["threads"]))
	return max(1, n)
}

// interpreter reports whether model m runs its converted programs on bonsaiai's CPU interpreter.
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
func openDevice(ctx context.Context, e airole.LoadEnv) (*bonsaiai.Device, error) {
	if e.NPU.AMD() {
		return bonsaiai.Open(ctx, bonsaiai.WithVendor(bonsaiai.VendorAMD))
	}
	var opts []bonsaiai.Option
	if llm.AppleFormat(e.NPU.Format) {
		opts = append(opts, bonsaiai.WithPlacement(bonsaiai.PlacementStrict))
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

// Point is a point of a picture, in its pixels.
type Point struct{ X, Y float64 }

// Box is a rectangle of a picture, in its pixels: Min its top left corner, Max its bottom right.
type Box struct{ Min, Max Point }

// Dx and Dy are the box's width and height.
func (b Box) Dx() float64 { return b.Max.X - b.Min.X }
func (b Box) Dy() float64 { return b.Max.Y - b.Min.Y }

// box is the box of corners x1, y1, x2, y2.
func box[F float32 | float64](x1, y1, x2, y2 F) Box {
	return Box{Point{float64(x1), float64(y1)}, Point{float64(x2), float64(y2)}}
}
