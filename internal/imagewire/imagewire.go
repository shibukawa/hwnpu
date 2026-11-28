// Package imagewire is what hwnpu-image asks hwnpud's engine of that is not a model: its list
// of the image models, why a name picks the model it picks, an export's family, an export a
// definition pins by its name, and an Anima conversion (hwnpu-image convert). The engine serves
// Host on an ai session (internal/aihost, internal/imagecmd), as it serves cachecmd's; the
// client links no engine (docs/design/hwnpud.md, "Clients without the engine").
package imagewire

import (
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/internal/aieng"
	"github.com/shibukawa/hwnpu/internal/airpc"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/daemon"
)

// Host is what hwnpud's engine serves of hwnpu-image on a session (internal/aihost). The logs
// and the outputs come back to the client by the functions; live says that the client's log
// redraws its bars (console.Live).
type Host interface {
	// ImageList writes the models of kinds (all of hwnpu-image's when none) with what each runs
	// from on a machine whose NPU is vendor's, as hwnpu-image list shows them.
	ImageList(ctx context.Context, kinds []string, vendor string, out func([]byte)) error
	// ImagePicked writes why name, a family or a use, picks the model it picks on this machine
	// (nothing for a model's name).
	ImagePicked(ctx context.Context, name string, log func([]byte)) error
	// ImageExport is the model an export's directory runs as (ai/imagegen.Export).
	ImageExport(ctx context.Context, dir string) (ai.Definition, error)
	// ImagePinned is the directory of the export for AMD's NPU the definitions pin by name
	// (DA3-SMALL's multi-view da3-small-mv3.amd-npu2), here or fetched from the sources; "" for
	// none.
	ImagePinned(ctx context.Context, name string, live bool, log func([]byte)) (string, error)
	// ImageConvert converts an Anima checkpoint into an export (hwnpu-image convert).
	ImageConvert(ctx context.Context, c Convert, live bool, log func([]byte)) error
}

// Convert is an Anima conversion: the checkpoint's directory Dir into the export Out (absolute
// paths), with hwnpu-image's flags.
type Convert struct {
	Dir, Out   string
	Backend    string // "": coreai
	Size       string // an aspect ratio's name
	Dit        string
	TextFormat string // "": the backend's
	DitFormat  string
	Group      int
	QChunk     int
}

// Session is a session with hwnpud and its Host.
type Session struct {
	s *airpc.Session
	h *airpc.Handle
}

// Dial opens a session with hwnpud, writing its notes (an older build, other settings) to log.
// Without hwnpud (HWNPU_DAEMON=off, or auto when it does not answer) it is aieng.ErrNone: what
// Host does takes the engine.
func Dial(ctx context.Context, log io.Writer) (*Session, error) {
	c, err := daemon.Use(ctx, daemon.DefaultMode(), log)
	if err != nil {
		return nil, err
	}
	if c == nil {
		return nil, aieng.ErrNone
	}
	s, err := c.DialAI(ctx)
	if err != nil {
		return nil, err
	}
	return &Session{s: s, h: airpc.Root[Host](s)}, nil
}

// Close ends the session.
func (s *Session) Close() error { return s.s.Close() }

// List is Host.ImageList, written to w.
func (s *Session) List(ctx context.Context, w io.Writer, vendor string, kinds ...string) error {
	return older(airpc.Call0(s.h, ctx, "ImageList", kinds, vendor, write(w)))
}

// Picked is Host.ImagePicked, written to log.
func (s *Session) Picked(ctx context.Context, name string, log io.Writer) error {
	return older(airpc.Call0(s.h, ctx, "ImagePicked", name, write(log)))
}

// Export is Host.ImageExport.
func (s *Session) Export(ctx context.Context, dir string) (ai.Definition, error) {
	d, err := airpc.Call1[ai.Definition](s.h, ctx, "ImageExport", dir)
	return d, older(err)
}

// Pinned is Host.ImagePinned, its log to log.
func (s *Session) Pinned(ctx context.Context, name string, log io.Writer) (string, error) {
	d, err := airpc.Call1[string](s.h, ctx, "ImagePinned", name, console.Live(log), write(log))
	return d, older(err)
}

// Convert is Host.ImageConvert, its log to log.
func (s *Session) Convert(ctx context.Context, c Convert, log io.Writer) error {
	return older(airpc.Call0(s.h, ctx, "ImageConvert", c, console.Live(log), write(log)))
}

// write is w as the function a log comes back by.
func write(w io.Writer) func([]byte) { return func(b []byte) { w.Write(b) } }

// older is err, told as what to do when hwnpud is of a build without the method.
func older(err error) error {
	if err == nil || !strings.Contains(err.Error(), "has no method") {
		return err
	}
	return fmt.Errorf("%w: hwnpud is older than hwnpu-image (go install ./cmd/hwnpud && hwnpu service restart)", err)
}

// Vendor is the vendor of the NPU a device backend's name stands for (HWNPU_BACKEND, --backend):
// "intel" for intel, levelzero and openvino, "apple" for coreml and coreai, "" for none (the NPU
// found, hwnpud's catalog's). It is internal/chat's reading of HWNPU_BACKEND (HereVendor).
func Vendor(backend string) string {
	switch strings.TrimSpace(backend) {
	case "intel", "levelzero", "openvino":
		return "intel"
	case "coreml", "coreai":
		return "apple"
	}
	return ""
}

// BackendEnv names the device backend the models are opened on (internal/llm.BackendEnv).
const BackendEnv = "HWNPU_BACKEND"
