// Package engine runs ai/sound's models in this process: the music, the sound effects, the
// pitch and the notes of internal/music, internal/sfx, internal/pitch and internal/notes.
// ai/engine imports it; ai/sound alone is their client through hwnpud (docs/design/hwnpud.md,
// "Clients without the engine").
package engine

import (
	"context"
	"fmt"
	"io"
	"math"
	"strings"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/ai/sound"
	"github.com/shibukawa/hwnpu/ai/sound/internal/bridge"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/llm"
)

func init() {
	bridge.Roles["music"].Prepare, bridge.Roles["music"].Load = prepareMusic, loadMusic
	bridge.Roles["sfx"].Prepare, bridge.Roles["sfx"].Load = prepareEffect, loadEffect
	bridge.Roles["pitch"].Prepare, bridge.Roles["pitch"].Load = preparePitch, loadPitch
	bridge.Roles["notes"].Prepare, bridge.Roles["notes"].Load = prepareNotes, loadNotes
}

// logf writes lines to w (nil for nowhere).
func logf(w io.Writer) func(string, ...any) {
	return func(format string, args ...any) {
		if w != nil {
			fmt.Fprintf(w, format+"\n", args...)
		}
	}
}

// programFormat is the program format the music and the sound effects are converted for on the
// NPU n: Core AI's on a Mac, OpenVINO IR on Intel's NPU. Others run neither (AMD's runs exports
// of IRON instead, which the roles make apart).
func programFormat(n airole.NPU) (string, error) {
	switch n.Format {
	case llm.BackendCoreAI, llm.BackendIntel:
		return n.Format, nil
	case "":
		return "", fmt.Errorf("no NPU here that runs Core AI's programs or OpenVINO IR (a model of device cpu runs on the CPU)")
	}
	return "", fmt.Errorf("the NPU here runs neither Core AI's programs nor OpenVINO IR (%s; a model of device cpu runs on the CPU)", n.Format)
}

// openNPU opens the NPU n as the music and the sound effects run on it (not AMD's), with the
// compile cache in cacheDir.
func openNPU(ctx context.Context, n airole.NPU, cacheDir string) (*hwnpu.Device, error) {
	if _, err := programFormat(n); err != nil {
		return nil, err
	}
	dev, format, err := llm.OpenDevice(ctx, n.Device, cacheDir)
	if err != nil {
		if n.Format == llm.BackendCoreAI {
			return nil, fmt.Errorf("the models run on Apple's Neural Engine through Core AI, which needs macOS 27 (a model of device cpu runs on the CPU): %w", err)
		}
		return nil, err
	}
	if n.Format != llm.BackendCoreAI && format != llm.BackendIntel {
		dev.Close()
		return nil, fmt.Errorf("%s runs neither Core AI's programs nor OpenVINO IR (a model of device cpu runs on the CPU)", dev.Info().Backend)
	}
	return dev, nil
}

// checkNPU logs the programs not all on the NPU (off: Placement's lines) and refuses them with
// a *sound.OffNPUError unless allow. A program whose placement could not be told is logged and
// does not count.
func checkNPU(log io.Writer, allow bool, off []string) error {
	var bad []string
	for _, l := range off {
		if strings.Contains(l, "placement not known") {
			logf(log)("  going on: %s", l)
			continue
		}
		logf(log)("  NOT all on the NPU: %s", l)
		bad = append(bad, l)
	}
	if len(bad) == 0 || allow {
		return nil
	}
	return &sound.OffNPUError{Programs: bad}
}

// fadeOut fades the last n frames of interleaved audio of channels channels out (all of it
// when it is shorter): a raised cosine from full to silence.
func fadeOut(pcm []float32, channels, n int) {
	frames := len(pcm) / channels
	n = min(n, frames)
	tail := pcm[(frames-n)*channels:]
	for i := range n {
		g := float32(0.5 * (1 + math.Cos(math.Pi*float64(i)/float64(n))))
		for c := range channels {
			tail[i*channels+c] *= g
		}
	}
}

// onNPUWhere is how a log says where a model of the NPU runs.
func onNPUWhere(onNPU bool) string {
	if onNPU {
		return "all on the NPU"
	}
	return "NOT all on the NPU"
}
