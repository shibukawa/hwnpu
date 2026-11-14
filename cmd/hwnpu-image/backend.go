package main

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"strings"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm"
)

// backendFlag is --backend of the commands that run converted programs (ocr, detect, pose,
// face-detect, depth, segment, label, embed). On a Mac hwnpu-image runs its programs on Core AI
// alone (macOS 27; imageDevice), converted for it (.aimodel).
type backendFlag struct {
	Backend string `enum:",coreai,intel,levelzero,openvino" default:"" help:"The device backend for the converted programs: coreai (Apple's Neural Engine through Core AI, macOS 27: a Mac's only one) or intel (Intel's NPU); default: HWNPU_BACKEND, else the NPU found."`
}

// imageDevice is the device backend hwnpu-image opens for the one a command names (name, else
// HWNPU_BACKEND, else the NPU found): on a Mac Core AI's, whatever names Apple's Neural Engine
// (coreml included, as HWNPU_BACKEND may for the other commands); Core ML is not used.
func imageDevice(name string) string {
	if name == "" {
		name = strings.TrimSpace(os.Getenv(llm.BackendEnv))
	}
	if runtime.GOOS == "darwin" {
		switch name {
		case "", llm.BackendCoreML, llm.BackendCoreAI:
			return llm.BackendCoreAI
		}
	}
	return name
}

// errNoCoreAI explains that a Mac without Core AI has no device for hwnpu-image.
func errNoCoreAI(err error) error {
	return fmt.Errorf("hwnpu-image runs on Apple's Neural Engine through Core AI, which needs macOS 27 (--cpu runs on the CPU): %w", err)
}

// imageFormat is the program format hwnpu-image converts for on the device a command names:
// coreai on a Mac (an error without Core AI), else llm.ProgramFormat's (intel; an error where
// neither is found, AMD's NPU running exports of IRON).
func imageFormat(ctx context.Context, name string) (string, error) {
	d := imageDevice(name)
	if d != llm.BackendCoreAI {
		return llm.ProgramFormat(ctx, d)
	}
	if _, _, err := llm.FindDevice(ctx, d); err != nil {
		return "", errNoCoreAI(err)
	}
	return llm.BackendCoreAI, nil
}

// openImageDevice opens the device of imageDevice(name), with the compile cache in cacheDir, and
// returns it with the program format it runs (coreai on a Mac).
func openImageDevice(ctx context.Context, name, cacheDir string, opts ...hwnpu.Option) (*hwnpu.Device, string, error) {
	d := imageDevice(name)
	dev, format, err := llm.OpenDevice(ctx, d, cacheDir, opts...)
	if d == llm.BackendCoreAI {
		if err != nil {
			return nil, "", errNoCoreAI(err)
		}
		format = llm.BackendCoreAI
	}
	return dev, format, err
}

// exportDevice is the device backend to open for an export of the program format: Core AI's for
// an export for Apple's Neural Engine (an older Core ML export is lowered to Core AI as it loads),
// else llm.ExportDevice's.
func exportDevice(format, name string) string {
	if llm.AppleFormat(format) {
		return llm.BackendCoreAI
	}
	return llm.ExportDevice(format, imageDevice(name))
}

// amdOr is the export of AMD's NPU for m (amdExport) when no other NPU was found, or, on a Mac,
// err: imageFormat's without Core AI.
func amdOr(err error, m chat.Model, tool string) (string, error) {
	if runtime.GOOS == "darwin" {
		return "", err
	}
	return amdExport(m, tool)
}
