package main

import (
	"context"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/imagedev"
)

// backendFlag is --backend of the commands that run converted programs (ocr, detect, pose,
// face-detect, depth, segment, label, embed). On a Mac hwnpu-image runs its programs on Core AI
// alone (macOS 27; imagedev.Device), converted for it (.aimodel).
type backendFlag struct {
	Backend string `enum:",coreai,intel,levelzero,openvino" default:"" help:"The device backend for the converted programs: coreai (Apple's Neural Engine through Core AI, macOS 27: a Mac's only one) or intel (Intel's NPU); default: HWNPU_BACKEND, else the NPU found."`
}

// The devices of hwnpu-image are picked and opened by internal/imagedev, which ai/vision shares;
// these are its functions by the names the commands call them.

// imageDevice is imagedev.Device: the device backend opened for the one a command names.
func imageDevice(name string) string { return imagedev.Device(name) }

// imageFormat is imagedev.Format: the program format converted for on the device named.
func imageFormat(ctx context.Context, name string) (string, error) {
	return imagedev.Format(ctx, name)
}

// openImageDevice is imagedev.Open: the device of imageDevice(name) and its program format.
func openImageDevice(ctx context.Context, name, cacheDir string, opts ...hwnpu.Option) (*hwnpu.Device, string, error) {
	return imagedev.Open(ctx, name, cacheDir, opts...)
}

// exportDevice is imagedev.ExportDevice: the device backend to open for an export of the format.
func exportDevice(format, name string) string { return imagedev.ExportDevice(format, name) }

// placement is imagedev.Placement: where a loaded program runs, for the log.
func placement(cpu, onNPU bool) string { return imagedev.Placement(cpu, onNPU) }
