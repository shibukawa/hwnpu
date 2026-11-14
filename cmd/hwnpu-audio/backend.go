package main

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"strings"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
)

// On a Mac hwnpu-audio runs its programs on Apple's Neural Engine through Core AI alone (macOS
// 27), converted for it (.aimodel) from the checkpoints, as hwnpu-image does (design.md 16.196).
// Core ML is left to the CPU interpreter (--cpu, and pitch and notes without ONNX Runtime), which
// reads Core ML exports.

// audioDevice is the device backend hwnpu-audio opens (HWNPU_BACKEND, else the NPU found): on a
// Mac Core AI's for whatever names Apple's Neural Engine (coreml included, as HWNPU_BACKEND may
// for the other commands); Core ML is not used.
func audioDevice() string {
	name := strings.TrimSpace(os.Getenv(llm.BackendEnv))
	if runtime.GOOS == "darwin" {
		switch name {
		case "", llm.BackendCoreML, llm.BackendCoreAI:
			return llm.BackendCoreAI
		}
	}
	return name
}

// errNoCoreAI explains that a Mac without Core AI has no device for hwnpu-audio.
func errNoCoreAI(err error) error {
	return fmt.Errorf("hwnpu-audio runs on Apple's Neural Engine through Core AI, which needs macOS 27 (--cpu runs on the CPU): %w", err)
}

// audioFormat is the program format hwnpu-audio converts for: coreai on a Mac (an error without
// Core AI), else that of the NPU found (intel).
func audioFormat(ctx context.Context) (string, error) {
	d := audioDevice()
	if d != llm.BackendCoreAI {
		return llm.ProgramFormat(ctx, d)
	}
	if _, _, err := llm.FindDevice(ctx, d); err != nil {
		return "", errNoCoreAI(err)
	}
	return llm.BackendCoreAI, nil
}

// openAudioDevice opens the device of audioDevice, with the compile cache in cacheDir, and returns
// it with the program format it runs (coreai on a Mac, intel on Intel's NPU). cpu names what runs
// the model on the CPU instead, for the error of a device that runs neither.
func openAudioDevice(ctx context.Context, cacheDir, cpu string, opts ...hwnpu.Option) (*hwnpu.Device, string, error) {
	d := audioDevice()
	dev, format, err := llm.OpenDevice(ctx, d, cacheDir, opts...)
	if err != nil {
		if d == llm.BackendCoreAI {
			return nil, "", errNoCoreAI(err)
		}
		return nil, "", err
	}
	switch {
	case d == llm.BackendCoreAI:
		format = llm.BackendCoreAI
	case format != llm.BackendIntel:
		dev.Close()
		return nil, "", fmt.Errorf("%s runs neither Core AI's programs nor OpenVINO IR (%s)", dev.Info().Backend, cpu)
	}
	return dev, format, nil
}
