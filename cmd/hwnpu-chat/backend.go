package main

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"strings"

	"github.com/shibukawa/hwnpu/internal/llm"
)

// chatDevice is the device backend hwnpu-chat converts for and runs on for the one a command names
// (name, else HWNPU_BACKEND, else the NPU found): on a Mac Core AI's, whatever names Apple's Neural
// Engine (coreml included, as HWNPU_BACKEND may for the other commands); Core ML is not used. Its
// chat and embedding models are converted for Core AI from their checkpoints (docs/design.md
// 16.185, 16.191).
func chatDevice(name string) string {
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

// resolveBackend is chatDevice(name), checked on a Mac: Core AI needs macOS 27.
func resolveBackend(ctx context.Context, name string) (string, error) {
	d := chatDevice(name)
	if d == llm.BackendCoreAI {
		if _, _, err := llm.FindDevice(ctx, d); err != nil {
			return "", fmt.Errorf("hwnpu-chat runs on Apple's Neural Engine through Core AI, which needs macOS 27: %w", err)
		}
	}
	return d, nil
}
