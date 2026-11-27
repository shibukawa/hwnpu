package chat

import (
	"context"
	"os"
	"runtime"
	"strings"

	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/llm"
)

func init() { airole.Here = hereNPU }

// hereNPU is this machine's NPU (airole.Here), picked as hwnpu-chat, hwnpu-audio and hwnpu-image
// pick it: HWNPU_BACKEND, else the NPU found; on a Mac Core AI's (macOS 27), which runs the Neural
// Engine for them (Core ML is not used).
func hereNPU(ctx context.Context) airole.NPU {
	device := strings.TrimSpace(os.Getenv(llm.BackendEnv))
	if runtime.GOOS == "darwin" {
		switch device {
		case "", llm.BackendCoreML, llm.BackendCoreAI:
			device = llm.BackendCoreAI
		}
	}
	n := airole.NPU{Vendor: HereVendor(), Device: device}
	if OnlyAMD(ctx, device) {
		n.Target = cache.AMDTarget
		return n
	}
	if device == llm.BackendCoreAI {
		n.Format = llm.BackendCoreAI
		if _, _, err := llm.FindDevice(ctx, device); err == nil {
			n.Target = cache.CoreAITarget
		}
		return n
	}
	f, err := llm.ProgramFormat(ctx, device)
	if err != nil {
		return n
	}
	n.Target, n.Format = f, f
	return n
}
