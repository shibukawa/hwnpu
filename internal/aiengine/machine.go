package aiengine

import (
	"context"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/aieng"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/types"
)

// the machine of internal/chat and of the root package (aieng.Machine)
func init() { aieng.M = local{} }

func (local) RunFile() string { return chat.RunFile() }

func (local) Enumerate(ctx context.Context) ([]types.DeviceInfo, error) {
	return hwnpu.Enumerate(ctx)
}
