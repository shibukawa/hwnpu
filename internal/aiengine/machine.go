package aiengine

import (
	"context"
	"io"

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

func (local) List(w io.Writer, kinds ...string) {
	ks := make([]chat.Kind, len(kinds))
	for i, k := range kinds {
		ks[i] = chat.Kind(k)
	}
	chat.List(w, ks...)
}
