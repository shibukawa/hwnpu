package engine

import (
	"context"
	"time"

	"github.com/shibukawa/hwnpu/internal/aistatus"
	"github.com/shibukawa/hwnpu/internal/npustatus"
	"github.com/shibukawa/hwnpu/modelstore"
)

func init() {
	// ai/cli's status, which reads the devices with the engine's packages
	aistatus.Run = func(ctx context.Context, json bool, watch time.Duration, wake bool, store modelstore.Options) error {
		return npustatus.Cmd{JSON: json, Watch: watch, Wake: wake, Store: store}.Run(ctx)
	}
}
