// Package aistatus is how ai/cli's Status reaches the engine's NPU status (internal/npustatus,
// which opens the devices) without importing it: the engine (ai/engine) sets Run, and a
// process of a client alone has none (docs/design/hwnpud.md, "Clients without the engine").
package aistatus

import (
	"context"
	"time"

	"github.com/shibukawa/hwnpu/modelstore"
)

// Run writes the NPU's status as hwnpu status does, with store's cache: nil without the engine.
var Run func(ctx context.Context, json bool, watch time.Duration, wake bool, store modelstore.Options) error
