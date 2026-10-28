package hwnpu

import (
	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/backend/amd"
	"github.com/shibukawa/hwnpu/internal/backend/levelzero"
)

// Level Zero and AMD are registered on every platform: they report nothing when neither the
// device nor its runtime is present (amdxdna devices are only found on Linux).
func init() {
	backend.Register(levelzero.Provider{})
	backend.Register(amd.Provider{})
}
