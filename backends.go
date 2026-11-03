package hwnpu

import (
	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/backend/amd"
	"github.com/shibukawa/hwnpu/internal/backend/amdxrt"
	"github.com/shibukawa/hwnpu/internal/backend/levelzero"
	"github.com/shibukawa/hwnpu/internal/backend/openvino"
)

// Level Zero and AMD are registered on every platform: they report nothing when neither the
// device nor its runtime is present (amdxdna devices are only found on Linux). OpenVINO, for
// checking the Intel path without Intel's NPU, reports a device only with HWNPU_BACKEND=openvino.
func init() {
	backend.Register(levelzero.Provider{})
	backend.Register(amdxrt.Provider{}) // before amd: outside Windows it lists only when asked
	backend.Register(amd.Provider{})
	backend.Register(openvino.Provider{})
}
