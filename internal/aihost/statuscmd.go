package aihost

import (
	"context"

	"github.com/shibukawa/hwnpu/internal/npustatus"
)

var _ npustatus.Host = (*Host)(nil)

// NPUDevices opens the NPUs the engine finds, for hwnpu status --wake and hwnpu issue --wake: the
// engine owns the NPU, and the client links no backend to open it with.
func (h *Host) NPUDevices(ctx context.Context) (npustatus.Devices, error) {
	// not after the loads (loading): a status does not wait for a model, as a client's own
	// enumerating did not
	return npustatus.LocalDevices(ctx), nil
}
