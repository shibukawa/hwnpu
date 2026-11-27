package aihost

import (
	"context"

	"github.com/shibukawa/hwnpu/internal/cachecmd"
)

var _ cachecmd.ListHost = (*Host)(nil)

// ListModels writes the commands' list of models (hwnpu-audio list) as the engine's definitions,
// read with the client's files, give it: its output and the definitions' log at the client.
func (h *Host) ListModels(ctx context.Context, r cachecmd.ListRequest, out, errOut func([]byte)) error {
	return r.Exec(writer(out, false), writer(errOut, false))
}
