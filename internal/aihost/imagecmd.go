package aihost

import (
	"context"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/internal/imagecmd"
	"github.com/shibukawa/hwnpu/internal/imagewire"
)

// hwnpu-image's questions that are not a model (internal/imagewire.Host), answered in the
// engine with its definitions and cache (internal/imagecmd).

var _ imagewire.Host = (*Host)(nil)

// ImageList writes hwnpu-image's list of the models of kinds.
func (h *Host) ImageList(ctx context.Context, kinds []string, vendor string, out func([]byte)) error {
	return imagecmd.List(writer(out, false), vendor, kinds...)
}

// ImagePicked writes why name picks its model.
func (h *Host) ImagePicked(ctx context.Context, name string, log func([]byte)) error {
	return imagecmd.Picked(name, writer(log, false))
}

// ImageExport is the model an export's directory runs as.
func (h *Host) ImageExport(ctx context.Context, dir string) (ai.Definition, error) {
	return imagecmd.Export(dir)
}

// ImagePinned is the export the definitions pin by name for AMD's NPU, fetched when it is not
// here.
func (h *Host) ImagePinned(ctx context.Context, name string, live bool, log func([]byte)) (string, error) {
	return imagecmd.Pinned(ctx, name, writer(log, live))
}

// ImageConvert converts an Anima checkpoint into an export (hwnpu-image convert), its log at
// the client.
func (h *Host) ImageConvert(ctx context.Context, c imagewire.Convert, live bool, log func([]byte)) error {
	h.logf("hwnpu-image convert %s for a client", c.Out)
	return imagecmd.Convert(c, writer(log, live))
}
