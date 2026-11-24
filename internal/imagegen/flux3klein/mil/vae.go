package mil

import (
	kmil "github.com/shibukawa/bonsaiai/internal/imagegen/flux2klein/mil"
	"github.com/shibukawa/bonsaiai/internal/imagegen/fluxvae"
)

// VaeStages is the decoder of a w x h picture in stages, each a program of its own: FLUX.3 klein's
// VAE is FLUX.2's (AutoencoderKLFlux2), whose decoder is FLUX.1's (kmil.VaeStages).
func VaeStages(v Vae, w, h int) []fluxvae.VaeStage { return kmil.VaeStages(v, w, h) }

// VaeProgram is stage i of VaeStages(v, w, h) (kmil.VaeProgram).
func VaeProgram(v Vae, w, h, i int) *Part {
	p := kmil.VaeProgram(v, w, h, i)
	p.Model = "flux3klein"
	return p
}
