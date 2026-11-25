package mil

import "github.com/shibukawa/bonsaiai/internal/imagegen/fluxvae"

// The FLUX.2 VAE's decoder is FLUX.1's (internal/imagegen/fluxvae: conv_in, the mid block with its one-head
// attention, four up blocks of three residual blocks) over 32 latent channels, after
// post_quant_conv. The host un-normalizes the DiT's patches by the VAE's batch-norm statistics
// and un-patches them to the 32 channels at an 8th of the picture's sides.

// flux1 is v as FLUX.1's decoder (fluxvae) takes it.
func flux1(v Vae) fluxvae.Vae {
	return fluxvae.Vae{Channels: v.Channels, Layers: v.Layers, Groups: v.Groups, Latent: v.Latent, Scaling: 1, PostQuant: true, Quant: v.Quant}
}

// VaeStages is the decoder of a w x h picture in stages, each a program of its own.
func VaeStages(v Vae, w, h int) []fluxvae.VaeStage { return fluxvae.VaeStages(flux1(v), w, h) }

// VaeProgram is stage i of VaeStages(v, w, h): input x (1, C, H, W), the latents (1, 32, h/8,
// w/8) for the first; output the next stage's input, or the picture (1, 3, h, w) in [-1, 1]
// roughly (the host clamps).
func VaeProgram(v Vae, w, h, i int) *Part {
	p := fluxvae.VaeProgram(flux1(v), w, h, i)
	p.Model = "flux2klein"
	return p
}

// VaeEncStages is the encoder of a w x h picture (a reference of an edit) in stages, each a
// program of its own.
func VaeEncStages(v Vae, w, h int) []fluxvae.VaeStage { return fluxvae.VaeEncStages(flux1(v), w, h) }

// VaeEncProgram is stage i of VaeEncStages(v, w, h): input x (1, 3, h, w), the picture in [-1, 1],
// for the first; output the next stage's input, or the latents' means (1, 32, h/8, w/8), which the
// host patches and normalizes by the batch-norm statistics.
func VaeEncProgram(v Vae, w, h, i int) *Part {
	p := fluxvae.VaeEncProgram(flux1(v), w, h, i)
	p.Model = "flux2klein"
	return p
}
