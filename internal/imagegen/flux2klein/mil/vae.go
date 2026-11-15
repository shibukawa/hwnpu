package mil

import chroma "github.com/shibukawa/hwnpu/internal/imagegen/chroma/mil"

// The FLUX.2 VAE's decoder is FLUX.1's (Chroma's: conv_in, the mid block with its one-head
// attention, four up blocks of three residual blocks) over 32 latent channels, after
// post_quant_conv. The host un-normalizes the DiT's patches by the VAE's batch-norm statistics
// and un-patches them to the 32 channels at an 8th of the picture's sides.

// flux1 is v as Chroma's decoder takes it.
func flux1(v Vae) chroma.Vae {
	return chroma.Vae{Channels: v.Channels, Layers: v.Layers, Groups: v.Groups, Latent: v.Latent, Scaling: 1, PostQuant: true}
}

// VaeStages is the decoder of a w x h picture in stages, each a program of its own.
func VaeStages(v Vae, w, h int) []chroma.VaeStage { return chroma.VaeStages(flux1(v), w, h) }

// VaeProgram is stage i of VaeStages(v, w, h): input x (1, C, H, W), the latents (1, 32, h/8,
// w/8) for the first; output the next stage's input, or the picture (1, 3, h, w) in [-1, 1]
// roughly (the host clamps).
func VaeProgram(v Vae, w, h, i int) *Part {
	p := chroma.VaeProgram(flux1(v), w, h, i)
	p.Model = "flux2klein"
	return p
}
