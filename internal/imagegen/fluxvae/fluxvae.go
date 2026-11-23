// Package fluxvae describes FLUX.1's VAE (diffusers' AutoencoderKL with 16 or 32 latent channels:
// conv_in, the mid block with its one-head attention, four up blocks of three residual blocks)
// as MIL programs in stages, the decoder (VaeStages) and the encoder (VaeEncStages). FLUX.2
// klein's VAE (internal/imagegen/flux2klein) and FLUX.3 klein's take it; it was Chroma1's
// (removed, docs/not-adopted/chroma1.md) and stays as Chroma's code left it, so that the
// programs of FLUX.2 and FLUX.3 are the same bytes.
package fluxvae

import "github.com/shibukawa/hwnpu/internal/imagegen/milpart"

// Part is one program (milpart.Part).
type Part = milpart.Part

type graph = milpart.Graph

func part(name string, body func(g *graph)) *Part {
	return &Part{Name: name, Model: "chroma", Body: body}
}

// Vae is the FLUX VAE's shape (vae/config.json).
type Vae struct {
	Channels []int // block_out_channels (128, 256, 512, 512)
	Layers   int   // layers_per_block (2: the decoder's up blocks have 3)
	Groups   int
	Latent   int
	Scaling  float64
	Shift    float64
	// PostQuant: the latents go through post_quant_conv (a 1 x 1 convolution) before conv_in, in
	// place of the scaling and the shift (FLUX.2's VAE, internal/imagegen/flux2klein)
	PostQuant bool
	// Quant: the encoder's moments go through quant_conv (a 1 x 1 convolution) (FLUX.2's VAE)
	Quant bool
}

// maxScoreBytes is the largest fp16 score tensor (heads x queries x keys) an attention should
// make at once: past about 64 MiB the Neural Engine runs q k^T and its softmax 2 to 3 times
// slower per score (docs/design/image-apple.md).
const maxScoreBytes = 64 << 20
