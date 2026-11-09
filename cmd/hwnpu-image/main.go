// hwnpu-image runs image models on an NPU.
//
//	hwnpu-image generate "masterpiece, best quality, score_7, safe, 1girl, cherry blossoms" -o out.png
//	hwnpu-image generate -m ~/models/sana-sprint-0.6b-1024.amd-npu2 "a red fox in a snowy forest" -o fox.png
//	hwnpu-image depth photo.jpg                                  # photo.depth.png
//	hwnpu-image depth -m ~/models/moge-2-vits.amd-npu2 --normal --raw photo.jpg
//	hwnpu-image convert ~/models/Anima --backend coreml --size 1024 -m ~/models/anima-turbo-1024.coreml
//
// generate draws a picture from a prompt. With Anima (circlestone-labs/Anima), Qwen3-0.6B and
// Anima's LLM adapter read the prompt, the Cosmos-Predict2 2B DiT denoises the latents in
// flow-matching Euler steps and the Qwen-Image VAE decodes them, all on the NPU; the host
// tokenizes, looks up the embedding rows, patches the latents and writes the PNG. Anima-Turbo
// wants 8 to 12 steps and no classifier-free guidance. Anima reads Danbooru tags and natural
// language; its card recommends the prefix "masterpiece, best quality, score_7, safe, ".
//
// On Apple's Neural Engine and Intel's NPU, generate makes Anima ready as hwnpu-chat makes its
// models ready: the first run downloads the checkpoint from Hugging Face into --dir/Anima
// (hwnpu/models in the user cache directory) and converts it, in Go, into Core ML packages or
// OpenVINO IR for --size; later runs load it. The first load also compiles it for the NPU
// (minutes at 1024 x 1024), kept in --cache. convert makes such an export from a directory.
//
// --model names an export instead: on AMD's NPU, of tools/amd/iron_anima.py (see
// docs/amd-npu-models.md; ~/models/anima-turbo-1024.amd-npu2 when no Apple or Intel NPU is
// found), or of tools/amd/iron_sana.py: Sana or Sana-Sprint (NVlabs), a linear DiT with Gemma 2
// 2B reading the prompt (internal/imagegen/sana). Sana-Sprint draws in 2 steps (--steps 1 to 4)
// with the guidance scale embedded; Sana runs 20 steps with classifier-free guidance
// (--guidance, 4.5 by default; --negative). --cpu runs a Core ML export with the CPU
// interpreter (slow; for checking without the NPU).
//
// depth estimates the depth of pictures on AMD's NPU with Depth Anything 3 or MoGe-2 (see
// DepthCmd).
package main

import (
	"context"
	"os"
	"os/signal"

	"github.com/alecthomas/kong"
)

type cli struct {
	Generate GenerateCmd `cmd:"" help:"Draw a picture from a prompt (Anima, Sana, Sana-Sprint)."`
	Depth    DepthCmd    `cmd:"" help:"Estimate the depth of pictures (Depth Anything 3, MoGe-2; AMD's NPU)."`
	Convert  ConvertCmd  `cmd:"" help:"Convert an Anima directory into a Core ML or OpenVINO export."`
}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	var c cli
	k := kong.Parse(&c,
		kong.Name("hwnpu-image"),
		kong.Description("Image models on an NPU: generation and depth."),
		kong.UsageOnError(),
		kong.BindTo(ctx, (*context.Context)(nil)))
	k.FatalIfErrorf(k.Run())
}
