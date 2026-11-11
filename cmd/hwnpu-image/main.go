// hwnpu-image runs image models on an NPU.
//
//	hwnpu-image generate "masterpiece, best quality, score_7, safe, 1girl, cherry blossoms"   # image.png, shown
//	hwnpu-image generate -m anima "masterpiece, best quality, 1girl, cherry blossoms" -o girl.png   # Anima-Aesthetic, 25 DPM-Solver++ steps
//	hwnpu-image generate -m sana "a red fox in a snowy forest" -o fox.png
//	hwnpu-image depth photo.jpg                                  # photo.depth.png
//	hwnpu-image depth -m moge --normal --raw photo.jpg
//	hwnpu-image face-detect --draw photo.jpg                     # JSON lines, photo.faces.png
//	hwnpu-image pose --draw photo.jpg                            # JSON lines, photo.pose.png
//
// --model takes a name, a family or a use of models.yaml (image, anima-turbo, anima, anima:base,
// sana, sana:1.6b; depth, da3:small, da3metric, moge:vits; face-detect, yunet; pose, rtmo): a family alone picks for the machine's
// memory, as hwnpu-chat's -use. On AMD's NPU the model picked runs ~/models/NAME.amd-npu2. A
// value with a / (or a leading . or ~) is an export's directory. Without -o, generate writes
// image.png here (image (2).png, ... when taken) and shows it in the system's viewer.
//
//	hwnpu-image convert ~/models/Anima --backend coreml --size 1024 -m ~/models/anima-turbo-1024.coreml
//
// generate draws a picture from a prompt. With Anima (circlestone-labs/Anima), Qwen3-0.6B and
// Anima's LLM adapter read the prompt, the Cosmos-Predict2 2B DiT denoises the latents in
// flow-matching Euler steps and the Qwen-Image VAE decodes them, all on the NPU; the host
// tokenizes, looks up the embedding rows, patches the latents and writes the PNG. Anima-Turbo
// (the image use's model, anima-turbo) wants 8 to 12 steps and no classifier-free guidance;
// Anima-Aesthetic (anima) and Anima-Base (anima:base) 30 to 50 steps with the guidance at 4 to 5
// (--guidance, --negative), the DiT twice a step: about 6.5 times Anima-Turbo's time. Anima reads
// Danbooru tags and natural language; its card recommends the prefix "masterpiece, best
// quality, score_7, safe, " (without score_7 for Anima-Aesthetic).
//
// On Apple's Neural Engine and Intel's NPU, generate makes Anima ready as hwnpu-chat makes its
// models ready: the first run downloads the checkpoint from Hugging Face into --dir/Anima
// (hwnpu/models in the user cache directory) and converts it, in Go, into Core ML packages or
// OpenVINO IR for --size; later runs load it. The first load also compiles it for the NPU
// (minutes at 1024 x 1024), kept in --cache. convert makes such an export from a directory.
//
// On AMD's NPU, the exports are of tools/amd/iron_anima.py (see docs/amd-npu-models.md;
// ~/models/anima-turbo-1024.amd-npu2, anima-aesthetic-1024.amd-npu2) or of
// tools/amd/iron_sana.py: Sana (NVlabs), a linear DiT with Gemma 2 2B reading the prompt
// (internal/imagegen/sana). On Apple's Neural Engine, Sana is downloaded into --dir/Sana and
// converted into Core ML packages on its first run, as Anima is. Sana runs 20 steps with
// classifier-free guidance (--guidance, 4.5 by default; --negative). --cpu runs a Core ML export with the CPU
// interpreter (slow; for checking without the NPU).
//
// depth estimates the depth of pictures on AMD's NPU or Apple's Neural Engine with Depth
// Anything 3 or MoGe-2 (see DepthCmd). face-detect (FaceDetectCmd) finds faces, pose (PoseCmd)
// people's keypoints.
package main

import (
	"context"
	"os"
	"os/signal"

	"github.com/alecthomas/kong"
)

type cli struct {
	Generate   GenerateCmd   `cmd:"" help:"Draw a picture from a prompt (Anima, Sana)."`
	Depth      DepthCmd      `cmd:"" help:"Estimate the depth of pictures (Depth Anything 3, MoGe-2; AMD's and Apple's NPU)."`
	FaceDetect FaceDetectCmd `cmd:"" name:"face-detect" help:"Find the faces in pictures: boxes and five landmarks (YuNet; the CPU, AMD's or Apple's NPU)."`
	Pose       PoseCmd       `cmd:"" help:"Find the people in pictures and their 17 keypoints (RTMO; Apple's Neural Engine, Intel's NPU)."`
	Convert    ConvertCmd    `cmd:"" help:"Convert an Anima directory into a Core ML or OpenVINO export, or a Sana repository into a Core ML export."`
	List       ListCmd       `cmd:"" help:"List the image and depth models, their exports and what each family picks for a machine's memory."`
}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	var c cli
	k := kong.Parse(&c,
		kong.Name("hwnpu-image"),
		kong.Description("Image models on an NPU: generation, depth, faces and poses."),
		kong.UsageOnError(),
		kong.BindTo(ctx, (*context.Context)(nil)))
	k.FatalIfErrorf(k.Run())
}
