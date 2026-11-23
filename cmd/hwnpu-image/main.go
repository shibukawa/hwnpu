// hwnpu-image runs image models on an NPU.
//
//	hwnpu-image generate "masterpiece, best quality, score_7, safe, 1girl, cherry blossoms"   # image.png, shown
//	hwnpu-image generate -m anima "masterpiece, best quality, 1girl, cherry blossoms" -o girl.png   # Anima-Aesthetic, 25 DPM-Solver++ steps
//	hwnpu-image generate -m flux2-klein "a red fox in a snowy forest" -o fox.png
//	hwnpu-image depth photo.jpg                                  # photo.depth.png
//	hwnpu-image depth -m moge --normal --raw photo.jpg
//	hwnpu-image face-detect --draw photo.jpg                     # JSON lines, photo.faces.png
//	hwnpu-image pose --draw photo.jpg                            # JSON lines, photo.pose.png
//	hwnpu-image detect --draw photo.jpg                          # JSON lines, photo.objects.png
//	hwnpu-image ocr --text scan.png                              # the lines of text
//	hwnpu-image label photo.jpg                                  # JSON lines: the likeliest of ~440 common labels
//	hwnpu-image label -l "a cat,a dog,猫" photo.jpg               # zero-shot, any candidates, any language
//	hwnpu-image embed --compare photo.jpg -t "a red car"         # cosine and probability of a text and a picture
//
// --model takes a name, a family or a use of models.yaml (image, flux2-klein, anima-turbo, anima;
// depth, da3:small, da3metric, moge:vits; face-detect, yunet; pose, rtmo;
// detect, rtmdet:tiny; ocr, ppocr:medium; label, siglip2:large): a family alone picks for the
// machine's memory, as hwnpu-chat's --model. On AMD's NPU the model picked runs ~/models/NAME.amd-npu2. A
// value with a / (or a leading . or ~) is an export's directory. Without -o, generate writes
// image.png here (image (2).png, ... when taken) and shows it in the system's viewer.
//
//	hwnpu-image convert ~/models/Anima --size 1024 -m ~/models/anima-turbo-1024.coreai
//
// On a Mac hwnpu-image runs on Apple's Neural Engine through Core AI (macOS 27) alone: its
// programs are converted for Core AI (.aimodel) and kept compiled in Core AI's own cache (a
// second start loads in about 0.1 s). Core ML is not used; --cpu runs a Core ML export with the
// CPU interpreter, for checking without the NPU.
//
// generate draws a picture from a prompt. With Anima (circlestone-labs/Anima), Qwen3-0.6B and
// Anima's LLM adapter read the prompt, the Cosmos-Predict2 2B DiT denoises the latents in
// flow-matching Euler steps and the Qwen-Image VAE decodes them, all on the NPU; the host
// tokenizes, looks up the embedding rows, patches the latents and writes the PNG. Anima-Turbo
// (the image use's model, anima-turbo) wants 8 to 12 steps and no classifier-free guidance;
// Anima-Aesthetic (anima) 25 DPM-Solver++ steps with the guidance at 4 to 5 (--guidance,
// --negative), the DiT twice a step at the noisier ones: about 4.5 times Anima-Turbo's time. Anima reads
// Danbooru tags and natural language; its card recommends the prefix "masterpiece, best
// quality, score_7, safe, " (without score_7 for Anima-Aesthetic).
//
// On Apple's Neural Engine and Intel's NPU, generate makes Anima ready as hwnpu-chat makes its
// models ready: the first run downloads the checkpoint from Hugging Face (hwnpu/original in
// the user cache directory) and converts it into hwnpu/models/Anima, in Go, into Core AI programs or
// OpenVINO IR for --size; later runs load it. The first load also compiles it for the NPU
// (minutes at 1024 x 1024), kept in --cache. convert makes such an export from a directory.
//
// On AMD's NPU, the exports are of tools/amd/iron_anima.py (see docs/amd-npu-models.md;
// ~/models/anima-turbo.amd-npu2, anima-aesthetic.amd-npu2, of the eight aspect ratios of --size).
// --cpu runs a Core ML export with the CPU interpreter (slow; for checking without the NPU).
//
// generate also runs FLUX.2 klein 4B (flux2-klein: 4 steps without guidance; --image gives reference images to edit or
// compose from; tools/amd/iron_flux2klein.py, internal/imagegen/flux2klein; the image use's model
// from 8 GB). FLUX.2 klein also runs on Apple's Neural Engine, downloaded and converted into Core
// AI programs on its first run as Anima is.
// A model whose license restricts its use (Anima's non-commercial one) is asked once, on a terminal, before its first run; --accept-license (or
// HWNPU_ACCEPT_LICENSE=ID) accepts it without asking.
//
// depth estimates the depth of pictures on AMD's NPU or Apple's Neural Engine with Depth
// Anything 3 or MoGe-2 (see DepthCmd). face-detect (FaceDetectCmd) finds faces, pose (PoseCmd)
// people's keypoints, detect (DetectCmd) objects and their labels, ocr (OCRCmd) lines of text;
// on AMD's NPU, which has no form of RTMO, RTMDet and PP-OCR yet, these three run on the CPU
// with ONNX Runtime. label (LabelCmd) and embed (EmbedCmd) run SigLIP 2 on AMD's NPU and on
// Apple's Neural Engine (downloaded and converted into Core AI programs on the first run): labels
// from any candidates, and embeddings of pictures and texts in one space.
package main

import (
	"context"
	"os"

	"github.com/alecthomas/kong"

	"github.com/shibukawa/hwnpu/internal/console"
	shared "github.com/shibukawa/hwnpu/internal/store"
)

type cli struct {
	Generate   GenerateCmd   `cmd:"" help:"Draw a picture from a prompt (FLUX.2 klein, FLUX.3 klein, Anima)."`
	Depth      DepthCmd      `cmd:"" help:"Estimate the depth of pictures (Depth Anything 3, MoGe-2; AMD's and Apple's NPU)."`
	FaceDetect FaceDetectCmd `cmd:"" name:"face-detect" help:"Find the faces in pictures: boxes and five landmarks (YuNet; the CPU, AMD's or Apple's NPU)."`
	Pose       PoseCmd       `cmd:"" help:"Find the people in pictures and their 17 keypoints (RTMO; Apple's Neural Engine, Intel's NPU, the CPU)."`
	Detect     DetectCmd     `cmd:"" help:"Find the objects in pictures: boxes and COCO's 80 labels (RTMDet; Apple's Neural Engine, Intel's NPU, the CPU)."`
	OCR        OCRCmd        `cmd:"" name:"ocr" help:"Read the text in pictures: lines, their boxes and text in 50 languages (PP-OCRv6; Apple's Neural Engine, Intel's NPU, the CPU)."`
	Label      LabelCmd      `cmd:"" help:"Label pictures: the likeliest of candidate labels in any language, or of a list of common ones (SigLIP 2; AMD's NPU, Apple's Neural Engine)."`
	Embed      EmbedCmd      `cmd:"" help:"Embed pictures and texts in one space, for search and similarity (SigLIP 2; AMD's NPU, Apple's Neural Engine)."`
	Segment    SegmentCmd    `cmd:"" help:"Cut out the thing a point or a box picks in pictures: its mask and a transparent cutout (EfficientSAM; AMD's NPU, Apple's Neural Engine, Intel's NPU, the CPU)."`
	Convert    ConvertCmd    `cmd:"" help:"Convert an Anima directory into a Core AI export (--backend coreml: the Core ML one --cpu runs; intel: OpenVINO IR)."`
	List       ListCmd       `cmd:"" help:"List the image and depth models, their exports and what each family picks for a machine's memory."`

	AcceptLicense bool `name:"accept-license" help:"Accept the license of a model whose license restricts its use (Anima: non-commercial) without being asked; kept, as an answer of yes is."`
}

func main() {
	ctx, stop := console.Interrupt() // Ctrl-C ends the command, and the process when the command does not return
	defer stop()
	var c cli
	k := kong.Parse(&c,
		kong.Name("hwnpu-image"),
		kong.Description("Image models on an NPU: generation, depth, faces, poses, objects, text, labels and cutouts."),
		kong.UsageOnError(),
		kong.BindTo(ctx, (*context.Context)(nil)))
	acceptLicenses = c.AcceptLicense
	console.MeasureEnergy(console.Log(os.Stderr)) // HWNPU_ENERGY: the idle power, then each phase's energy in the last lines
	console.RecordRun()                           // for hwnpu issue: the arguments, the log and how it ended
	err := console.EndInterrupted(k.Run())
	if err == nil {
		console.Done(console.Log(os.Stderr)) // the times, when the command said console.Running; HWNPU_PROFILE's profile
	}
	shared.Wait(console.Log(os.Stderr)) // what was made here goes in the sources written back to (after the times)
	k.FatalIfErrorf(err)
}
