// hwnpu-image runs image models on an NPU.
//
//	hwnpu-image generate "masterpiece, best quality, score_7, safe, 1girl, cherry blossoms"   # image.png, shown
//	hwnpu-image generate -m anima "masterpiece, best quality, 1girl, cherry blossoms" -o girl.png   # Anima-Aesthetic, 25 DPM-Solver++ steps
//	hwnpu-image generate -m sana "a red fox in a snowy forest" -o fox.png
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
// --model takes a name, a family or a use of models.yaml (image, anima-turbo, anima, sana,
// sana:1.6b; depth, da3:small, da3metric, moge:vits; face-detect, yunet; pose, rtmo;
// detect, rtmdet:tiny; ocr, ppocr:medium; label, siglip2:large): a family alone picks for the
// machine's memory, as hwnpu-chat's --model. On AMD's NPU the model picked runs ~/models/NAME.amd-npu2. A
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
// Anima-Aesthetic (anima) 25 DPM-Solver++ steps with the guidance at 4 to 5 (--guidance,
// --negative), the DiT twice a step at the noisier ones: about 4.5 times Anima-Turbo's time. Anima reads
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
// On AMD's NPU (and Qwen-Image-2.1-Turbo on Apple's Neural Engine, downloaded into --dir/Qwen-Image
// and converted on its first run: internal/imagegen/qwenimage/mil), generate also runs
// Qwen-Image-2.1 (qwen-image: Qwen-Image-2.1-Turbo, 8 steps;
// qwen-image:2.1, 40 steps, guidance with --guidance and --negative; tools/amd/iron_qwenimage.py,
// internal/imagegen/qwenimage) and Chroma1 (chroma: Chroma1-Flash, Heun's 8 steps; chroma:1-hd,
// guidance 3 with DPM-Solver++ 2M 25 steps; tools/amd/iron_chroma.py, internal/imagegen/chroma).
// A model whose license restricts its use (Qwen-Image-2.1's research license, Anima's
// non-commercial one) is asked once, on a terminal, before its first run; --accept-license (or
// HWNPU_ACCEPT_LICENSE=ID) accepts it without asking.
//
// depth estimates the depth of pictures on AMD's NPU or Apple's Neural Engine with Depth
// Anything 3 or MoGe-2 (see DepthCmd). face-detect (FaceDetectCmd) finds faces, pose (PoseCmd)
// people's keypoints, detect (DetectCmd) objects and their labels, ocr (OCRCmd) lines of text;
// on AMD's NPU, which has no form of RTMO, RTMDet and PP-OCR yet, these three run on the CPU
// with ONNX Runtime. label (LabelCmd) and embed (EmbedCmd) run SigLIP 2 on AMD's NPU: labels
// from any candidates, and embeddings of pictures and texts in one space.
package main

import (
	"context"
	"os"
	"os/signal"

	"github.com/alecthomas/kong"

	"github.com/shibukawa/hwnpu/internal/console"
)

type cli struct {
	Generate   GenerateCmd   `cmd:"" help:"Draw a picture from a prompt (Anima, Sana, Qwen-Image-2.1)."`
	Depth      DepthCmd      `cmd:"" help:"Estimate the depth of pictures (Depth Anything 3, MoGe-2; AMD's and Apple's NPU)."`
	FaceDetect FaceDetectCmd `cmd:"" name:"face-detect" help:"Find the faces in pictures: boxes and five landmarks (YuNet; the CPU, AMD's or Apple's NPU)."`
	Pose       PoseCmd       `cmd:"" help:"Find the people in pictures and their 17 keypoints (RTMO; Apple's Neural Engine, Intel's NPU, the CPU)."`
	Detect     DetectCmd     `cmd:"" help:"Find the objects in pictures: boxes and COCO's 80 labels (RTMDet; Apple's Neural Engine, Intel's NPU, the CPU)."`
	OCR        OCRCmd        `cmd:"" name:"ocr" help:"Read the text in pictures: lines, their boxes and text in 50 languages (PP-OCRv6; Apple's Neural Engine, Intel's NPU, the CPU)."`
	Label      LabelCmd      `cmd:"" help:"Label pictures: the likeliest of candidate labels in any language, or of a list of common ones (SigLIP 2; AMD's NPU)."`
	Embed      EmbedCmd      `cmd:"" help:"Embed pictures and texts in one space, for search and similarity (SigLIP 2; AMD's NPU)."`
	Segment    SegmentCmd    `cmd:"" help:"Cut out the thing a point or a box picks in pictures: its mask and a transparent cutout (EfficientSAM; AMD's NPU, the CPU)."`
	Convert    ConvertCmd    `cmd:"" help:"Convert an Anima directory into a Core ML or OpenVINO export, or a Sana repository into a Core ML export."`
	List       ListCmd       `cmd:"" help:"List the image and depth models, their exports and what each family picks for a machine's memory."`

	AcceptLicense bool `name:"accept-license" help:"Accept the license of a model whose license restricts its use (Qwen-Image-2.1: research only) without being asked; kept, as an answer of yes is."`
}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	var c cli
	k := kong.Parse(&c,
		kong.Name("hwnpu-image"),
		kong.Description("Image models on an NPU: generation, depth, faces, poses, objects, text, labels and cutouts."),
		kong.UsageOnError(),
		kong.BindTo(ctx, (*context.Context)(nil)))
	acceptLicenses = c.AcceptLicense
	console.MeasureEnergy(console.Log(os.Stderr)) // HWNPU_ENERGY: the idle power, then each phase's energy in the last lines
	k.FatalIfErrorf(k.Run())
	console.Done(console.Log(os.Stderr)) // the times, when the command said console.Running; HWNPU_PROFILE's profile
}
