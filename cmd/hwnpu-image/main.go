// hwnpu-image draws a picture from a prompt with Anima (circlestone-labs/Anima) on an NPU:
// Qwen3-0.6B and Anima's LLM adapter read the prompt, the Cosmos-Predict2 2B DiT denoises the
// latents in flow-matching Euler steps, and the Qwen-Image VAE decodes them, all on the NPU. The
// host tokenizes, looks up the embedding rows, patches the latents and writes the PNG.
//
//	hwnpu-image -prompt "masterpiece, best quality, score_7, safe, 1girl, cherry blossoms" -o out.png
//
// On Apple's Neural Engine and Intel's NPU it makes Anima ready as hwnpu-chat makes its models
// ready: the first run downloads the checkpoint from Hugging Face into -dir/Anima (hwnpu/models
// in the user cache directory) and converts it, in Go, into Core ML packages or OpenVINO IR for
// -size (an export a size, the formats and the layout version in its name); later runs load it.
// The first load also compiles it for the NPU (minutes at 1024 x 1024), kept in -cache.
//
// -model names an export instead: on AMD's NPU, of tools/amd/iron_anima.py (see
// docs/amd-npu-models.md; ~/models/anima-turbo-1024.amd-npu2 when no Apple or Intel NPU is
// found); or one -convert wrote from an Anima directory:
//
//	hwnpu-image -convert ~/models/Anima -backend coreml -size 1024 -model ~/models/anima-turbo-1024.coreml
//
// -cpu runs a Core ML export with the CPU interpreter (slow; for checking without the NPU). Anima-Turbo wants 8 to 12 steps and no classifier-free guidance; this runs no negative
// prompt. Anima reads Danbooru tags and natural language; its card recommends the prefix
// "masterpiece, best quality, score_7, safe, ".
package main

import (
	"cmp"
	"context"
	"encoding/binary"
	"flag"
	"fmt"
	"image/png"
	"log"
	"math"
	"os"
	"os/signal"
	"path/filepath"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/imagegen/anima"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
)

// amdModel is the IRON export used when -model is not given and no Apple or Intel NPU is found.
func amdModel() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, "models", "anima-turbo-1024.amd-npu2")
}

func main() {
	model := flag.String("model", "", "an Anima export to run (AMD: of tools/amd/iron_anima.py), or with -convert the one to write; default: the one of -dir, downloaded and converted when missing")
	dir := flag.String("dir", chat.ModelsDir(), "where the models are kept (as hwnpu-chat's -dir): Anima's checkpoint and exports in Anima/")
	prompt := flag.String("prompt", "", "what to draw (Danbooru tags and natural language)")
	out := flag.String("o", "anima.png", "the PNG to write")
	steps := flag.Int("steps", 8, "Euler steps (Anima-Turbo: 8 to 12)")
	shift := flag.Float64("shift", 3.0, "the sigma schedule's shift (Anima's scheduler: 3)")
	seed := flag.Uint64("seed", 0, "the noise's seed (0: from the time)")
	noise := flag.String("noise", "", "the initial latents as raw float32 (16, H/8, W/8), in place of the seed's")
	latents := flag.String("latents", "", "also write the final latents as raw float32 here")
	convert := flag.String("convert", "", "convert the Anima directory (diffusion_models/, text_encoders/, diffusers/) into -model and exit")
	backend := flag.String("backend", "", "the program format to convert for: coreml (Apple's Neural Engine) or intel (Intel's NPU); default: the NPU found (-convert: coreml)")
	size := flag.Int("size", 1024, "the image's width and height in pixels, a multiple of 16 (an export each)")
	dit := flag.String("dit", "anima-turbo-v1.1.safetensors", "the diffusion model, a file of split_files/diffusion_models/ in circlestone-labs/Anima")
	textFormat := flag.String("text-format", "", "the text encoder's weights (fp16; lut8, lut6 on Core ML; int8 on Intel; default int8 on Core ML, where fp16's 1.0 GB is over a program's limit, fp16 on Intel)")
	ditFormat := flag.String("dit-format", "", "the DiT's weights (fp16; lut8, lut6 on Core ML; int8, int4 on Intel; default int8 on Core ML: faster than fp16 at half its memory, and nearer it than lut8; fp16 on Intel)")
	group := flag.Int("group", 4, "the DiT's blocks a program holds")
	qchunk := flag.Int("qchunk", 512, "the queries an attention scores at a time (M3: 512 is 25 % faster than 1024 at 1024 x 1024)")
	cacheDir := flag.String("cache", "", "compile cache (default: hwnpu in the user cache directory; \"-\" for none)")
	cpu := flag.Bool("cpu", false, "run a Core ML export with the CPU interpreter instead of the NPU")
	dump := flag.String("dump", "", "a directory to write the conditioning and each step's velocity to (raw bf16), to check against tools/ref/ref_anima.py")
	flag.Parse()
	o := anima.ConvertOptions{Backend: *backend, Width: *size, Height: *size, Dit: *dit,
		TextFormat: *textFormat, DitFormat: *ditFormat, Group: *group, QChunk: *qchunk}
	if *convert != "" {
		if *model == "" {
			log.Fatal("hwnpu-image: -convert needs -model, the export to write")
		}
		o.Backend = cmp.Or(o.Backend, llm.BackendCoreML)
		text, dit := anima.Formats(o.Backend)
		o.TextFormat, o.DitFormat = cmp.Or(o.TextFormat, text), cmp.Or(o.DitFormat, dit)
		t0 := time.Now()
		if err := anima.Convert(*convert, *model, o, os.Stderr); err != nil {
			log.Fatal(err)
		}
		fmt.Fprintf(os.Stderr, "%s: converted in %.0f s\n", *model, time.Since(t0).Seconds())
		return
	}
	if *prompt == "" {
		log.Fatal("hwnpu-image: -prompt is required")
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	logf := func(format string, args ...any) { fmt.Fprintf(os.Stderr, format+"\n", args...) }

	if *model == "" {
		format := llm.BackendCoreML // the CPU interpreter runs Core ML exports
		var err error
		if !*cpu {
			format, err = llm.ProgramFormat(ctx, *backend)
		}
		switch {
		case err != nil && *backend == "" && isDir(amdModel()):
			*model = amdModel() // AMD's NPU: an export of tools/amd/iron_anima.py
		case err != nil:
			log.Fatalf("hwnpu-image: %v (on AMD's NPU, -model names an export of tools/amd/iron_anima.py)", err)
		default:
			o.Backend = format
			if *model, err = anima.Prepare(ctx, anima.Setup{Dir: filepath.Join(*dir, "Anima"), ConvertOptions: o, Log: os.Stderr}); err != nil {
				log.Fatal(err)
			}
		}
	}
	t0 := time.Now()
	var dev *hwnpu.Device
	if !*cpu {
		var opts []hwnpu.Option
		if *cacheDir == "" {
			if d, err := os.UserCacheDir(); err == nil {
				*cacheDir = filepath.Join(d, "hwnpu")
			}
		}
		if *cacheDir != "" && *cacheDir != "-" {
			// without it, Core ML compiles each package to a new temporary path and the Neural
			// Engine's compiled bundles are never reused: every start compiled the whole DiT
			// again (118 s at 512 x 512 on M3)
			opts = append(opts, hwnpu.WithCacheDir(*cacheDir))
		}
		if !anima.IsProgramExport(*model) {
			opts = append(opts, hwnpu.WithVendor(hwnpu.VendorAMD)) // IRON graphs
		}
		d, err := hwnpu.Open(ctx, opts...)
		if err != nil {
			log.Fatal(err)
		}
		defer d.Close()
		dev = d
	}
	if !anima.IsProgramExport(*model) {
		iron.ShareExport(*model, logf)
	}
	p, err := anima.Open(ctx, dev, *model)
	if err != nil {
		log.Fatal(err)
	}
	defer p.Close()
	p.Logf = logf
	p.Dump = *dump
	logf("loaded %s (%dx%d) in %.1f s", *model, p.Dit.Width, p.Dit.Height, time.Since(t0).Seconds())

	cond, err := p.Encode(ctx, *prompt)
	if err != nil {
		log.Fatal(err)
	}
	var x []float32
	if *noise != "" {
		if x, err = readF32(*noise); err != nil {
			log.Fatal(err)
		}
	} else {
		s := *seed
		if s == 0 {
			s = uint64(time.Now().UnixNano())
		}
		logf("seed %d", s)
		x = p.Noise(s)
	}
	sig := anima.Sigmas(*steps, *shift)
	t1 := time.Now()
	x, err = p.Denoise(ctx, cond, x, sig, func(i int, d time.Duration) {
		logf("step %d/%d: sigma %.3f -> %.3f, %.1f s", i+1, *steps, sig[i], sig[i+1], d.Seconds())
	})
	if err != nil {
		log.Fatal(err)
	}
	logf("denoised in %.1f s", time.Since(t1).Seconds())
	if *latents != "" {
		if err := writeF32(*latents, x); err != nil {
			log.Fatal(err)
		}
	}
	t2 := time.Now()
	img, err := p.Decode(ctx, x)
	if err != nil {
		log.Fatal(err)
	}
	logf("decoded in %.1f s", time.Since(t2).Seconds())
	f, err := os.Create(*out)
	if err != nil {
		log.Fatal(err)
	}
	if err := png.Encode(f, img); err != nil {
		log.Fatal(err)
	}
	if err := f.Close(); err != nil {
		log.Fatal(err)
	}
	logf("%s: %.1f s in all", *out, time.Since(t0).Seconds())
}

func readF32(path string) ([]float32, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out, nil
}

func writeF32(path string, xs []float32) error {
	b := make([]byte, 4*len(xs))
	for i, x := range xs {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return os.WriteFile(path, b, 0o644)
}

func isDir(path string) bool {
	fi, err := os.Stat(path)
	return err == nil && fi.IsDir()
}
