// hwnpu-image draws a picture from a prompt with Anima (circlestone-labs/Anima) on AMD's NPU:
// Qwen3-0.6B and Anima's LLM adapter read the prompt, the Cosmos-Predict2 2B DiT denoises the
// latents in flow-matching Euler steps, and the Qwen-Image VAE decodes them, all as IRON graphs
// on the NPU. The host tokenizes, looks up the embedding rows, patches the latents and writes
// the PNG.
//
//	hwnpu-image -prompt "masterpiece, best quality, score_7, safe, 1girl, cherry blossoms" -o out.png
//
// -model is an export of tools/amd/iron_anima.py (see docs/amd-npu-models.md), for one image
// size. Anima-Turbo wants 8 to 12 steps and no classifier-free guidance; this runs no negative
// prompt. Anima reads Danbooru tags and natural language; its card recommends the prefix
// "masterpiece, best quality, score_7, safe, ".
package main

import (
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
	"github.com/shibukawa/hwnpu/internal/imagegen/anima"
)

func defaultModel() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, "models", "anima-turbo-1024.amd-npu2")
}

func main() {
	model := flag.String("model", defaultModel(), "the Anima export (tools/amd/iron_anima.py)")
	prompt := flag.String("prompt", "", "what to draw (Danbooru tags and natural language)")
	out := flag.String("o", "anima.png", "the PNG to write")
	steps := flag.Int("steps", 8, "Euler steps (Anima-Turbo: 8 to 12)")
	shift := flag.Float64("shift", 3.0, "the sigma schedule's shift (Anima's scheduler: 3)")
	seed := flag.Uint64("seed", 0, "the noise's seed (0: from the time)")
	noise := flag.String("noise", "", "the initial latents as raw float32 (16, H/8, W/8), in place of the seed's")
	latents := flag.String("latents", "", "also write the final latents as raw float32 here")
	dump := flag.String("dump", "", "a directory to write the conditioning and each step's velocity to (raw bf16), to check against tools/ref/ref_anima.py")
	flag.Parse()
	if *prompt == "" {
		log.Fatal("hwnpu-image: -prompt is required")
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	logf := func(format string, args ...any) { fmt.Fprintf(os.Stderr, format+"\n", args...) }

	t0 := time.Now()
	dev, err := hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD))
	if err != nil {
		log.Fatal(err)
	}
	defer dev.Close()
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
