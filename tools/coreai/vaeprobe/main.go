// Command vaeprobe runs an image model's staged programs (a VAE's decoder or encoder, exported as
// vae_s0, vae_s1, ...) one after the other on the Neural Engine, each stage's output the next one's
// input, and writes every stage's output, for finding the stage a picture goes wrong at
// (docs/design/apple-coreai.md). With the fp64 references beside it (ref_anima.py, ref_flux.py)
// each stage is judged from its own input, so a wrong stage shows by itself.
//
//	go build -o vaeprobe ./tools/coreai/vaeprobe
//	HWNPU_DAEMON=off HWNPU_BACKEND=coreai ./vaeprobe OUT EXPORT.coreai FUNCTION STAGE...
//	  e.g. ./vaeprobe out ~/Library/Caches/hwnpu/models/Anima/anima-...-v6.coreai s512x512 vae_s0 ... vae_s9
//	python3 tools/coreai/vaeprobe/ref_anima.py CKPT out     (each stage against fp64: PSNR, rows that stand out)
//	python3 tools/coreai/vaeprobe/cmp.py out-a out-b        (two runs' outputs, byte for byte)
//
// It writes OUT/STAGE.NAME.DTYPE.[DIMS].bin for each stage's output and, when the last output is a
// picture (1, 3, h, w), OUT.png. The first stage's input is random (seed 42) unless one of these
// is set: PROBE_INPUT (a file of the input's bytes: another run's output of the stage before, to
// see a stage alone on a right input), PROBE_LATENT (FLUX's latents (1, 32, 128, 128) fp16, scaled to
// the size), PROBE_IMAGE (a PNG, for an encoder's first stage).
//
// The programs load as hwnpu's: from the export's prebuilt programs when it has them for this chip
// (EXPORT.coreai.prebuilt/ARCH), compiled on the device with HWNPU_COREAI_PREBUILT=off. It takes
// the Neural Engine lock (llm.OpenDevice). Core AI keeps its cache by the executable's name: build
// it under one name for each thing compared (vaeprobe-pre, vaeprobe-dev).
//
// A wrong compile can stay in the system's compile cache (aned's) and come back for the same
// program under any process name, with coreai-cache removed too: a copy of the program made
// structurally new (an extra multiply by 1 on its input) compiles afresh, and tells a wrong
// program from a wrong cache entry.
package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/types"
)

func f16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b>>16) & 0x8000
	exp := int((b>>23)&0xff) - 127 + 15
	man := b & 0x7fffff
	switch {
	case exp <= 0:
		return sign
	case exp >= 31:
		return sign | 0x7c00
	}
	return sign | uint16(exp)<<10 | uint16(man>>13)
}

// picture reads a PNG as the encoder's input: the centre cropped to w x h's aspect, scaled
// (bilinear) to w x h, channels first, v/127.5 - 1 (flux2klein.EncodeImage), as fp16.
func picture(path string, w, h int) []byte {
	f, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		panic(err)
	}
	b := img.Bounds()
	sw, sh := b.Dx(), b.Dy()
	cw, ch := sw, sh
	if sw*h > sh*w {
		cw = sh * w / h
	} else {
		ch = sw * h / w
	}
	x0, y0 := (sw-cw)/2, (sh-ch)/2
	out := make([]byte, 2*3*w*h)
	for y := 0; y < h; y++ {
		fy := (float64(y)+0.5)*float64(ch)/float64(h) - 0.5
		for x := 0; x < w; x++ {
			fx := (float64(x)+0.5)*float64(cw)/float64(w) - 0.5
			ix, iy := int(math.Floor(fx)), int(math.Floor(fy))
			ax, ay := fx-float64(ix), fy-float64(iy)
			var c [3]float64
			for _, d := range [][3]float64{{0, 0, (1 - ax) * (1 - ay)}, {1, 0, ax * (1 - ay)}, {0, 1, (1 - ax) * ay}, {1, 1, ax * ay}} {
				px := min(max(ix+int(d[0]), 0), cw-1) + x0 + b.Min.X
				py := min(max(iy+int(d[1]), 0), ch-1) + y0 + b.Min.Y
				r, g, bl, _ := img.At(px, py).RGBA()
				c[0] += d[2] * float64(r>>8)
				c[1] += d[2] * float64(g>>8)
				c[2] += d[2] * float64(bl>>8)
			}
			for k := 0; k < 3; k++ {
				binary.LittleEndian.PutUint16(out[2*(k*w*h+y*w+x):], f16(float32(c[k]/127.5-1)))
			}
		}
	}
	return out
}

// latent reads a (1, 32, 128, 128) fp16 latent (the encoder's at 1024 x 1024) and scales it
// (bilinear) to (1, 32, h, w).
func latent(path string, w, h int) []byte {
	raw, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}
	const S = 128
	out := make([]byte, 2*32*w*h)
	for c := 0; c < 32; c++ {
		at := func(x, y int) float64 {
			x, y = min(max(x, 0), S-1), min(max(y, 0), S-1)
			return float64(f32of16(binary.LittleEndian.Uint16(raw[2*(c*S*S+y*S+x):])))
		}
		for y := 0; y < h; y++ {
			fy := (float64(y)+0.5)*S/float64(h) - 0.5
			for x := 0; x < w; x++ {
				fx := (float64(x)+0.5)*S/float64(w) - 0.5
				ix, iy := int(math.Floor(fx)), int(math.Floor(fy))
				ax, ay := fx-float64(ix), fy-float64(iy)
				v := (1-ax)*(1-ay)*at(ix, iy) + ax*(1-ay)*at(ix+1, iy) + (1-ax)*ay*at(ix, iy+1) + ax*ay*at(ix+1, iy+1)
				binary.LittleEndian.PutUint16(out[2*(c*w*h+y*w+x):], f16(float32(v)))
			}
		}
	}
	return out
}

func f32of16(u uint16) float32 {
	sign := uint32(u>>15) << 31
	exp := int(u>>10) & 0x1f
	man := uint32(u & 0x3ff)
	switch exp {
	case 0:
		return math.Float32frombits(sign) + float32(man)/1024/16384*map[bool]float32{true: -1, false: 1}[sign != 0]
	case 31:
		return math.Float32frombits(sign | 0x7f800000 | man<<13)
	}
	return math.Float32frombits(sign | uint32(exp-15+127)<<23 | man<<13)
}

// writePNG writes a (1, 3, h, w) fp16 output in -1..1 as a PNG.
func writePNG(path string, b []byte, w, h int) {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			var c [3]uint8
			for k := 0; k < 3; k++ {
				v := f32of16(binary.LittleEndian.Uint16(b[2*(k*w*h+y*w+x):]))
				c[k] = uint8(min(max((v+1)*127.5+0.5, 0), 255))
			}
			img.Set(x, y, color.RGBA{c[0], c[1], c[2], 255})
		}
	}
	f, _ := os.Create(path)
	png.Encode(f, img)
	f.Close()
}

func main() {
	// vaeprobe OUTDIR EXPORT FUNC STAGES...
	out, exp, fn := os.Args[1], os.Args[2], os.Args[3]
	ctx := context.Background()
	t0 := time.Now()
	dev, _, err := llm.OpenDevice(ctx, "", "")
	if err != nil {
		panic(err)
	}
	fmt.Printf("device open (NE lock) %.1fs\n", time.Since(t0).Seconds())
	gr, _ := dev.Graph()
	var gs []*hwnpu.Graph
	total := time.Now()
	for _, st := range os.Args[4:] {
		t := time.Now()
		g, err := gr.Compile(ctx, hwnpu.GraphSource{Format: types.GraphFormatNative, Path: filepath.Join(exp, st+".aimodel#"+fn)})
		if err != nil {
			panic(fmt.Sprint(st, ": ", err))
		}
		fmt.Printf("load %s %.1fs cached=%v\n", st, time.Since(t).Seconds(), g.Cached())
		gs = append(gs, g)
	}
	fmt.Printf("load total %.1fs\n", time.Since(total).Seconds())
	rng := rand.New(rand.NewSource(42))
	var prev []byte
	for i, g := range gs {
		r, _ := g.NewRequest()
		var outs []*hwnpu.Buffer
		var outArgs []hwnpu.ArgInfo
		for _, a := range g.Args() {
			b, err := dev.AllocBuffer(a.Bytes)
			if err != nil {
				panic(err)
			}
			if a.Kind == types.ArgInput {
				bs := b.Bytes()
				if prev != nil && len(prev) == len(bs) {
					copy(bs, prev)
				} else if p := os.Getenv("PROBE_INPUT"); p != "" && i == 0 {
					// the first stage's input from a file: another run's output of the stage before
					raw, err := os.ReadFile(p)
					if err != nil || len(raw) != len(bs) {
						panic(fmt.Sprint("PROBE_INPUT: ", err, len(raw), len(bs)))
					}
					copy(bs, raw)
				} else if p := os.Getenv("PROBE_LATENT"); p != "" && i == 0 && len(a.Dims) == 4 && a.Dims[1] == 32 {
					copy(bs, latent(p, a.Dims[3], a.Dims[2]))
					fmt.Printf("  %s input %s %v from %s\n", os.Args[4+i], a.Name, a.Dims, filepath.Base(p))
				} else if p := os.Getenv("PROBE_IMAGE"); p != "" && i == 0 && len(a.Dims) == 4 && a.Dims[1] == 3 {
					copy(bs, picture(p, a.Dims[3], a.Dims[2]))
					fmt.Printf("  %s input %s %v from %s\n", os.Args[4+i], a.Name, a.Dims, filepath.Base(p))
				} else {
					fmt.Printf("  %s input %s %v %v random\n", os.Args[4+i], a.Name, a.Dims, a.DType)
					for j := 0; j+1 < len(bs); {
						v := float32(rng.NormFloat64())
						if a.DType == types.F32 {
							binary.LittleEndian.PutUint32(bs[j:], math.Float32bits(v))
							j += 4
						} else {
							binary.LittleEndian.PutUint16(bs[j:], f16(v))
							j += 2
						}
					}
				}
				b.SyncToDevice(0, a.Bytes)
			} else {
				outs, outArgs = append(outs, b), append(outArgs, a)
			}
			if err := r.Bind(a.Index, b); err != nil {
				panic(err)
			}
		}
		t := time.Now()
		if err := r.Run(ctx); err != nil {
			panic(fmt.Sprint(os.Args[4+i], ": ", err))
		}
		prev = nil
		for k, b := range outs {
			b.SyncFromDevice(0, outArgs[k].Bytes)
			name := fmt.Sprintf("%s/%s.%s.%v.%v.bin", out, os.Args[4+i], outArgs[k].Name, outArgs[k].DType, outArgs[k].Dims)
			os.WriteFile(name, b.Bytes(), 0o644)
			if len(outs) == 1 {
				prev = append([]byte(nil), b.Bytes()...)
			}
			if d := outArgs[k].Dims; i == len(gs)-1 && len(d) == 4 && d[1] == 3 {
				writePNG(out+".png", b.Bytes(), d[3], d[2])
			}
		}
		fmt.Printf("run %s %.2fs\n", os.Args[4+i], time.Since(t).Seconds())
	}
}
