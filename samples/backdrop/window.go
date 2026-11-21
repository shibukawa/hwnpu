//go:build !headless

package main

import (
	_ "embed"
	"errors"
	"fmt"
	"image"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
	"golang.org/x/image/draw"
)

//go:embed weigh.kage
var weighSource []byte

//go:embed blur.kage
var blurSource []byte

//go:embed backdrop.kage
var mixSource []byte

// What is put behind the person.
const (
	behindBlur    = "blurred"
	behindPicture = "a picture"
	behindColour  = "a colour"
	behindNone    = "as it is"
)

// game is the window: the newest frame drawn over what is put behind the person, 60 times a
// second, whatever the camera's and the model's beat is.
type game struct {
	o                windowOptions
	weigh, blur, mix *ebiten.Shader

	// the newest frame: its size, which one it is, how many came in the last second
	w, h, seq, persec int
	frame             *ebiten.Image // the frame
	small             *ebiten.Image // its matte at the model's size
	matte             *ebiten.Image // the matte at the frame's size
	weighed           *ebiten.Image // the frame weighed by what is not a person
	half, a, b        *ebiten.Image // the same at half and at a quarter, where it is blurred
	behind            *ebiten.Image // what is put behind the person, at the frame's size
	backdrop          *ebiten.Image // the picture asked for, cut to the frame's shape
	alpha             []byte        // the matte's bytes for the GPU

	modes  []string // what B goes through
	mode   int
	show   bool // the matte instead of the frame
	blurBy float64
	opened time.Time
	tick   int
	failed string
}

func runWindow(o windowOptions) error {
	g := &game{o: o, blurBy: o.blur, show: o.matte, opened: time.Now()}
	for _, s := range []struct {
		to  **ebiten.Shader
		src []byte
	}{{&g.weigh, weighSource}, {&g.blur, blurSource}, {&g.mix, mixSource}} {
		sh, err := ebiten.NewShader(s.src)
		if err != nil {
			return fmt.Errorf("a shader: %w", err)
		}
		*s.to = sh
	}
	// what was asked for first, then the others
	switch {
	case o.picture != nil:
		g.modes = []string{behindPicture, behindBlur, behindNone}
	case o.colour != nil:
		g.modes = []string{behindColour, behindBlur, behindNone}
	default:
		g.modes = []string{behindBlur, behindNone}
	}
	ebiten.SetWindowSize(960, 540)
	ebiten.SetWindowResizingMode(ebiten.WindowResizingModeEnabled)
	ebiten.SetWindowTitle(o.title)
	if err := ebiten.RunGame(g); err != nil && !errors.Is(err, ebiten.Termination) {
		return err
	}
	return nil
}

// size makes the pictures on the GPU for frames of w x h and a matte of mw x mh, giving back
// those of another size.
func (g *game) size(w, h, mw, mh int) {
	if g.frame != nil && g.w == w && g.h == h && g.small.Bounds().Dx() == mw && g.small.Bounds().Dy() == mh {
		return
	}
	for _, img := range []*ebiten.Image{g.frame, g.small, g.matte, g.weighed, g.half, g.a, g.b, g.behind, g.backdrop} {
		if img != nil {
			img.Deallocate()
		}
	}
	first := g.frame == nil
	g.w, g.h = w, h
	g.frame, g.matte, g.weighed, g.behind = ebiten.NewImage(w, h), ebiten.NewImage(w, h), ebiten.NewImage(w, h), ebiten.NewImage(w, h)
	g.small = ebiten.NewImage(mw, mh)
	g.half = ebiten.NewImage(max(1, w/2), max(1, h/2))
	g.a, g.b = ebiten.NewImage(max(1, w/4), max(1, h/4)), ebiten.NewImage(max(1, w/4), max(1, h/4))
	g.alpha = make([]byte, 4*mw*mh)
	g.backdrop = nil
	if g.o.picture != nil {
		g.backdrop = cover(g.o.picture, w, h)
	}
	if first {
		ww, wh := w, h
		for ww > 1280 || wh > 900 {
			ww, wh = ww/2, wh/2
		}
		ebiten.SetWindowSize(ww, wh)
	}
}

func (g *game) Update() error {
	if frame, matte, seq, persec := g.o.newest(); seq != g.seq && frame != nil {
		g.size(frame.Rect.Dx(), frame.Rect.Dy(), matte.W, matte.H)
		g.frame.WritePixels(frame.Pix)
		for i, v := range matte.A {
			a := uint8(min(max(v, 0), 1)*255 + 0.5)
			g.alpha[4*i], g.alpha[4*i+1], g.alpha[4*i+2], g.alpha[4*i+3] = a, a, a, a
		}
		g.small.WritePixels(g.alpha)
		g.seq, g.persec = seq, persec
	}
	if g.o.failed != nil {
		g.failed = g.o.failed()
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyEscape) || inpututil.IsKeyJustPressed(ebiten.KeyQ) {
		return ebiten.Termination
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyB) {
		g.mode = (g.mode + 1) % len(g.modes)
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyM) {
		g.show = !g.show
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyArrowUp) {
		g.blurBy = min(g.blurBy+0.005, 0.1)
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyArrowDown) {
		g.blurBy = max(g.blurBy-0.005, 0)
	}
	g.tick++
	if g.o.closeAfter > 0 && time.Since(g.opened) > g.o.closeAfter {
		return ebiten.Termination
	}
	return nil
}

// stretch draws src over the whole of dst, between its pixels in straight lines, in place of
// what dst had.
func stretch(dst, src *ebiten.Image) {
	op := &ebiten.DrawImageOptions{Filter: ebiten.FilterLinear, Blend: ebiten.BlendCopy}
	op.GeoM.Scale(float64(dst.Bounds().Dx())/float64(src.Bounds().Dx()), float64(dst.Bounds().Dy())/float64(src.Bounds().Dy()))
	dst.DrawImage(src, op)
}

// blurred makes g.behind the frame's own picture of what is behind the person, blurred: the
// frame weighed by what is not a person, made a quarter of its size and blurred there along each
// side, then made the frame's size.
func (g *game) blurred() {
	op := &ebiten.DrawRectShaderOptions{Blend: ebiten.BlendCopy}
	op.Images[0], op.Images[1] = g.frame, g.matte
	g.weighed.DrawRectShader(g.w, g.h, g.weigh, op)
	stretch(g.half, g.weighed)
	stretch(g.a, g.half)
	// The nine samples of a pass reach 4 steps, and n passes along a side reach √n times as
	// far. A step of more than a pixel and a half leaves the samples' pattern in the picture,
	// so a wide blur takes more passes instead (at most 16: they are a quarter of the frame's
	// size, a 16th of its work each).
	reach := g.blurBy * float64(g.w) / 4 / 4 // in steps of one pass, at a quarter of the size
	n := min(max(int(math.Ceil(reach*reach/(1.5*1.5))), 1), 16)
	step := float32(reach / math.Sqrt(float64(n)))
	qw, qh := g.a.Bounds().Dx(), g.a.Bounds().Dy()
	for i := 0; i < n && step > 0; i++ {
		for _, p := range []struct {
			dst, src *ebiten.Image
			step     []float32
		}{{g.b, g.a, []float32{step, 0}}, {g.a, g.b, []float32{0, step}}} {
			op := &ebiten.DrawRectShaderOptions{Blend: ebiten.BlendCopy}
			op.Images[0] = p.src
			op.Uniforms = map[string]any{"Step": p.step}
			p.dst.DrawRectShader(qw, qh, g.blur, op)
		}
	}
	stretch(g.behind, g.a)
}

// cover is img cut to the shape of w x h and made that size, filling it.
func cover(img image.Image, w, h int) *ebiten.Image {
	b := img.Bounds()
	scale := max(float64(w)/float64(b.Dx()), float64(h)/float64(b.Dy()))
	cw, ch := int(float64(w)/scale+0.5), int(float64(h)/scale+0.5)
	crop := image.Rect(0, 0, cw, ch).Add(image.Pt(b.Min.X+(b.Dx()-cw)/2, b.Min.Y+(b.Dy()-ch)/2)).Intersect(b)
	fit := image.NewRGBA(image.Rect(0, 0, w, h))
	draw.CatmullRom.Scale(fit, fit.Bounds(), img, crop, draw.Src, nil)
	return ebiten.NewImageFromImage(fit)
}

// compose draws the frame over what is put behind the person into dst, the frame's pixels
// scaled by scale and moved by (dx, dy).
func (g *game) compose(dst *ebiten.Image, scale, dx, dy float64) {
	stretch(g.matte, g.small)
	behind, weighed := g.behind, float32(0)
	switch g.modes[g.mode] {
	case behindBlur:
		g.blurred()
		weighed = 1
	case behindPicture:
		behind = g.backdrop
	case behindColour:
		g.behind.Fill(*g.o.colour)
	default:
		behind = g.frame
	}
	show := float32(0)
	if g.show {
		show = 1
	}
	op := &ebiten.DrawRectShaderOptions{}
	if g.o.mirror {
		op.GeoM.Scale(-scale, scale)
		op.GeoM.Translate(dx+scale*float64(g.w), dy)
	} else {
		op.GeoM.Scale(scale, scale)
		op.GeoM.Translate(dx, dy)
	}
	op.Images[0], op.Images[1], op.Images[2] = g.frame, g.matte, behind
	op.Uniforms = map[string]any{"Weighed": weighed, "Show": show}
	dst.DrawRectShader(g.w, g.h, g.mix, op)
}

// draw draws the window into dst: the frame, and over it the line of what is going on.
func (g *game) draw(dst *ebiten.Image) {
	if g.frame == nil {
		line := "waiting for the camera's first frame"
		if g.failed != "" {
			line = g.failed
		}
		ebitenutil.DebugPrint(dst, line)
		return
	}
	sw, sh := dst.Bounds().Dx(), dst.Bounds().Dy()
	scale := min(float64(sw)/float64(g.w), float64(sh)/float64(g.h))
	g.compose(dst, scale, (float64(sw)-scale*float64(g.w))/2, (float64(sh)-scale*float64(g.h))/2)
	line := fmt.Sprintf("%dx%d, %d mattes a second", g.w, g.h, g.persec)
	if g.failed != "" {
		line = g.failed
	}
	ebitenutil.DebugPrint(dst, fmt.Sprintf("%s\nB: behind (%s)  M: matte  blur %.3f (up/down)  Q: quit  %.0f fps",
		line, g.modes[g.mode], g.blurBy, ebiten.ActualFPS()))
}

func (g *game) Draw(screen *ebiten.Image) {
	g.draw(screen)
	if g.o.record != "" && g.frame != nil && g.tick%60 == 1 {
		// the window as it is, every second
		w, h := screen.Bounds().Dx(), screen.Bounds().Dy()
		off := ebiten.NewImage(w, h)
		g.draw(off)
		img := image.NewRGBA(image.Rect(0, 0, w, h))
		off.ReadPixels(img.Pix)
		off.Deallocate()
		if f, err := os.Create(filepath.Join(g.o.record, fmt.Sprintf("window-%04d.png", g.tick/60))); err == nil {
			png.Encode(f, img)
			f.Close()
		}
	}
}

func (g *game) Layout(w, h int) (int, int) { return w, h }
