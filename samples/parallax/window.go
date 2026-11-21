//go:build !headless

package main

import (
	_ "embed"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
)

//go:embed parallax.kage
var shaderSource []byte

// game is the window: the picture drawn by the shader from where the view is, 60 times a
// second, whatever the tracker's beat is.
type game struct {
	scene  *Scene
	params Params
	o      windowOptions

	shader *ebiten.Shader
	// pics and nears are the layers' pictures and nearness, from the farthest
	pics, nears []*ebiten.Image
	view        View
	// the parting's step, when the picture is parted beside the window: a small picture of it,
	// its line, how much is done
	inset  *ebiten.Image
	step   string
	total  float64
	active bool
	white  *ebiten.Image // one white pixel, to draw the bar and the frames with
	opened time.Time
	show   bool // the nearness instead of the picture
	tick   int
	dumped int
	status string
}

func runWindow(s *Scene, p Params, o windowOptions) error {
	shader, err := ebiten.NewShader(shaderSource)
	if err != nil {
		return fmt.Errorf("the shader: %w", err)
	}
	g := &game{params: p, o: o, shader: shader, white: ebiten.NewImage(1, 1), opened: time.Now()}
	g.white.Fill(color.White)
	g.take(s)
	w, h := s.W, s.H
	for w > 1280 || h > 900 {
		w, h = w/2, h/2
	}
	ebiten.SetWindowSize(w, h)
	ebiten.SetWindowResizingMode(ebiten.WindowResizingModeEnabled)
	ebiten.SetWindowTitle(o.title)
	if err := ebiten.RunGame(g); err != nil && !errors.Is(err, ebiten.Termination) {
		return err
	}
	return nil
}

// take makes s the scene the window draws: its layers' pictures and nearness go to the GPU, and
// those of the scene before are given back.
func (g *game) take(s *Scene) {
	for _, img := range append(g.pics, g.nears...) {
		img.Deallocate()
	}
	g.scene, g.pics, g.nears = s, nil, nil
	for _, l := range s.layers() {
		g.pics = append(g.pics, ebiten.NewImageFromImage(l.Pix))
		g.nears = append(g.nears, ebiten.NewImageFromImage(l.NearImage()))
	}
}

// rect fills a rectangle of the screen with a colour.
func (g *game) rect(dst *ebiten.Image, x, y, w, h float64, c color.Color) {
	op := &ebiten.DrawImageOptions{}
	op.GeoM.Scale(w, h)
	op.GeoM.Translate(x, y)
	op.ColorScale.ScaleWithColor(c)
	dst.DrawImage(g.white, op)
}

func (g *game) Update() error {
	if pr := g.o.progress; pr != nil {
		// the parting's news: the layers so far, the picture of its step, its line
		scene, inset, fresh, text, total, active := pr.take()
		if scene != nil {
			g.take(scene)
		}
		if fresh {
			if g.inset != nil {
				g.inset.Deallocate()
				g.inset = nil
			}
			if inset != nil {
				g.inset = ebiten.NewImageFromImage(inset)
			}
		}
		g.step, g.total, g.active = text, total, active
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyEscape) || inpututil.IsKeyJustPressed(ebiten.KeyQ) {
		return ebiten.Termination
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyD) {
		g.show = !g.show
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyArrowUp) {
		g.params.Strength = min(g.params.Strength+0.005, 0.1)
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyArrowDown) {
		g.params.Strength = max(g.params.Strength-0.005, 0)
	}
	g.tick++
	var to View
	switch {
	case g.o.dump != "":
		// the views of "parallax render --frames N", to compare the two frame by frame
		to = sweep(float64(g.dumped) / float64(g.o.frames))
		g.view = to
	case g.o.sweep:
		to = sweep(float64(g.tick) / 180) // a turn every 3 seconds
		g.status = "sweep"
	case g.o.target != nil:
		to, g.status = g.o.target()
	default:
		// the pointer stands in for the head
		x, y := ebiten.CursorPosition()
		w, h := ebiten.WindowSize()
		to = View{X: clamp(2*float64(x)/float64(w) - 1), Y: clamp(1 - 2*float64(y)/float64(h))}
		g.status = "pointer"
	}
	if g.o.sweep {
		g.view = to
	} else {
		g.view = g.view.Follow(to, 0.15)
	}
	if g.o.frames > 0 && g.dumped >= g.o.frames {
		return ebiten.Termination
	}
	if g.o.closeAfter > 0 && time.Since(g.opened) > g.o.closeAfter {
		return ebiten.Termination
	}
	return nil
}

// drawScene draws the scene as seen from the view into dst, the picture's pixels scaled by
// scale and moved by (dx, dy): its layers from the farthest, each over the ones before.
func (g *game) drawScene(dst *ebiten.Image, scale, dx, dy float64) {
	vx, vy, shift, focus, zoom := g.params.uniforms(g.view, g.scene.W)
	show := float32(0)
	if g.show {
		show = 1
	}
	for i := range g.pics {
		op := &ebiten.DrawRectShaderOptions{}
		op.GeoM.Scale(scale, scale)
		op.GeoM.Translate(dx, dy)
		op.Images[0], op.Images[1] = g.pics[i], g.nears[i]
		op.Uniforms = map[string]any{
			"View":  []float32{float32(vx), float32(vy)},
			"Shift": float32(shift),
			"Focus": float32(focus),
			"Zoom":  float32(zoom),
			"Show":  show,
		}
		dst.DrawRectShader(g.scene.W, g.scene.H, g.shader, op)
	}
}

// frame draws the window into dst: the picture, and over it the lines of what is going on.
func (g *game) frame(dst *ebiten.Image) {
	sw, sh := dst.Bounds().Dx(), dst.Bounds().Dy()
	scale := min(float64(sw)/float64(g.scene.W), float64(sh)/float64(g.scene.H))
	g.drawScene(dst, scale, (float64(sw)-scale*float64(g.scene.W))/2, (float64(sh)-scale*float64(g.scene.H))/2)
	if !g.o.quiet {
		ebitenutil.DebugPrint(dst, fmt.Sprintf("%s\nview %+.2f %+.2f  strength %.3f (up/down)  D: depth  Q: quit  %.0f fps",
			g.status, g.view.X, g.view.Y, g.params.Strength, ebiten.ActualFPS()))
		g.drawStep(dst)
	}
}

func (g *game) Draw(screen *ebiten.Image) {
	g.frame(screen)
	if g.o.record != "" && g.tick%120 == 1 {
		// the window as it is, every 2 seconds: how the parting went, to look at after
		w, h := screen.Bounds().Dx(), screen.Bounds().Dy()
		off := ebiten.NewImage(w, h)
		g.frame(off)
		img := image.NewRGBA(image.Rect(0, 0, w, h))
		off.ReadPixels(img.Pix)
		off.Deallocate()
		if f, err := os.Create(filepath.Join(g.o.record, fmt.Sprintf("window-%04d.png", g.tick/120))); err == nil {
			png.Encode(f, img)
			f.Close()
		}
	}
	if g.o.dump != "" && g.dumped < g.o.frames {
		// the frame at the picture's own size, as the shader draws it, to compare with Render
		off := ebiten.NewImage(g.scene.W, g.scene.H)
		g.drawScene(off, 1, 0, 0)
		img := image.NewRGBA(image.Rect(0, 0, g.scene.W, g.scene.H))
		off.ReadPixels(img.Pix)
		off.Deallocate()
		name := filepath.Join(g.o.dump, fmt.Sprintf("frame-%03d_%+.2f_%+.2f.png", g.dumped, g.view.X, g.view.Y))
		if f, err := os.Create(name); err == nil {
			png.Encode(f, img)
			f.Close()
		}
		g.dumped++
	}
}

// drawStep draws how the parting of the picture into layers is going, at the bottom of the
// window: the line of the step, a bar of how much is done, and in the corner a small picture of
// the step (the picture with a layer's shape painted over, then what the model drew there).
func (g *game) drawStep(screen *ebiten.Image) {
	if g.o.progress == nil || g.step == "" {
		return
	}
	sw, sh := float64(screen.Bounds().Dx()), float64(screen.Bounds().Dy())
	shade := color.RGBA{0, 0, 0, 170}
	if g.inset != nil {
		// the corner picture, a quarter of the window wide, in a frame
		iw, ih := float64(g.inset.Bounds().Dx()), float64(g.inset.Bounds().Dy())
		scale := sw / 4 / iw
		x, y := sw-iw*scale-10, sh-ih*scale-44
		g.rect(screen, x-2, y-2, iw*scale+4, ih*scale+4, color.White)
		op := &ebiten.DrawImageOptions{Filter: ebiten.FilterLinear}
		op.GeoM.Scale(scale, scale)
		op.GeoM.Translate(x, y)
		screen.DrawImage(g.inset, op)
	}
	g.rect(screen, 0, sh-34, sw, 34, shade)
	ebitenutil.DebugPrintAt(screen, g.step, 8, int(sh)-30)
	if g.active {
		// the bar: its ground, and how much is done
		g.rect(screen, 8, sh-10, sw-16, 5, color.RGBA{90, 90, 90, 255})
		g.rect(screen, 8, sh-10, (sw-16)*min(max(g.total, 0), 1), 5, color.RGBA{90, 200, 120, 255})
	}
}

func (g *game) Layout(w, h int) (int, int) { return w, h }
