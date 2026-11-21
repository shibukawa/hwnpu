//go:build !headless

package main

import (
	_ "embed"
	"errors"
	"fmt"
	"image"
	"image/png"
	"os"
	"path/filepath"

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
	show        bool // the nearness instead of the picture
	tick        int
	dumped      int
	status      string
}

func runWindow(s *Scene, p Params, o windowOptions) error {
	shader, err := ebiten.NewShader(shaderSource)
	if err != nil {
		return fmt.Errorf("the shader: %w", err)
	}
	g := &game{scene: s, params: p, o: o, shader: shader}
	for _, l := range s.layers() {
		g.pics = append(g.pics, ebiten.NewImageFromImage(l.Pix))
		g.nears = append(g.nears, ebiten.NewImageFromImage(l.NearImage()))
	}
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

func (g *game) Update() error {
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

func (g *game) Draw(screen *ebiten.Image) {
	sw, sh := screen.Bounds().Dx(), screen.Bounds().Dy()
	scale := min(float64(sw)/float64(g.scene.W), float64(sh)/float64(g.scene.H))
	g.drawScene(screen, scale, (float64(sw)-scale*float64(g.scene.W))/2, (float64(sh)-scale*float64(g.scene.H))/2)
	if !g.o.quiet {
		ebitenutil.DebugPrint(screen, fmt.Sprintf("%s\nview %+.2f %+.2f  strength %.3f (up/down)  D: depth  Q: quit  %.0f fps",
			g.status, g.view.X, g.view.Y, g.params.Strength, ebiten.ActualFPS()))
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

func (g *game) Layout(w, h int) (int, int) { return w, h }
