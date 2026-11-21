//go:build !headless

package main

import (
	"bytes"
	_ "embed"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/hajimehoshi/ebiten/v2/inpututil"
	"github.com/hajimehoshi/ebiten/v2/text/v2"
	"github.com/shibukawa/hwnpu/ai"
	"golang.org/x/image/font/gofont/goregular"

	"github.com/shibukawa/hwnpu/ai/vision"
)

//go:embed parallax.kage
var shaderSource []byte

// game is the window. While the picture is made deep (o.progress) its main area shows each
// step's result: the depth, then for each layer the layer lifted out and the picture with what
// it hid drawn in, and last the layers put together; the picture itself and the camera's view are small in the corner, and a bar at the
// bottom goes to 100 % when it is all done. Then it is the picture drawn by the shader from where
// the view is, 60 times a second, whatever the tracker's beat is.
type game struct {
	scene  *Scene
	params Params
	o      windowOptions

	shader *ebiten.Shader
	// pics and nears are the layers' pictures and nearness, from the farthest
	pics, nears []*ebiten.Image
	view        View
	white       *ebiten.Image // one white pixel, to draw the bar and the frames with
	face        *text.GoTextFaceSource
	opened      time.Time
	show        bool // the nearness instead of the picture
	tick        int
	dumped      int
	status      string
	gain        float64   // the head's gain as last set (o.gain)
	gainAt      time.Time // when it was last changed, to show it a while

	b *making // while the picture is made deep; nil once it is shown
	// done is when the making ended, for the last line to fade
	done     time.Time
	doneText string
}

// making is what the window shows while the picture is made deep.
type making struct {
	w, h      int           // the picture's size
	orig      *ebiten.Image // the picture, in the corner
	queue     []step        // the steps to show after this one
	step      step          // the step shown
	live      news          // the newest of the drawing's line and how far it is
	cur, prev *ebiten.Image // the main area's picture of this step and of the one before
	curStage  stage
	since     time.Time // when this step came
	bar       float64   // the bar as drawn, following the progress
	checker   *ebiten.Image
	// the putting together of the layers, once the scene is ready
	composeAt time.Time
	// the camera's corner picture
	cam    *ebiten.Image
	camSrc image.Image
}

// composeTime is how long the layers take to come together at the end.
const composeTime = 2200 * time.Millisecond

func runWindow(s *Scene, p Params, o windowOptions) error {
	shader, err := ebiten.NewShader(shaderSource)
	if err != nil {
		return fmt.Errorf("the shader: %w", err)
	}
	face, err := text.NewGoTextFaceSource(bytes.NewReader(goregular.TTF))
	if err != nil {
		return err
	}
	g := &game{params: p, o: o, shader: shader, face: face, white: ebiten.NewImage(1, 1), opened: time.Now()}
	g.white.Fill(color.White)
	w, h := 0, 0
	if s != nil {
		g.take(s)
		w, h = s.W, s.H
	}
	if o.progress != nil && o.picture != nil {
		b := o.picture.Bounds()
		w, h = b.Dx(), b.Dy()
		g.b = &making{w: w, h: h, orig: ebiten.NewImageFromImage(o.picture), since: time.Now(), curStage: -1}
	}
	for w > 1280 || h > 900 {
		w, h = w/2, h/2
	}
	ebiten.SetWindowSize(w, h+panelHeight(g.b != nil))
	ebiten.SetWindowResizingMode(ebiten.WindowResizingModeEnabled)
	ebiten.SetWindowTitle(o.title)
	if err := ebiten.RunGame(g); err != nil && !errors.Is(err, ebiten.Termination) {
		return err
	}
	return nil
}

// panelHeight is the height of the panel at the bottom while the picture is made.
func panelHeight(building bool) int {
	if building {
		return 66
	}
	return 0
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

// frameRect draws the outline of a rectangle, t pixels thick.
func (g *game) frameRect(dst *ebiten.Image, x, y, w, h, t float64, c color.Color) {
	g.rect(dst, x, y, w, t, c)
	g.rect(dst, x, y+h-t, w, t, c)
	g.rect(dst, x, y, t, h, c)
	g.rect(dst, x+w-t, y, t, h, c)
}

// sayFit writes a line as say does, in a smaller size down to 11 to fit in width w, and cut
// short with an ellipsis if it still does not.
func (g *game) sayFit(dst *ebiten.Image, s string, x, y, w, size float64, c color.Color) {
	for ; size > 11; size-- {
		if tw, _ := text.Measure(s, &text.GoTextFace{Source: g.face, Size: size}, 0); tw <= w {
			break
		}
	}
	r := []rune(s)
	for len(r) > 1 {
		if tw, _ := text.Measure(string(r), &text.GoTextFace{Source: g.face, Size: size}, 0); tw <= w {
			break
		}
		r = append(r[:len(r)-2], '…')
	}
	g.say(dst, string(r), x, y+(16-size)/2, size, c)
}

// say writes a line at x, y (its top) in the window's font.
func (g *game) say(dst *ebiten.Image, s string, x, y, size float64, c color.Color) {
	op := &text.DrawOptions{}
	op.GeoM.Translate(x, y)
	op.ColorScale.ScaleWithColor(c)
	text.Draw(dst, s, &text.GoTextFace{Source: g.face, Size: size}, op)
}

func (g *game) Update() error {
	if g.b != nil {
		g.updateBuild()
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
	if g.o.gain != nil {
		// how far the view turns for the head's move: the arrow keys left and right, the wheel
		step := 0.0
		if inpututil.IsKeyJustPressed(ebiten.KeyArrowRight) {
			step += 0.5
		}
		if inpututil.IsKeyJustPressed(ebiten.KeyArrowLeft) {
			step -= 0.5
		}
		if _, wy := ebiten.Wheel(); wy != 0 {
			step += 0.25 * wy
		}
		if step != 0 || g.gain == 0 {
			g.gain = g.o.gain(step)
			if step != 0 {
				g.gainAt = time.Now()
			}
		}
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

// updateBuild takes the making's news and goes on to the next step queued once this one was
// shown its while (minShow), and takes the camera's newest frame.
func (g *game) updateBuild() {
	b := g.b
	n := g.o.progress.take()
	b.queue = append(b.queue, n.steps...)
	b.live = n
	if len(b.queue) > 0 && (b.curStage < 0 || time.Since(b.since) >= minShow(b.curStage)) {
		g.next()
	}
	target := b.step.total
	if b.curStage == stageDrawing {
		target = n.total
	}
	if !b.composeAt.IsZero() {
		t := min(time.Since(b.composeAt).Seconds()/composeTime.Seconds(), 1)
		target = b.step.total + n.wCompose*t
		if t >= 1 {
			// put together: the picture is shown deep, without the corner pictures
			g.b, g.done, g.doneText = nil, time.Now(), "Done: move your head (or the pointer) before the picture"
			if g.o.target == nil && !g.o.sweep {
				g.doneText = "Done: move the pointer over the picture"
			}
			w, h := ebiten.WindowSize()
			ebiten.SetWindowSize(w, h-panelHeight(true))
			return
		}
	}
	b.bar += (target - b.bar) * 0.12
	if g.o.camera != nil {
		if frame, _, _, _ := g.o.camera(); frame != nil && frame != b.camSrc {
			fb := frame.Bounds()
			if b.cam == nil || b.cam.Bounds().Dx() != fb.Dx() || b.cam.Bounds().Dy() != fb.Dy() {
				if b.cam != nil {
					b.cam.Deallocate()
				}
				b.cam = ebiten.NewImage(fb.Dx(), fb.Dy())
			}
			b.cam.WritePixels(toRGBA(frame).Pix)
			b.camSrc = frame
		}
	}
}

// next shows the next step queued: its picture over the one before, the scene to put together.
func (g *game) next() {
	b := g.b
	st := b.queue[0]
	b.queue = b.queue[1:]
	b.step, b.curStage, b.since = st, st.stage, time.Now()
	if st.pic != nil {
		if b.prev != nil && b.prev != b.cur {
			b.prev.Deallocate()
		}
		b.prev, b.cur = b.cur, ebiten.NewImageFromImage(st.pic)
	}
	if st.scene != nil {
		g.take(st.scene)
		b.composeAt = time.Now()
	}
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

// frame draws the window into dst.
func (g *game) frame(dst *ebiten.Image) {
	if g.b != nil {
		g.drawBuild(dst)
		return
	}
	sw, sh := dst.Bounds().Dx(), dst.Bounds().Dy()
	scale := min(float64(sw)/float64(g.scene.W), float64(sh)/float64(g.scene.H))
	g.drawScene(dst, scale, (float64(sw)-scale*float64(g.scene.W))/2, (float64(sh)-scale*float64(g.scene.H))/2)
	if !g.o.quiet {
		gain := ""
		if g.o.gain != nil {
			gain = fmt.Sprintf("  head gain %.2f (left/right, wheel)", g.gain)
		}
		ebitenutil.DebugPrint(dst, fmt.Sprintf("%s\nview %+.2f %+.2f  strength %.3f (up/down)%s  D: depth  Q: quit  %.0f fps",
			g.status, g.view.X, g.view.Y, g.params.Strength, gain, ebiten.ActualFPS()))
		if since := time.Since(g.gainAt); !g.gainAt.IsZero() && since < 1500*time.Millisecond {
			// the gain, large for a moment after it changes
			a := uint8(255 * min(1, (1.5-since.Seconds())/0.5))
			g.say(dst, fmt.Sprintf("head gain %.2f", g.gain), float64(sw)/2-90, float64(sh)/2-20, 28, color.RGBA{a, a, a, a})
		}
		if since := time.Since(g.done); !g.done.IsZero() && since < 4*time.Second {
			a := uint8(255 * min(1, (4-since.Seconds())/1.5))
			g.rect(dst, 0, float64(sh)-40, float64(sw), 40, color.RGBA{0, 0, 0, a * 2 / 3})
			g.say(dst, g.doneText, 14, float64(sh)-31, 18, color.RGBA{a, a, a, a})
		}
	}
}

// fit is where a picture of w x h goes in the area (x, y, aw, ah): its scale and its top left.
func fit(w, h int, x, y, aw, ah float64) (scale, dx, dy float64) {
	scale = min(aw/float64(w), ah/float64(h))
	return scale, x + (aw-scale*float64(w))/2, y + (ah-scale*float64(h))/2
}

// drawPic draws a picture scaled by scale about the centre of where fit puts it, with alpha.
func drawPic(dst, img *ebiten.Image, scale, dx, dy, grow, alpha float64) {
	if img == nil {
		return
	}
	w, h := float64(img.Bounds().Dx()), float64(img.Bounds().Dy())
	op := &ebiten.DrawImageOptions{Filter: ebiten.FilterLinear}
	op.GeoM.Translate(-w/2, -h/2)
	op.GeoM.Scale(scale*grow, scale*grow)
	op.GeoM.Translate(dx+w*scale/2, dy+h*scale/2)
	op.ColorScale.ScaleAlpha(float32(alpha))
	dst.DrawImage(img, op)
}

// ease goes from 0 to 1 smoothly as t does.
func ease(t float64) float64 {
	t = min(max(t, 0), 1)
	return t * t * (3 - 2*t)
}

// drawBuild draws the making of the deep picture: the step in the main area, the corner
// pictures, and the panel with its line and the bar of the whole.
func (g *game) drawBuild(dst *ebiten.Image) {
	b := g.b
	sw, sh := float64(dst.Bounds().Dx()), float64(dst.Bounds().Dy())
	panel := float64(panelHeight(true))
	dst.Fill(color.RGBA{18, 20, 24, 255})
	scale, dx, dy := fit(b.w, b.h, 0, 0, sw, sh-panel)
	t := time.Since(b.since).Seconds()
	in := ease(t / 0.4) // a new picture fades in over 0.4 s
	switch {
	case !b.composeAt.IsZero():
		g.drawCompose(dst, scale, dx, dy)
	case b.curStage == stageDepth:
		// the picture, dim and still, while the depth model loads and estimates: nothing moves
		// that does not move with the work
		drawPic(dst, b.cur, scale, dx, dy, 1, 0.35)
	case b.curStage == stageCut || b.curStage == stageDrawing:
		// the layer lifted out of the picture, on a checkerboard where it has nothing; it stays
		// while the picture model draws what it hid (the panel says how far)
		g.drawChecker(dst, dx, dy, float64(b.w)*scale, float64(b.h)*scale)
		drawPic(dst, b.cur, scale, dx, dy, 1, in)
	default:
		// the depth, what the model drew: each fading in over the one before
		if in < 1 {
			drawPic(dst, b.prev, scale, dx, dy, 1, 1)
		}
		drawPic(dst, b.cur, scale, dx, dy, 1, in)
		if b.curStage == stageNear {
			g.drawLegend(dst, dx+12, dy+12, in)
		}
	}
	if b.composeAt.IsZero() {
		g.drawCorner(dst, sw, sh-panel)
	}
	// the panel: the step's line, and the bar of the whole
	g.rect(dst, 0, sh-panel, sw, panel, color.RGBA{10, 11, 14, 255})
	line := b.step.text
	if b.curStage == stageDrawing {
		line = b.live.text // the drawing as it goes
	}
	if !b.composeAt.IsZero() {
		line = "Putting the layers together"
	}
	g.sayFit(dst, line, 14, sh-panel+8, sw-100, 16, color.RGBA{230, 232, 236, 255})
	if l := b.live.load; l.use != "" && b.composeAt.IsZero() {
		g.drawLoading(dst, l, sw, sh-panel+31)
	}
	pct := fmt.Sprintf("%3.0f %%", 100*min(max(b.bar, 0), 1))
	g.say(dst, pct, sw-64, sh-panel+8, 16, color.RGBA{150, 210, 255, 255})
	g.rect(dst, 14, sh-16, sw-28, 6, color.RGBA{50, 54, 62, 255})
	g.rect(dst, 14, sh-16, (sw-28)*min(max(b.bar, 0), 1), 6, color.RGBA{90, 180, 255, 255})
}

// drawLoading draws the second line of the panel while a model is made ready: what is done to
// it, the time it has taken, and when its load took a known time the last time here, a bar of
// the time gone over that (slowing toward its end, never full, as hwnpu's load bars do) and the
// time left; then the newest line of its log (a download's share and bytes).
func (g *game) drawLoading(dst *ebiten.Image, l loading, sw, y float64) {
	what := map[string]string{"depth": "the depth model", "image": "the picture model", "face-detect": "the face model"}[l.use]
	if what == "" {
		what = l.use
	}
	doing := "Loading " + what
	switch l.stage {
	case stagePhase:
		doing = strings.ToUpper(what[:1]) + what[1:] + ": " + l.phase
	case ai.StagePrepare:
		doing = "Downloading and converting " + what + " (its first use here)"
	case ai.StageCompile:
		doing = "Compiling " + what + " for this machine's NPU (its first load here)"
	}
	gone := time.Since(l.since)
	s := fmt.Sprintf("%s (%s): %s", doing, l.model, clock(gone))
	barW := 0.0
	if l.expected >= time.Second && l.stage != ai.StagePrepare {
		if left := l.expected - gone; left > time.Second {
			s += fmt.Sprintf(", about %s left", clock(left))
		} else {
			s += fmt.Sprintf(", longer than the %s it took the last time", clock(l.expected))
		}
		barW = 120
	}
	if l.line != "" {
		s += " · " + l.line
	}
	gray := color.RGBA{150, 156, 166, 255}
	g.sayFit(dst, s, 14, y, sw-28-barW-max(barW/10, 0), 13, gray)
	if barW > 0 {
		x := sw - 14 - barW
		g.rect(dst, x, y+6, barW, 4, color.RGBA{50, 54, 62, 255})
		g.rect(dst, x, y+6, barW*pseudo(gone.Seconds()/l.expected.Seconds()), 4, color.RGBA{150, 156, 166, 255})
	}
}

// pseudo is the share shown of a load after x of the time it took the last time: the time
// itself up to 0.8, then ever slower toward 1 (0.9 when that time is over, 0.97 at twice), so
// that a load that takes longer still moves and never shows as done.
func pseudo(x float64) float64 {
	if x <= 0.8 {
		return max(x, 0)
	}
	return 1 - 0.04/(x-0.6)
}

// clock is a time as the panel shows it: 42s, 3m05s, 1h02m.
func clock(d time.Duration) string {
	d = d.Round(time.Second)
	switch {
	case d < time.Minute:
		return fmt.Sprintf("%ds", int(d.Seconds()))
	case d < time.Hour:
		return fmt.Sprintf("%dm%02ds", int(d.Minutes()), int(d.Seconds())%60)
	}
	return fmt.Sprintf("%dh%02dm", int(d.Hours()), int(d.Minutes())%60)
}

// drawCompose draws the layers coming together: the farthest is there as it is, and each
// nearer one drops in after the one behind it, from nearer the viewer (larger, with its shadow
// under it), and settles in its place, where the deep picture takes over.
func (g *game) drawCompose(dst *ebiten.Image, scale, dx, dy float64) {
	b := g.b
	t := time.Since(b.composeAt).Seconds() / composeTime.Seconds()
	n := len(g.pics)
	w, h := float64(b.w), float64(b.h)
	for i, pic := range g.pics {
		if i == 0 {
			drawPic(dst, pic, scale, dx, dy, 1, 1)
			continue
		}
		// the layers but the farthest come in turn over the first 80 % of the time
		start := 0.1 + 0.6*float64(i-1)/float64(max(n-1, 1))
		u := ease((t - start) / 0.35)
		if u <= 0 {
			continue
		}
		lift := 1 + 0.18*(1-u) // from nearer the viewer
		place := func(op *ebiten.DrawImageOptions, ox, oy float64) {
			op.GeoM.Translate(-w/2, -h/2)
			op.GeoM.Scale(scale*lift, scale*lift)
			op.GeoM.Translate(dx+w*scale/2+ox, dy+h*scale/2+oy)
		}
		shadow := &ebiten.DrawImageOptions{Filter: ebiten.FilterLinear}
		place(shadow, 14*(1-u), 18*(1-u))
		shadow.ColorScale.Scale(0, 0, 0, float32(0.5*(1-u)*min(u*4, 1)))
		dst.DrawImage(pic, shadow)
		op := &ebiten.DrawImageOptions{Filter: ebiten.FilterLinear}
		place(op, 0, 0)
		op.ColorScale.ScaleAlpha(float32(min(u*2, 1)))
		dst.DrawImage(pic, op)
	}
}

// drawLegend draws the depth's colours from far to near at x, y.
func (g *game) drawLegend(dst *ebiten.Image, x, y, alpha float64) {
	const w, h = 180.0, 12.0
	a := uint8(255 * alpha)
	g.rect(dst, x-8, y-6, w+16, h+38, color.RGBA{0, 0, 0, uint8(150 * alpha)})
	for i := 0; i < int(w); i++ {
		v := float32(i) / float32(w-1)
		c := depthColors([]float32{v}, 1, 1).Pix
		g.rect(dst, x+float64(i), y, 1, h, color.RGBA{c[0], c[1], c[2], a})
	}
	g.say(dst, "far", x, y+h+4, 13, color.RGBA{a, a, a, a})
	g.say(dst, "near", x+w-30, y+h+4, 13, color.RGBA{a, a, a, a})
}

// drawChecker fills a rectangle with the grey squares of a transparent picture's ground.
func (g *game) drawChecker(dst *ebiten.Image, x, y, w, h float64) {
	b := g.b
	if b.checker == nil {
		b.checker = ebiten.NewImage(2, 2)
		b.checker.WritePixels([]byte{60, 62, 68, 255, 44, 46, 52, 255, 44, 46, 52, 255, 60, 62, 68, 255})
	}
	op := &ebiten.DrawImageOptions{}
	op.GeoM.Scale(16, 16)
	sub := dst.SubImage(image.Rect(int(x), int(y), int(x+w), int(y+h))).(*ebiten.Image)
	for yy := y; yy < y+h; yy += 32 {
		for xx := x; xx < x+w; xx += 32 {
			o := *op
			o.GeoM.Translate(xx, yy)
			sub.DrawImage(b.checker, &o)
		}
	}
}

// drawCorner draws the corner pictures above the panel, at the right: the picture itself, and
// left of it the camera's view with what the detector found.
func (g *game) drawCorner(dst *ebiten.Image, sw, bottom float64) {
	b := g.b
	tw := math.Round(sw / 5)
	x := sw - tw - 12
	label := color.RGBA{220, 222, 226, 255}
	scale := tw / float64(b.w)
	th := float64(b.h) * scale
	y := bottom - th - 12
	g.rect(dst, x-2, y-2, tw+4, th+4, color.RGBA{230, 230, 230, 255})
	drawPic(dst, b.orig, scale, x, y, 1, 1)
	g.rect(dst, x, y, tw, 20, color.RGBA{0, 0, 0, 150})
	g.say(dst, "the picture", x+6, y+3, 12, label)
	if g.o.camera == nil || b.cam == nil {
		return
	}
	_, found, head, ok := g.o.camera()
	if !ok && len(found.Faces) > 0 { // found in this frame, before the tracker has smoothed it
		head, ok = vision.NearestHead(found.Faces, found.Frame, vision.HeadOptions{})
	}
	cw, ch := float64(b.cam.Bounds().Dx()), float64(b.cam.Bounds().Dy())
	ctw := math.Round(tw * 1.6) // the camera larger: the face's points are to be seen
	cs := ctw / cw
	cth := ch * cs
	cx, cy := x-ctw-14, bottom-cth-12
	tw = ctw
	g.rect(dst, cx-2, cy-2, tw+4, cth+4, color.RGBA{230, 230, 230, 255})
	// mirrored, as a mirror shows the viewer
	op := &ebiten.DrawImageOptions{Filter: ebiten.FilterLinear}
	op.GeoM.Scale(-cs, cs)
	op.GeoM.Translate(cx+tw, cy)
	dst.DrawImage(b.cam, op)
	fb := found.Frame
	if fb.Dx() > 0 {
		fs := tw / float64(fb.Dx())
		px := func(p vision.Point) (float64, float64) {
			return cx + tw - (p.X-float64(fb.Min.X))*fs, cy + (p.Y-float64(fb.Min.Y))*fs
		}
		for i, f := range found.Faces {
			c := color.RGBA{90, 220, 140, 255}
			if i > 0 {
				c = color.RGBA{200, 200, 90, 255}
			}
			x0, y0 := px(vision.Point{X: float64(f.Box.Max.X), Y: float64(f.Box.Min.Y)})
			x1, y1 := px(vision.Point{X: float64(f.Box.Min.X), Y: float64(f.Box.Max.Y)})
			g.frameRect(dst, x0, y0, x1-x0, y1-y0, 2, c)
			for _, p := range []vision.Point{f.RightEye, f.LeftEye, f.Nose, f.MouthRight, f.MouthLeft} {
				lx, ly := px(p)
				g.rect(dst, lx-2, ly-2, 4, 4, color.RGBA{255, 120, 90, 255})
			}
			g.say(dst, fmt.Sprintf("%.2f", f.Score), x0+3, y1-16, 11, c)
		}
	}
	g.rect(dst, cx, cy, tw, 20, color.RGBA{0, 0, 0, 150})
	if ok {
		g.say(dst, fmt.Sprintf("head %+.2f %+.2f, %.2f m away", head.X, head.Y, head.Z), cx+6, cy+3, 12, label)
	} else {
		g.say(dst, "camera: looking for a face", cx+6, cy+3, 12, label)
	}
	if g.gain > 0 {
		g.rect(dst, cx, cy+cth-20, tw, 20, color.RGBA{0, 0, 0, 150})
		g.say(dst, fmt.Sprintf("gain %.2f (left/right, wheel)", g.gain), cx+6, cy+cth-17, 12, label)
	}
}

func (g *game) Draw(screen *ebiten.Image) {
	g.frame(screen)
	if g.o.record != "" && g.tick%120 == 1 {
		// the window as it is, every 2 seconds: how the making went, to look at after
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
	if g.o.dump != "" && g.dumped < g.o.frames && g.scene != nil {
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
