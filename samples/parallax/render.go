package main

import (
	"fmt"
	"image"
	"image/color/palette"
	"image/draw"
	"image/gif"
	"image/png"
	"os"
	"path/filepath"
)

// smaller is the scene at width w (its height in proportion), each pixel between the four of
// the scene's that are nearest; the scene itself when it is no wider.
func (s *Scene) smaller(w int) *Scene {
	if w <= 0 || w >= s.W {
		return s
	}
	h := max(1, s.H*w/s.W)
	out := &Scene{W: w, H: h, Pix: image.NewRGBA(image.Rect(0, 0, w, h)), Near: make([]float32, w*h)}
	if s.Behind != nil {
		out.Behind = s.Behind.smaller(w)
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			sx := (float64(x) + 0.5) * float64(s.W) / float64(w)
			sy := (float64(y) + 0.5) * float64(s.H) / float64(h)
			out.Pix.SetRGBA(x, y, s.colorAt(sx, sy))
			out.Near[y*w+x] = float32(s.nearAt(sx, sy))
		}
	}
	return out
}

// render writes the scene as seen from a head that goes around once, in n frames: an animated
// GIF of width w at out, and the frames at the scene's own size as PNG files in pngDir ("" for
// none). It draws on the CPU (Scene.Render), without a window.
func render(s *Scene, p Params, out string, w, n int, pngDir string) error {
	if pngDir != "" {
		if err := os.MkdirAll(pngDir, 0o755); err != nil {
			return err
		}
	}
	small := s.smaller(w)
	anim := &gif.GIF{}
	frame := image.NewRGBA(image.Rect(0, 0, small.W, small.H))
	full := image.NewRGBA(image.Rect(0, 0, s.W, s.H))
	for i := range n {
		v := sweep(float64(i) / float64(n))
		small.Render(frame, v, p)
		pal := image.NewPaletted(frame.Bounds(), palette.Plan9)
		draw.FloydSteinberg.Draw(pal, frame.Bounds(), frame, image.Point{})
		anim.Image = append(anim.Image, pal)
		anim.Delay = append(anim.Delay, 8) // hundredths of a second: 12 frames a second
		if pngDir != "" {
			s.Render(full, v, p)
			f, err := os.Create(filepath.Join(pngDir, fmt.Sprintf("frame-%03d_%+.2f_%+.2f.png", i, v.X, v.Y)))
			if err != nil {
				return err
			}
			if err := png.Encode(f, full); err != nil {
				f.Close()
				return err
			}
			if err := f.Close(); err != nil {
				return err
			}
		}
	}
	f, err := os.Create(out)
	if err != nil {
		return err
	}
	if err := gif.EncodeAll(f, anim); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "%d frames of %d x %d: %s\n", n, small.W, small.H, out)
	return nil
}
