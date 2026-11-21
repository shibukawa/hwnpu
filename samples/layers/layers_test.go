package main

import (
	"bytes"
	"encoding/binary"
	"image"
	"image/color"
	"math"
	"testing"
)

func TestCuts(t *testing.T) {
	// three groups of values: the cuts go in the middle of the gaps between them
	var v []float32
	for i := 0; i < 300; i++ {
		v = append(v, 0.02+0.0001*float32(i))
	}
	for i := 0; i < 100; i++ {
		v = append(v, 0.40+0.001*float32(i))
	}
	for i := 0; i < 200; i++ {
		v = append(v, 0.85+0.0005*float32(i))
	}
	got := cuts(v, 3)
	if len(got) != 2 || !(got[0] > 0.15 && got[0] < 0.30) || !(got[1] > 0.60 && got[1] < 0.75) {
		t.Errorf("cuts = %v, want one between the groups each", got)
	}
	if one := cuts(v, 2); len(one) != 1 || !(one[0] > 0.06 && one[0] < 0.85) {
		t.Errorf("a cut in two = %v", one)
	}
	if none := cuts(v, 1); len(none) != 0 {
		t.Errorf("one class has cuts: %v", none)
	}
}

func TestCurveAndFit(t *testing.T) {
	c := curve([]float32{0, 0.1, 1, 2, -1}, 30)
	if c[0] != 0 || math.Abs(float64(c[1])-0.4037) > 1e-3 || math.Abs(float64(c[2])-1) > 1e-6 || c[3] != c[2] || c[4] != 0 {
		t.Errorf("curve = %v", c)
	}
	// y = 0.5 x + 0.1 outside the mask; inside it y is something else, which is not looked at
	x := make([]float32, 100)
	y := make([]float32, 100)
	mask := make([]float32, 100)
	for i := range x {
		x[i] = float32(i) / 100
		y[i] = 0.5*x[i] + 0.1
		if i >= 60 {
			mask[i], y[i] = 1, 9
		}
	}
	if a, b := fit(x, y, mask); math.Abs(a-0.5) > 1e-4 || math.Abs(b-0.1) > 1e-4 {
		t.Errorf("fit = %v, %v, want 0.5, 0.1", a, b)
	}
	if a, b := fit(x[:4], y[:4], mask[:4]); a != 1 || b != 0 {
		t.Errorf("a fit of too few points = %v, %v", a, b)
	}
}

func TestMasks(t *testing.T) {
	const w, h = 9, 5
	m := make([]float32, w*h)
	m[2*w+4] = 1
	g := grow(m, w, h, 1)
	if g[2*w+3] != 1 || g[1*w+5] != 1 || g[2*w+2] != 0 || covered(g) != 9.0/45 {
		t.Errorf("grown by 1: %v", g)
	}
	s := soften(g, w, h, 1)
	if !(s[2*w+4] == 1 && s[2*w+2] > 0 && s[2*w+2] < 1 && s[2*w+1] == 0) {
		t.Errorf("softened: %v", s[2*w:3*w])
	}
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for i := 0; i < len(img.Pix); i += 4 {
		img.Pix[i], img.Pix[i+1], img.Pix[i+2], img.Pix[i+3] = 10, 250, 10, 255 // a green picture
	}
	c, word := key(img)
	if word == "green" {
		t.Errorf("the key of a green picture is green")
	}
	p := paint(img, m, c)
	if got := p.RGBAAt(4, 2); got != c {
		t.Errorf("painted: %v, want %v", got, c)
	}
	if got := p.RGBAAt(0, 0); got != (color.RGBA{10, 250, 10, 255}) {
		t.Errorf("outside the mask: %v", got)
	}
	half := make([]float32, w*h)
	for i := range half {
		half[i] = 0.5
	}
	if got := mix(img, p, half).RGBAAt(4, 2); got.R != uint8((10+int(c.R)+1)/2) {
		t.Errorf("mixed: %v", got)
	}
	l := cutOut(img, s)
	if l.NRGBAAt(4, 2).A != 255 || l.NRGBAAt(0, 0).A != 0 || l.NRGBAAt(4, 2).G != 250 {
		t.Errorf("cut out: %v %v", l.NRGBAAt(4, 2), l.NRGBAAt(0, 0))
	}
	if st := stretch(img, 18, 10); st.Rect.Dx() != 18 || st.Rect.Dy() != 10 || st.RGBAAt(9, 5).G < 245 {
		t.Errorf("stretched: %v %v", st.Rect, st.RGBAAt(9, 5))
	}
}

// unpack is PackBits undone, n bytes of it.
func unpack(t *testing.T, b []byte, n int) []byte {
	var out []byte
	for i := 0; i < len(b); {
		c := int(int8(b[i]))
		i++
		switch {
		case c >= 0:
			out = append(out, b[i:i+c+1]...)
			i += c + 1
		case c != -128:
			for k := 0; k < 1-c; k++ {
				out = append(out, b[i])
			}
			i++
		}
	}
	if len(out) != n {
		t.Fatalf("unpacked %d bytes, want %d", len(out), n)
	}
	return out
}

func TestPackBits(t *testing.T) {
	rows := [][]byte{
		{}, {7}, {7, 7}, {1, 2, 3}, {1, 2, 2, 3}, {1, 2, 2, 2, 3}, bytes.Repeat([]byte{9}, 300),
		append(bytes.Repeat([]byte{1, 2}, 100), bytes.Repeat([]byte{5}, 129)...),
	}
	for _, row := range rows {
		p := packBits(nil, row)
		if got := unpack(t, p, len(row)); !bytes.Equal(got, row) {
			t.Errorf("%v packed to %v unpacks to %v", row, p, got)
		}
	}
	if p := packBits(nil, bytes.Repeat([]byte{9}, 300)); len(p) != 6 {
		t.Errorf("300 equal bytes pack to %d bytes", len(p))
	}
}

// TestWritePSD reads the document back section by section: the header, the layers' records and
// their channels, and the merged picture.
func TestWritePSD(t *testing.T) {
	const w, h = 6, 4
	merged := image.NewRGBA(image.Rect(0, 0, w, h))
	back := image.NewNRGBA(image.Rect(0, 0, w, h))
	front := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			merged.SetRGBA(x, y, color.RGBA{uint8(10 * x), uint8(20 * y), 7, 255})
			back.SetNRGBA(x, y, color.NRGBA{uint8(x), uint8(y), 200, 255})
			if x >= 2 && x < 4 {
				front.SetNRGBA(x, y, color.NRGBA{250, uint8(x + y), 1, 128})
			}
		}
	}
	var buf bytes.Buffer
	if err := WritePSD(&buf, merged, []Layer{{"1 far", back}, {"2 手前", front}}); err != nil {
		t.Fatal(err)
	}
	b := buf.Bytes()
	be := binary.BigEndian
	if string(b[:4]) != "8BPS" || be.Uint16(b[4:]) != 1 || be.Uint16(b[12:]) != 3 || be.Uint32(b[14:]) != h || be.Uint32(b[18:]) != w ||
		be.Uint16(b[22:]) != 8 || be.Uint16(b[24:]) != 3 {
		t.Fatalf("the header: % x", b[:26])
	}
	at := 26
	if be.Uint32(b[at:]) != 0 || be.Uint32(b[at+4:]) != 0 {
		t.Fatalf("the colour mode data and the resources are not empty")
	}
	at += 8
	section := int(be.Uint32(b[at:]))
	info := int(be.Uint32(b[at+4:]))
	if info%2 != 0 || section != info+8 {
		t.Fatalf("the layers' section is %d bytes, its layers %d", section, info)
	}
	end := at + 4 + section
	at += 8
	if n := int16(be.Uint16(b[at:])); n != 2 {
		t.Fatalf("%d layers", n)
	}
	at += 2
	type record struct {
		name  string
		sizes [4]int
	}
	var recs []record
	for l := 0; l < 2; l++ {
		if be.Uint32(b[at:]) != 0 || be.Uint32(b[at+4:]) != 0 || be.Uint32(b[at+8:]) != h || be.Uint32(b[at+12:]) != w || be.Uint16(b[at+16:]) != 4 {
			t.Fatalf("layer %d's rectangle and channels: % x", l, b[at:at+18])
		}
		at += 18
		var r record
		for k, id := range []int16{-1, 0, 1, 2} {
			if int16(be.Uint16(b[at:])) != id {
				t.Fatalf("layer %d's channel %d is %d", l, k, int16(be.Uint16(b[at:])))
			}
			r.sizes[k] = int(be.Uint32(b[at+2:]))
			at += 6
		}
		if string(b[at:at+8]) != "8BIMnorm" || b[at+8] != 255 {
			t.Fatalf("layer %d's blending: %q", l, b[at:at+12])
		}
		at += 12
		extra := int(be.Uint32(b[at:]))
		at += 4
		x := b[at : at+extra]
		if be.Uint32(x) != 0 || be.Uint32(x[4:]) != 0 {
			t.Fatalf("layer %d has a mask or blending ranges", l)
		}
		n := int(x[8])
		r.name = string(x[9 : 9+n])
		rest := x[8+(1+n+3)/4*4:]
		if string(rest[:8]) != "8BIMluni" || int(be.Uint32(rest[8:])) != len(rest)-12 {
			t.Fatalf("layer %d's Unicode name: % x", l, rest)
		}
		recs = append(recs, r)
		at += extra
	}
	if recs[0].name != "1 far" || recs[1].name != "2 ??" {
		t.Errorf("the names: %q %q", recs[0].name, recs[1].name)
	}
	for l, img := range []*image.NRGBA{back, front} {
		for k, c := range []int{3, 0, 1, 2} {
			if be.Uint16(b[at:]) != 1 {
				t.Fatalf("layer %d channel %d is not packed", l, k)
			}
			data := b[at+2 : at+recs[l].sizes[k]]
			rows := 0
			for y := 0; y < h; y++ {
				rows += int(be.Uint16(data[2*y:]))
			}
			if rows != len(data)-2*h {
				t.Fatalf("layer %d channel %d: the rows are %d bytes of %d", l, k, rows, len(data)-2*h)
			}
			got := unpack(t, data[2*h:], w*h)
			for i, v := range got {
				if v != img.Pix[4*i+c] {
					t.Fatalf("layer %d channel %d pixel %d = %d, want %d", l, k, i, v, img.Pix[4*i+c])
				}
			}
			at += recs[l].sizes[k]
		}
	}
	at += at % 2 // the layers are padded to an even length; the global mask's length follows
	if at+4 != end || be.Uint32(b[at:]) != 0 {
		t.Fatalf("the layers end at %d, the section at %d", at+4, end)
	}
	at = end
	if be.Uint16(b[at:]) != 1 {
		t.Fatalf("the merged picture is not packed")
	}
	at += 2
	lens := b[at : at+2*3*h]
	at += len(lens)
	for k := 0; k < 3; k++ {
		n := 0
		for y := 0; y < h; y++ {
			n += int(be.Uint16(lens[2*(k*h+y):]))
		}
		got := unpack(t, b[at:at+n], w*h)
		for i, v := range got {
			if v != merged.Pix[4*i+k] {
				t.Fatalf("the merged picture's channel %d pixel %d = %d, want %d", k, i, v, merged.Pix[4*i+k])
			}
		}
		at += n
	}
	if at != len(b) {
		t.Errorf("%d bytes after the merged picture", len(b)-at)
	}
}

func TestSeam(t *testing.T) {
	// a picture of stripes; a square of it is the mask (its edges inside the blocks seam looks at)
	const w, h = 256, 256
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	m := make([]float32, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetRGBA(x, y, color.RGBA{uint8(40 + 20*(y/8%2)), 120, 200, 255})
			if x >= 67 && x < 189 && y >= 67 && y < 189 {
				m[y*w+x] = 1
			}
		}
	}
	if s := seam(img, m, w, h); s > 10 {
		t.Errorf("a picture that goes on across the mask's edge has a seam of %v", s)
	}
	if s := seam(paint(img, m, color.RGBA{0, 60, 20, 255}), m, w, h); s < 80 {
		t.Errorf("a flat shape in the mask has a seam of %v", s)
	}
	if s := seam(img, make([]float32, w*h), w, h); s != 0 {
		t.Errorf("no mask has a seam of %v", s)
	}
}
