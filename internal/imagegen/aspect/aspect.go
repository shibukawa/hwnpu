// Package aspect is the pictures' sizes of the image models by their aspect ratio, the names
// --size takes (1:1, 1:1s, 2:3, 3:4, 9:16, 3:2, 4:3, 16:9), and the exports are made for.
package aspect

import (
	"fmt"
	"hash/crc32"
	"slices"
	"strconv"
	"strings"
)

// Aspect is an aspect ratio's name and the picture's size in pixels it stands for.
type Aspect struct {
	Name          string
	Width, Height int
}

// All are the eight sizes, by their aspect ratio, of a model whose pictures are multiples of 16
// pixels (Anima, FLUX.2 klein, Chroma). "1:1s" is the small square (512 x 512); the others are
// 1024 x 1024 or the same area as it (about 1.0 M pixels), so that a VAE's middle attention keeps
// at most 16384 pixels per dimension (the latents' (h/8) x (w/8), see llm.NeuralEngineMaxDim).
var All = []Aspect{
	{"1:1", 1024, 1024},
	{"1:1s", 512, 512},
	{"2:3", 832, 1248},
	{"3:4", 880, 1168},
	{"9:16", 752, 1360},
	{"3:2", 1248, 832},
	{"4:3", 1168, 880},
	{"16:9", 1360, 752},
}

// All32 are All for a model whose pictures are multiples of 32 pixels (Sana, Qwen-Image): the
// same ratios as near as multiples of 32 give at about the same area (3:4 exactly 864 x 1152).
var All32 = []Aspect{
	{"1:1", 1024, 1024},
	{"1:1s", 512, 512},
	{"2:3", 832, 1248},
	{"3:4", 864, 1152},
	{"9:16", 768, 1344},
	{"3:2", 1248, 832},
	{"4:3", 1152, 864},
	{"16:9", 1344, 768},
}

// Of are the aspects of a model whose pictures are multiples of multiple pixels (16 or 32).
func Of(multiple int) []Aspect {
	if multiple == 32 {
		return All32
	}
	return All
}

// Names are the aspects' names, comma separated.
func Names() string {
	names := make([]string, len(All))
	for i, a := range All {
		names[i] = a.Name
	}
	return strings.Join(names, ", ")
}

// With are as, the aspects of the same names as overrides replaced by them: a model's own sizes
// for some of the ratios (FLUX.2 klein's 16:9 and 9:16).
func With(as []Aspect, overrides ...Aspect) []Aspect {
	out := slices.Clone(as)
	for _, o := range overrides {
		for i := range out {
			if out[i].Name == o.Name {
				out[i] = o
			}
		}
	}
	return out
}

// Parse is the picture's size of an aspect ratio's name (one of Of(multiple)).
func Parse(name string, multiple int) (width, height int, err error) {
	return ParseIn(name, Of(multiple))
}

// ParseIn is the picture's size of an aspect ratio's name, one of as.
func ParseIn(name string, as []Aspect) (width, height int, err error) {
	for _, a := range as {
		if strings.EqualFold(name, a.Name) {
			return a.Width, a.Height, nil
		}
	}
	return 0, 0, fmt.Errorf("%q is not an aspect ratio; use one of %s", name, Names())
}

// ParseOrSide is Parse, or WIDTHxHEIGHT (multiples of multiple; the sizes an export holds beyond
// the eight, such as 768x1152), or a picture of side x side pixels for a pixel count (the square
// sizes --size took before it took names).
func ParseOrSide(spec string, multiple int) (width, height int, err error) {
	return ParseOrSideIn(spec, Of(multiple), multiple)
}

// ParseOrSideIn is ParseOrSide with the aspect ratios' sizes of as.
func ParseOrSideIn(spec string, as []Aspect, multiple int) (width, height int, err error) {
	if w, h, err := ParseIn(spec, as); err == nil {
		return w, h, nil
	}
	if ws, hs, ok := strings.Cut(strings.ToLower(spec), "x"); ok {
		w, ew := strconv.Atoi(ws)
		h, eh := strconv.Atoi(hs)
		if ew != nil || eh != nil || w <= 0 || h <= 0 || w%multiple != 0 || h%multiple != 0 {
			return 0, 0, fmt.Errorf("--size %s: not WIDTHxHEIGHT in multiples of %d pixels", spec, multiple)
		}
		return w, h, nil
	}
	n, err := strconv.Atoi(spec)
	if err != nil || n <= 0 {
		return 0, 0, fmt.Errorf("--size %s: not an aspect ratio (%s), WIDTHxHEIGHT, nor a pixel count", spec, Names())
	}
	return n, n, nil
}

// Name is the name of a width x height picture among Of(multiple), "" when it is none of them.
func Name(width, height, multiple int) string {
	for _, a := range Of(multiple) {
		if a.Width == width && a.Height == height {
			return a.Name
		}
	}
	return ""
}

// Sizes are the aspects' sizes (width, height) of Of(multiple).
func Sizes(multiple int) [][2]int { return SizesOf(Of(multiple)) }

// SizesOf are the sizes (width, height) of as.
func SizesOf(as []Aspect) [][2]int {
	out := make([][2]int, len(as))
	for i, a := range as {
		out[i] = [2]int{a.Width, a.Height}
	}
	return out
}

// Tag names the sizes of an export in its name: "WxH" for one, "512+1024" for squares and a few,
// and for more than two "8sizes-HASH" (the count and the first 8 hex digits of the sizes' CRC-32),
// so that another set of sizes is another export.
func Tag(sizes [][2]int) string {
	if len(sizes) > 2 {
		var b strings.Builder
		for _, s := range sizes {
			fmt.Fprintf(&b, "%dx%d;", s[0], s[1])
		}
		return fmt.Sprintf("%dsizes-%08x", len(sizes), crc32.ChecksumIEEE([]byte(b.String())))
	}
	if len(sizes) == 1 {
		return fmt.Sprintf("%dx%d", sizes[0][0], sizes[0][1])
	}
	var parts []string
	for _, s := range sizes {
		if s[0] == s[1] {
			parts = append(parts, fmt.Sprint(s[0]))
		} else {
			parts = append(parts, fmt.Sprintf("%dx%d", s[0], s[1]))
		}
	}
	return strings.Join(parts, "+")
}
