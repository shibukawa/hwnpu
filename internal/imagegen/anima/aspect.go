package anima

import (
	"fmt"
	"strings"
)

// Aspect is an aspect ratio name of --size and the picture's size in pixels it stands for.
type Aspect struct {
	Name          string
	Width, Height int
}

// Aspects are the eight sizes Anima's exports are made for, by their aspect ratio. "1:1s" is the
// small square (512 x 512); the others are 1024 x 1024 or the same area as it (about 1.0 M
// pixels), so that the VAE's middle attention keeps at most 16384 pixels per dimension (the
// latents' (h/8) x (w/8), see llm.NeuralEngineMaxDim).
var Aspects = []Aspect{
	{"1:1", 1024, 1024},
	{"1:1s", 512, 512},
	{"2:3", 832, 1248},
	{"3:4", 880, 1168},
	{"9:16", 752, 1360},
	{"3:2", 1248, 832},
	{"4:3", 1168, 880},
	{"16:9", 1360, 752},
}

// ParseAspect is the picture's size of an aspect ratio name (one of Aspects). Pixels are not
// accepted: the name is what an export is keyed by.
func ParseAspect(name string) (width, height int, err error) {
	for _, a := range Aspects {
		if strings.EqualFold(name, a.Name) {
			return a.Width, a.Height, nil
		}
	}
	names := make([]string, len(Aspects))
	for i, a := range Aspects {
		names[i] = a.Name
	}
	return 0, 0, fmt.Errorf("anima: %q is not an aspect ratio; use one of %s", name, strings.Join(names, ", "))
}

// AspectName is the name of a width x height picture, "" when it is none of Aspects.
func AspectName(width, height int) string {
	for _, a := range Aspects {
		if a.Width == width && a.Height == height {
			return a.Name
		}
	}
	return ""
}
