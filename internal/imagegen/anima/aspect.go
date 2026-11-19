package anima

import (
	"fmt"

	"github.com/shibukawa/hwnpu/internal/imagegen/aspect"
)

// Aspect is an aspect ratio name of --size and the picture's size in pixels it stands for.
type Aspect = aspect.Aspect

// Aspects are the eight sizes Anima's exports are made for, by their aspect ratio (aspect.All).
var Aspects = aspect.All

// ParseAspect is the picture's size of an aspect ratio name (one of Aspects). Pixels are not
// accepted: the name is what an export is keyed by.
func ParseAspect(name string) (width, height int, err error) {
	w, h, err := aspect.Parse(name, 16)
	if err != nil {
		err = fmt.Errorf("anima: %w", err)
	}
	return w, h, err
}

// AspectName is the name of a width x height picture, "" when it is none of Aspects.
func AspectName(width, height int) string { return aspect.Name(width, height, 16) }
