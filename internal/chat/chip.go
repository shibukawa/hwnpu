package chat

import (
	"slices"
	"strings"
)

// Chips are the chips models.yaml's not_on names: Apple's, by their family (an M1 Pro is an M1).
var Chips = []string{"M1", "M2", "M3", "M4", "M5"}

// chipName is the machine's chip of Chips ("M1" for an Apple M1 Pro), "" for another.
var chipName = func() string { return chipOf(cpuBrand()) }

// chipOf is the chip of Chips that a CPU's brand ("Apple M1 Pro") names, "" for none.
func chipOf(brand string) string {
	name, ok := strings.CutPrefix(brand, "Apple ")
	if !ok {
		return ""
	}
	name, _, _ = strings.Cut(name, " ")
	if slices.Contains(Chips, name) {
		return name
	}
	return ""
}
