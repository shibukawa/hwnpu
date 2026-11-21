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

// Chip is the machine's chip of Chips ("M1" for an Apple M1 Pro), "" for another.
func Chip() string { return chipName() }

// ExportChips are the chips whose Core AI exports of a model differ, one standing for each kind:
// the M1, whose chat models have no whole programs (wholeSlower) and read prompts with the one
// that generates where its Neural Engine would hold too much (neuralEngineHold), and the M2 for
// the later ones, which convert alike. A chip's choice that changes what is converted makes it
// a kind of its own here.
var ExportChips = []string{"M1", "M2"}

// ActAsChip makes the process convert as a Mac of chip (of Chips; "" for this machine's) would:
// hwnpu-dev models export makes each chip's Core AI exports on any machine. It returns what
// undoes it.
func ActAsChip(chip string) (undo func()) {
	was := chipName
	if chip != "" {
		chipName = func() string { return chip }
	}
	return func() { chipName = was }
}
