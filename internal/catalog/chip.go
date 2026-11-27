package catalog

import (
	"slices"
	"strings"
)

// Chips are the chips models.yaml's not_on names: Apple's, by their family (an M1 Pro is an M1).
var Chips = []string{"M1", "M2", "M3", "M4", "M5"}

// ChipName is the machine's chip of Chips ("M1" for an Apple M1 Pro), "" for another; a
// test's own (ActAsChip).
var ChipName = func() string { return chipOf(cpuBrand()) }

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
func Chip() string { return ChipName() }

// ExportChips are the chips whose Core AI exports of a model differ, one standing for each kind:
// the M1, whose chat models have no whole programs (wholeSlower) and read prompts with the one
// that generates where its Neural Engine would hold too much (neuralEngineHold), and the M2 for
// the later ones, which convert alike. A chip's choice that changes what is converted makes it
// a kind of its own here.
var ExportChips = []string{"M1", "M2"}

// ActAsChip makes the process convert as a Mac of chip (of Chips; "" for this machine's) would:
// hwnpu-dev models package makes each chip's Core AI exports on any machine. It returns what
// undoes it.
func ActAsChip(chip string) (undo func()) {
	was := ChipName
	if chip != "" {
		ChipName = func() string { return chip }
	}
	return func() { ChipName = was }
}

// archChips are the chips of the architectures of the Neural Engine, by the first three letters
// of coreai-build's names for them (h13g an M1, h13s its Pro; h18g an M6).
var archChips = map[string]string{"h13": "M1", "h14": "M2", "h15": "M3", "h16": "M4", "h17": "M5", "h18": "M6"}

// TakesExportsOf reports whether a Mac whose Neural Engine is of the architecture arch (h13g)
// takes the Core AI exports made as a Mac of chip converts (ActAsChip): its own chip's, or, when
// its chip is none of ExportChips, those of the last of them (the M2, which stands for the later
// ones). hwnpu-dev models package prebuilds an export for the architectures that take it alone:
// where a chip has an export of its own (an M1's FLUX.2 klein), the others' is not its to load.
func TakesExportsOf(arch, chip string) bool {
	own := archChips[arch[:min(len(arch), 3)]]
	if own == chip {
		return true
	}
	return !slices.Contains(ExportChips, own) && chip == ExportChips[len(ExportChips)-1]
}
