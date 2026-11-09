package chat

import (
	"cmp"
	"fmt"
	"io"
	"math"
	"os"
	"slices"
	"strconv"
	"strings"

	"github.com/shibukawa/hwnpu/internal/mem"
)

// Category is a use that names the model recommended for it on the machine (MODEL.md's rows):
// -use translate. Picks go by the machine's memory: the one of the largest Machine the machine
// has. CATEGORY:TAG picks the model of that tag among them (translate:0.8b).
type Category struct {
	Name    string
	Aliases []string
	About   string
	Picks   []CategoryPick
}

// CategoryPick is a model of a category and the least memory of a machine, in GB, it is
// recommended for there.
type CategoryPick struct {
	Machine int
	Model   string // its name in Catalog
}

// Categories are MODEL.md's rows, with its columns of memory: the categories of the definition
// files (models.yaml, LoadModels).
var Categories []Category

// MemoryEnv overrides the machine's memory, in GB, that names without a tag pick by (to pick
// as another machine would, or where the memory cannot be read).
const MemoryEnv = "HWNPU_MEMORY"

// MachineMemory is the machine's memory in GB (GiB, rounded: a 16 GB Mac is 16): HWNPU_MEMORY
// when set, else the physical memory, else 8, the smallest column of MODEL.md.
func MachineMemory() int {
	if v := os.Getenv(MemoryEnv); v != "" {
		if n, err := strconv.Atoi(strings.TrimSuffix(strings.ToLower(strings.TrimSpace(v)), "gb")); err == nil && n > 0 {
			return n
		}
	}
	if b := mem.Physical(); b > 0 {
		return int(math.Round(float64(b) / (1 << 30)))
	}
	return 8
}

// Columns are MODEL.md's columns of machine memory, in GB: the memory tags (8g, 16g, ...).
var Columns = []int{8, 16, 24, 32, 48, 64}

// MemoryTag reads a tag of memory, N g or N gb (8g, 16GB): the model a machine of N GB gets.
func MemoryTag(tag string) (int, bool) {
	t := tagOf(tag)
	num, ok := strings.CutSuffix(t, "gb")
	if !ok {
		if num, ok = strings.CutSuffix(t, "g"); !ok {
			return 0, false
		}
	}
	n, err := strconv.Atoi(strings.TrimSpace(num)) // 8 g, 8 GB too
	return n, err == nil && n > 0
}

// fold is a name as names are compared: in lower case without '-', '_', '.' and spaces, so that
// plamo-3-translate, PLaMo3Translate and plamo3translate are one.
func fold(s string) string {
	return strings.Map(func(r rune) rune {
		switch r {
		case '-', '_', '.', ' ':
			return -1
		}
		return r
	}, strings.ToLower(s))
}

// tagOf is a tag as tags are compared: lower case, dots kept (0.8b and 08b are not the same).
func tagOf(s string) string { return strings.ToLower(strings.TrimSpace(s)) }

// Alias is the model that runs where m is picked: the one Aliases names for it, or m. The
// change is logged.
func Alias(m Model, log io.Writer) Model {
	to, ok := Aliases[m.Name]
	if !ok {
		return m
	}
	if log != nil {
		fmt.Fprintf(log, "%s runs %s (an alias of the definitions)\n", m.Name, to)
	}
	return byName(to)
}

// byName finds a model of the catalog by its exact name.
func byName(name string) Model {
	for _, m := range Catalog {
		if m.Name == name {
			return m
		}
	}
	panic("chat: no model " + name + " in the catalog")
}

// Pick finds the model a name of a family or a category means on a machine of machine GB:
// FAMILY, FAMILY:TAG, CATEGORY or CATEGORY:TAG. A tag is a model's (2b, e4b) or a machine's
// memory (8g, 16gb: the pick of a machine of that memory, MODEL.md's column). Without a tag, it is the model of the largest
// Machine up to the machine's memory (the smallest when none fits). why says how it was picked.
// ok is false when the name is not a family or a category.
func Pick(name string, machine int) (m Model, why string, ok bool, err error) {
	m, why, ok, err = pickModel(name, machine)
	if ok && err == nil {
		if to, aliased := Aliases[m.Name]; aliased {
			why += fmt.Sprintf(", which runs %s (alias)", to)
			m = byName(to)
		}
	}
	return m, why, ok, err
}

func pickModel(name string, machine int) (m Model, why string, ok bool, err error) {
	base, tag, tagged := strings.Cut(name, ":")
	key := fold(base)
	var cands []Model
	kind := "family"
	for _, c := range Categories {
		if fold(c.Name) == key || slices.ContainsFunc(c.Aliases, func(a string) bool { return fold(a) == key }) {
			kind, base = "category", c.Name
			for _, p := range c.Picks {
				m := byName(p.Model)
				m.Machine = p.Machine // the category's column, not the family's
				cands = append(cands, m)
			}
			break
		}
	}
	if cands == nil {
		for _, m := range Catalog {
			// a model without a tag (one like another) is picked by its name, not its family
			if m.Family != "" && m.Tag != "" && fold(m.Family) == key {
				base = m.Family
				cands = append(cands, m)
			}
		}
	}
	if cands == nil {
		return Model{}, "", false, nil
	}
	if gb, ok := MemoryTag(tag); tagged && ok {
		// a tag of memory (8g, 16gb) is a column of MODEL.md's table: the pick of such a machine
		machine, tagged = gb, false
	}
	if tagged {
		var tags []string
		var hits []Model
		for _, c := range cands {
			if tagOf(c.Tag) == tagOf(tag) {
				hits = append(hits, c)
			}
			tags = append(tags, c.Tag)
		}
		switch len(hits) {
		case 1:
			return hits[0], fmt.Sprintf("%s:%s", base, hits[0].Tag), true, nil
		case 0:
			return Model{}, "", true, fmt.Errorf("chat: %s %s has no tag %q (it has %s, or a machine's memory as %s:16g)", kind, base, tag, strings.Join(tags, ", "), base)
		}
		var names []string
		for _, h := range hits {
			names = append(names, h.Family+":"+h.Tag)
		}
		return Model{}, "", true, fmt.Errorf("chat: %s:%s is any of %s: name the family", base, tag, strings.Join(names, ", "))
	}
	// the largest column the machine has; among models of one column the larger
	best, found := cands[0], false
	for _, c := range cands {
		if c.Machine > machine {
			continue
		}
		if !found || c.Machine > best.Machine || c.Machine == best.Machine && c.Memory > best.Memory {
			best, found = c, true
		}
	}
	if !found {
		best = slices.MinFunc(cands, func(a, b Model) int { return cmp.Compare(a.Machine, b.Machine) })
		return best, fmt.Sprintf("%s on a machine of %d GB: %s:%s, the smallest (none is recommended below %d GB)", base, machine, base, best.Tag, best.Machine), true, nil
	}
	why = fmt.Sprintf("%s on a machine of %d GB: %s:%s", base, machine, base, best.Tag)
	var others []string
	for _, c := range cands {
		if c.Name != best.Name {
			others = append(others, base+":"+c.Tag)
		}
	}
	if len(others) > 0 {
		why += fmt.Sprintf(" (%s to choose another, or a machine's memory as %s:16g)", strings.Join(others, ", "), base)
	}
	return best, why, true, nil
}
