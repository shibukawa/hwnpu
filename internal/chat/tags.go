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

	"github.com/shibukawa/hwnpu/internal/catalog"
	"github.com/shibukawa/hwnpu/internal/mem"
)

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
// Machine up to the machine's memory (the smallest when none fits). The model is On the machine
// (of the tag's memory for a tag of memory). why says how it was picked. ok is false when the
// name is not a family or a category.
func Pick(name string, machine int) (m Model, why string, ok bool, err error) {
	if _, tag, tagged := strings.Cut(name, ":"); tagged {
		if gb, isMemory := MemoryTag(tag); isMemory {
			machine = gb
		}
	}
	m, why, ok, err = pickModel(name, machine)
	if ok && err == nil {
		if to, aliased := Aliases[m.Name]; aliased {
			why += fmt.Sprintf(", which runs %s (alias)", to)
			m = byName(to)
		}
		m = m.On(machine)
	}
	return m, why, ok, err
}

// notOnHere is, when m is not picked on this machine's chip (Model.NotOn), the model of its
// family that a machine of machine GB picks instead, with ok; Model{} when it has none. m
// itself, ok false, when m is picked here.
func notOnHere(m Model, machine int) (alt Model, ok bool) {
	chip := catalog.ChipName()
	if chip == "" || !slices.Contains(m.NotOn, chip) {
		return m, false
	}
	found := false
	for _, c := range Catalog {
		if m.Family == "" || c.Family != m.Family || c.Tag == "" || c.Machine > machine || slices.Contains(c.NotOn, chip) {
			continue
		}
		c = c.On(machine)
		if !found || c.Machine > alt.Machine || c.Machine == alt.Machine && c.Memory > alt.Memory {
			alt, found = c, true
		}
	}
	return alt, found
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
				if alt, ok := notOnHere(m, p.Machine); ok {
					m = alt // the column's model is not picked on this chip: its family's next
				} else if alt.Name == "" {
					continue
				}
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
	// the model policy: a model it refuses is not picked (a tag of one is an error)
	var allowed []Model
	var refusals []string
	for _, c := range cands {
		if err := Allowed(c); err != nil {
			refusals = append(refusals, fmt.Sprintf("%s: %s", c.Name, err.(*PolicyError).Why))
			continue
		}
		allowed = append(allowed, c)
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
			return hits[0], fmt.Sprintf("%s:%s", base, hits[0].Tag), true, Allowed(hits[0])
		case 0:
			return Model{}, "", true, fmt.Errorf("chat: %s %s has no tag %q (it has %s, or a machine's memory as %s:16g)", kind, base, tag, strings.Join(tags, ", "), base)
		}
		var names []string
		for _, h := range hits {
			names = append(names, h.Family+":"+h.Tag)
		}
		return Model{}, "", true, fmt.Errorf("chat: %s:%s is any of %s: name the family", base, tag, strings.Join(names, ", "))
	}
	if kind == "category" && len(refusals) > 0 && !slices.ContainsFunc(allowed, func(c Model) bool { return c.Machine <= machine }) {
		// none of the use's own models for this machine is allowed: others of its kind
		if more := fallbacks(cands); len(more) > 0 {
			allowed = append(allowed, more...)
			refusals = append(refusals, fmt.Sprintf("so %s of its kind", strings.Join(names(more), ", ")))
		}
	}
	if len(allowed) == 0 {
		return Model{}, "", true, fmt.Errorf("chat: the model policy (%s, policy of models.yaml) refuses all the models of %s %s: %s", PolicyEnv, kind, base, strings.Join(refusals, "; "))
	}
	cands = allowed
	defer func() {
		if len(refusals) > 0 && err == nil {
			why += "; refused by the model policy: " + strings.Join(refusals, "; ")
		}
	}()
	// the largest column the machine has; among models of one column the larger on this machine
	for i := range cands {
		cands[i] = cands[i].On(machine)
	}
	best, found := cands[0], false
	chip := catalog.ChipName()
	var skipped []string
	for _, c := range cands {
		if c.Machine > machine {
			continue
		}
		if chip != "" && slices.Contains(c.NotOn, chip) {
			skipped = append(skipped, base+":"+c.Tag)
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
	if len(skipped) > 0 {
		why += fmt.Sprintf(", not %s on an %s", strings.Join(skipped, ", "), chip)
	}
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

// fallbacks are, for a use whose own models (cands) the model policy refuses on this machine, the
// other models of their kind and features that it allows and a family picks (Machine above 0),
// of the most open license class among them (an open license before a non-commercial one).
func fallbacks(cands []Model) []Model {
	features := slices.Clone(cands[0].Features)
	for _, c := range cands[1:] {
		features = slices.DeleteFunc(features, func(f string) bool { return !slices.Contains(c.Features, f) })
	}
	var out []Model
	best := len(LicenseClasses)
	for _, m := range Catalog {
		if m.Kind != cands[0].Kind || m.Machine == 0 || slices.ContainsFunc(cands, func(c Model) bool { return c.Name == m.Name }) ||
			slices.ContainsFunc(features, func(f string) bool { return !slices.Contains(m.Features, f) }) || Allowed(m) != nil {
			continue
		}
		class := slices.Index(LicenseClasses, m.License.Class)
		if class < 0 {
			class = len(LicenseClasses)
		}
		switch {
		case class < best:
			out, best = []Model{m}, class
		case class == best:
			out = append(out, m)
		}
	}
	return out
}

func names(ms []Model) []string {
	var s []string
	for _, m := range ms {
		s = append(s, m.Name)
	}
	return s
}
