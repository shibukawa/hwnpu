package chat

import (
	"fmt"
	"io"
	"path/filepath"
	"slices"
	"strings"
	"text/tabwriter"
)

// List writes the models of kinds (all for none) for -list: each model's name, tag, kind,
// features, size, repository, whether it is downloaded to dir, and what it is; then what each use
// and family of them picks on a machine of each of MODEL.md's columns; the aliases; and the
// definition files read.
func List(w io.Writer, dir string, ks ...Kind) {
	of := func(m Model) bool { return len(ks) == 0 || slices.Contains(ks, m.Kind) }
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "NAME\tTAG\tKIND\tFEATURES\tSIZE\tREPOSITORY\tDOWNLOADED\tABOUT")
	for _, m := range Catalog {
		if !of(m) {
			continue
		}
		got := "no"
		if Downloaded(filepath.Join(dir, m.DirName())) {
			got = "yes"
		}
		features := strings.Join(m.Features, ",")
		if features == "" {
			features = "-"
		}
		tag, size := "-", "-"
		if m.Tag != "" {
			tag = m.Family + ":" + m.Tag
		}
		if m.Bytes > 0 {
			size = fmt.Sprintf("%.1f GB", m.Bytes/1e9)
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", m.Name, tag, m.Kind, features, size, m.Source(), got, m.About)
	}
	tw.Flush()
	// MODEL.md's table: what each use and family picks on a machine of each column's memory
	machine := MachineMemory()
	fmt.Fprintf(w, "\nUses and families pick by the machine's memory (this one: %d GB), or by a tag: a model's (plamo3translate:2b)\nor a memory's (plamo3translate:8g):\n", machine)
	tw = tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	head := "  NAME\tHERE"
	for _, c := range Columns {
		head += fmt.Sprintf("\t%dg", c)
	}
	fmt.Fprintln(tw, head+"\tTAGS\tABOUT")
	// a family's cells are its tags; a use's name the family too, as its models are of several
	row := func(name, tags, about string, family bool) {
		m, _, _, _ := Pick(name, machine)
		line := fmt.Sprintf("  %s\t%s", name, m.Name)
		for _, c := range Columns {
			p, _, _, _ := Pick(name, c)
			switch {
			case family && p.Tag != "":
				line += "\t" + p.Tag
			case p.Tag != "":
				line += "\t" + p.Family + ":" + p.Tag
			default:
				line += "\t" + p.Name
			}
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\n", line, tags, about)
	}
	for _, c := range Categories {
		var tags []string
		mine := false
		for _, p := range c.Picks {
			t := byName(p.Model)
			mine = mine || of(t)
			if t.Tag != "" {
				tags = append(tags, t.Family+":"+t.Tag)
			} else {
				tags = append(tags, t.Name)
			}
		}
		if !mine {
			continue
		}
		about := c.About
		if len(c.Aliases) > 0 {
			about += " (also " + strings.Join(c.Aliases, ", ") + ")"
		}
		row(c.Name, strings.Join(tags, ", "), about, false)
	}
	seen := map[string]bool{}
	for _, f := range Catalog {
		if f.Tag == "" || seen[f.Family] || !of(f) {
			continue
		}
		seen[f.Family] = true
		var tags []string
		for _, t := range Catalog {
			if t.Family == f.Family && t.Tag != "" {
				tags = append(tags, t.Tag)
			}
		}
		row(f.Family, strings.Join(tags, ", "), "", true)
	}
	tw.Flush()
	fmt.Fprintf(w, "%s=GB picks as a machine of that memory would.\n", MemoryEnv)
	if len(Aliases) > 0 {
		fmt.Fprintln(w, "\nAliases (the model on the right runs wherever the one on the left is picked):")
		var froms []string
		for from := range Aliases {
			froms = append(froms, from)
		}
		slices.Sort(froms)
		for _, from := range froms {
			fmt.Fprintf(w, "  %s -> %s\n", from, Aliases[from])
		}
	}
	fmt.Fprintf(w, "\nDefinitions: %s (yours: %s, %s, -models)\n", strings.Join(ModelFiles, ", "), UserModelsFile(), ModelsEnv)
}
