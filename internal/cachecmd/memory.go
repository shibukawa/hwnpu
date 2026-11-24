package cachecmd

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"text/tabwriter"

	"github.com/shibukawa/bonsaiai/internal/chat"
	"github.com/shibukawa/bonsaiai/modelstore"
)

// MemoryCmd is bonsaiai memory.
type MemoryCmd struct {
	Model  string `arg:"" optional:"" help:"A model: its name (LFM3-1.7B), its repository, or a family or a use (the model it picks here). None: the models with a memory chosen."`
	GB     string `arg:"" optional:"" name:"gb" help:"The memory to run it as, in GB (8, 16g): as on a machine of that much memory. None: show what it runs as."`
	Forget bool   `help:"Forget the memory chosen for the model: run it as on this machine."`
}

func (c MemoryCmd) Run(ctx context.Context) error {
	return run(ctx, Request{Command: "memory", Memory: c})
}

func (c MemoryCmd) exec(e *Env) error {
	s, err := e.open()
	if err != nil {
		return err
	}
	if c.Model == "" {
		if c.GB != "" || c.Forget {
			return errors.New("name the model")
		}
		return listMemories(e, s)
	}
	var mem modelstore.Memory
	switch {
	case c.Forget && c.GB != "":
		return errors.New("--forget takes no memory")
	case c.Forget:
		mem, err = s.SetMemory(c.Model, 0)
	case c.GB != "":
		gb, ok := parseGB(c.GB)
		if !ok {
			return fmt.Errorf("%q is not a memory in GB (8, 16g)", c.GB)
		}
		before, err := s.Memory(c.Model)
		if err != nil {
			return err
		}
		if mem, err = s.SetMemory(c.Model, gb); err != nil {
			return err
		}
		if before.GB != mem.GB {
			defer fmt.Fprintf(e.Out, "The programs %s made for %d GB stay in the cache until it is pruned; bonsaiai cache clear %s removes them all, and the next run gets those of %d GB alone.\n", mem.Model, before.GB, mem.Model, mem.GB)
		}
	default:
		mem, err = s.Memory(c.Model)
	}
	if err != nil {
		return err
	}
	fmt.Fprintln(e.Out, describe(mem))
	return nil
}

// listMemories prints this machine's memory and the models with a memory chosen.
func listMemories(e *Env, s *modelstore.Store) error {
	fmt.Fprintf(e.Out, "this machine: %d GB (%s)\n", chat.MachineMemory(), chat.RunFile())
	chosen := s.Memories()
	if len(chosen) == 0 {
		fmt.Fprintln(e.Out, "no model has a memory chosen: each runs as on this machine")
		return nil
	}
	w := tabwriter.NewWriter(e.Out, 0, 4, 2, ' ', 0)
	names := make([]string, 0, len(chosen))
	for n := range chosen {
		names = append(names, n)
	}
	slices.Sort(names)
	for _, n := range names {
		fmt.Fprintf(w, "%s\t%d GB\t%s\n", n, chosen[n], how(n, chosen[n]))
	}
	return w.Flush()
}

// describe says how a model runs as mem says.
func describe(mem modelstore.Memory) string {
	src := fmt.Sprintf("this machine's memory; bonsaiai memory %s GB chooses another", mem.Model)
	if mem.Chosen != 0 {
		src = fmt.Sprintf("chosen, this machine %d GB; bonsaiai memory %s --forget forgets it", mem.Machine, mem.Model)
	}
	return fmt.Sprintf("%s: runs as on a machine of %d GB (%s)\n  %s", mem.Model, mem.GB, src, how(mem.Model, mem.GB))
}

// how says what models.yaml gives model name on a machine of gb GB: its programs and memory.
func how(name string, gb int) string {
	m, ok := chat.Lookup(name)
	if !ok {
		return ""
	}
	m = m.On(gb)
	var parts []string
	for _, f := range []struct{ name, value string }{
		{"prefill", m.Run.Prefill}, {"format", m.Run.Format}, {"context", itoa(m.Run.Context)},
		{"paged", itoa(m.Run.Paged)}, {"tier", m.Run.Tier},
	} {
		if f.value != "" {
			parts = append(parts, f.name+" "+f.value)
		}
	}
	out := "as its definition gives every machine"
	if len(parts) > 0 {
		out = strings.Join(parts, ", ")
	}
	if mb := m.MemoryFor(chat.HereVendor()); mb > 0 {
		out += fmt.Sprintf("; about %.1f GB running here", mb)
	}
	return out
}

func itoa(n int) string {
	if n == 0 {
		return ""
	}
	return strconv.Itoa(n)
}

// parseGB reads a memory: 8, 8g, 8GB.
func parseGB(s string) (int, bool) {
	if gb, ok := chat.MemoryTag(s); ok {
		return gb, true
	}
	n, err := strconv.Atoi(strings.TrimSpace(s))
	return n, err == nil && n > 0
}
