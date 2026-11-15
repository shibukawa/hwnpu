package chat

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"testing"
)

// RunName is the file in hwnpu's directory of settings (beside the user's models.yaml,
// UserModelsFile) that keeps the memory the user chose to run each model as (ChooseMemory).
const RunName = "run.json"

// RunFile is the path of RunName: hwnpu's directory of settings (~/.config/hwnpu,
// $XDG_CONFIG_HOME/hwnpu, %LOCALAPPDATA%\hwnpu) of this machine, not the cache, so that what is
// chosen stays when the cache's models are pruned and is not shared with another machine that
// reads the same cache (HWNPU_CACHE). "" for none: in go test without XDG_CONFIG_HOME, which
// never reads the machine's.
func RunFile() string {
	if testing.Testing() && os.Getenv("XDG_CONFIG_HOME") == "" {
		return ""
	}
	if f := UserModelsFile(); f != "" {
		return filepath.Join(filepath.Dir(f), RunName)
	}
	return ""
}

// runFile is what RunName holds: {"models": {"LFM3-1.7B": {"memory": 8}}}.
type runFile struct {
	Models map[string]runEntry `json:"models"`
}

type runEntry struct {
	Memory int `json:"memory,omitempty"` // GB, as MachineMemory
}

// readRun reads RunFile, empty when it is not there or does not read.
func readRun() runFile {
	r := runFile{Models: map[string]runEntry{}}
	p := RunFile()
	if p == "" {
		return r
	}
	data, err := os.ReadFile(p)
	if err != nil {
		return r
	}
	var f runFile
	if json.Unmarshal(data, &f) != nil {
		return r
	}
	for name, e := range f.Models {
		if e.Memory > 0 {
			r.Models[name] = e
		}
	}
	return r
}

// ChosenMemory is the memory, in GB, the user chose to run model m as on this machine
// (ChooseMemory), 0 for none. Every command of hwnpu runs m so (Resolve, ResolveFor, Pick's
// model): a 16 GB machine that keeps 8 for LFM3-1.7B, to run it beside other programs, runs it
// as models.yaml defines it for 8 GB (its variants), not as for 16, in hwnpu-chat and in an
// application alike, and converts or fetches the programs of that alone. How a model runs on a
// machine of some memory is the definition's (models.yaml); the user chooses only the machine.
func ChosenMemory(m Model) int { return readRun().Models[m.Name].Memory }

// ChosenMemories are the models with a memory chosen (ChooseMemory), by name.
func ChosenMemories() map[string]int {
	out := map[string]int{}
	for name, e := range readRun().Models {
		out[name] = e.Memory
	}
	return out
}

// ChooseMemory keeps gb as the memory to run model m as on this machine (ChosenMemory); 0
// forgets it.
func ChooseMemory(m Model, gb int) error {
	if gb < 0 {
		return fmt.Errorf("chat: %d GB of memory", gb)
	}
	p := RunFile()
	if p == "" {
		return errors.New("chat: no directory of settings to keep the memory of a model in")
	}
	r := readRun()
	if gb == 0 {
		if _, ok := r.Models[m.Name]; !ok {
			return nil
		}
		delete(r.Models, m.Name)
	} else {
		r.Models[m.Name] = runEntry{Memory: gb}
	}
	if len(r.Models) == 0 {
		if err := os.Remove(p); err != nil && !errors.Is(err, fs.ErrNotExist) {
			return err
		}
		return nil
	}
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	// whole or not at all: another process may read it
	tmp := p + ".tmp"
	if err := os.WriteFile(tmp, append(data, '\n'), 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, p)
}

// runsAs is the memory model m runs as on a machine of machine GB: machine when it was given
// (given: a flag or a tag of memory), else the one chosen for m (ChosenMemory, logged), else
// machine.
func runsAs(m Model, machine int, given bool, log io.Writer) int {
	if given {
		return machine
	}
	gb := ChosenMemory(m)
	if gb == 0 {
		return machine
	}
	if log != nil {
		fmt.Fprintf(log, "%s: runs as on a machine of %d GB, as chosen (%s; hwnpu memory %s --forget forgets it)\n", m.Name, gb, RunFile(), m.Name)
	}
	return gb
}

// Over is r with the fields o gives.
func (r Run) Over(o Run) Run { return r.over(o) }

// Named is the model of the definitions name means on this machine: by its name or repository,
// or the one a family or a use picks here (lfm3, translate), through the aliases.
func Named(name string) (Model, error) {
	m, ok, err := lookupOn(name, MachineMemory(), nil)
	if err != nil {
		return Model{}, err
	}
	if !ok {
		return Model{}, errNoModel(name)
	}
	return m, nil
}
