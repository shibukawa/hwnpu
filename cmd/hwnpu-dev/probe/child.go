package probe

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu"
)

// Each measurement runs in a child process, so that a framework or driver that aborts the
// process (Core ML's GPU path asserts on some graphs) costs one row of the report, not the run.

const childMode, childMark = "_measure", "npuprobe-result "

type childSpec struct {
	Kind       string        `json:"kind"` // try, several, split, chain, kv, attn, seam, cache or resident
	Backend    string        `json:"backend"`
	DeviceID   string        `json:"device_id"`
	Format     string        `json:"format"`
	Layers     int           `json:"layers"`
	Tokens     int           `json:"tokens"` // 0 means 1
	Units      string        `json:"units"`
	Quick      bool          `json:"quick"`
	N          int           `json:"n"`
	Layout     string        `json:"layout,omitempty"`      // split: files or functions; kv: the mode; attn: the form
	Used       []int         `json:"used,omitempty"`        // attn, seam: used cache lengths of the passes
	AttnLayers int           `json:"attn_layers,omitempty"` // seam: attention layers between the weighted parts
	Paths      []string      `json:"paths,omitempty"`       // cache: the programs, in the order to load
	CacheDir   string        `json:"cache_dir,omitempty"`   // hwnpu's compile cache (WithCacheDir), kept by the parent
	MinRun     time.Duration `json:"min_run"`
	Dir        string        `json:"dir"`
	IdleW      float64       `json:"idle_w"`
	Meter      bool          `json:"meter"`
}

// isolated runs spec in a child process and returns its results. A child that dies yields one
// result carrying the error.
func (p *prober) isolated(spec childSpec) []result {
	var rs []result
	if err := p.child(spec, &rs); err != nil {
		return []result{p.fail(spec, err.Error())}
	}
	return rs
}

// ChildArgs are the arguments that come before the child's own when this executable runs
// itself as a child (hwnpu-dev: "probe").
var ChildArgs []string

// child runs spec in a child process and decodes what it prints into out.
func (p *prober) child(spec childSpec, out any) error {
	spec.Backend, spec.DeviceID = p.dev.Info().Backend, p.dev.Info().ID
	spec.MinRun, spec.Dir, spec.IdleW, spec.Meter = p.minRun, p.dir, p.idleW, p.meter
	arg, _ := json.Marshal(spec)
	exe := p.exe
	if exe == "" {
		var err error
		if exe, err = os.Executable(); err != nil {
			return fmt.Errorf("executable: %w", err)
		}
	}
	cmd := exec.CommandContext(p.ctx, exe, append(slices.Clone(ChildArgs), childMode, string(arg))...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	stdout, err := cmd.Output()
	for _, line := range strings.Split(string(stdout), "\n") {
		if rest, ok := strings.CutPrefix(line, childMark); ok {
			if json.Unmarshal([]byte(rest), out) == nil {
				return nil
			}
		}
	}
	msg := "measurement process failed"
	if err != nil {
		msg += ": " + err.Error()
	}
	if s := strings.TrimSpace(stderr.String()); s != "" {
		msg += ": " + lastLines(s, 3)
	}
	return errors.New(msg)
}

func (p *prober) fail(spec childSpec, msg string) result {
	r := result{Format: spec.Format, Layers: spec.Layers, Tokens: max(spec.Tokens, 1), Units: spec.Units, Error: msg}
	if r.Units == "" {
		r.Units = "npu"
	}
	r.Params, r.WeightBytes = paramsAndBytes(spec.Format, spec.Layers)
	return r
}

func lastLines(s string, n int) string {
	lines := strings.Split(s, "\n")
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	return strings.Join(lines, " / ")
}

// runChild is the child side: it opens the same device, measures and prints the results.
func runChild(arg string) error {
	var spec childSpec
	if err := json.Unmarshal([]byte(arg), &spec); err != nil {
		return err
	}
	ctx := context.Background()
	infos, err := hwnpu.Enumerate(ctx)
	if err != nil {
		return err
	}
	var info *hwnpu.DeviceInfo
	for i := range infos {
		if infos[i].Backend == spec.Backend && infos[i].ID == spec.DeviceID {
			info = &infos[i]
		}
	}
	if info == nil {
		return fmt.Errorf("device %s %s not found", spec.Backend, spec.DeviceID)
	}
	opts := []hwnpu.Option{hwnpu.WithPlacement(hwnpu.PlacementPrefer)}
	if spec.CacheDir != "" {
		opts = append(opts, hwnpu.WithCacheDir(spec.CacheDir))
	} else if spec.Layout == "functions" {
		// Every part loads the same package; compile it once.
		cache := filepath.Join(spec.Dir, "coreml-cache")
		defer os.RemoveAll(cache)
		opts = append(opts, hwnpu.WithCacheDir(cache))
	}
	dev, err := hwnpu.OpenDevice(ctx, *info, opts...)
	if err != nil {
		return err
	}
	defer dev.Close()
	gr, err := dev.Graph()
	if err != nil {
		return err
	}
	p := &prober{ctx: ctx, dev: dev, gr: gr, coreml: info.Backend == "coreml", dir: spec.Dir, minRun: spec.MinRun, idleW: spec.IdleW, meter: spec.Meter}
	var rs any
	switch spec.Kind {
	case "try":
		rs = []result{p.try(spec.Format, spec.Layers, max(spec.Tokens, 1), spec.Units, spec.Quick)}
	case "several":
		rs = p.several(spec.Format, spec.Layers, spec.N)
	case "split":
		rs = p.split(spec.Format, spec.Layers, spec.N, spec.Layout)
	case "kv":
		rs = p.kv(spec.Layout, spec.Layers, spec.N)
	case "attn":
		rs = p.attn(spec.Layout, spec.Layers, max(spec.Tokens, 1), spec.N, spec.Used)
	case "seam":
		rs = p.seam(spec.Format, spec.Layers, spec.AttnLayers, max(spec.Tokens, 1), spec.Layout, spec.N, spec.Used)
	case "resident":
		rs = p.resident(spec.Format, spec.Layers, spec.Layout, spec.N)
	case "cache":
		rs = p.cacheLoads(spec.Format, spec.Layers, spec.Paths, spec.Layout, spec.N)
	case "chain":
		rs = p.chain(spec.Format, spec.Layers, spec.N, spec.Units)
	default:
		return fmt.Errorf("unknown measurement %q", spec.Kind)
	}
	b, _ := json.Marshal(rs)
	fmt.Printf("\n%s%s\n", childMark, b)
	return nil
}
