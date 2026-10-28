// Command npuprobe measures the NPU of this machine with generated models, without Python, model
// files or root:
//
//	go run github.com/shibukawa/hwnpu/cmd/npuprobe@latest          # bench, then limit
//	go run github.com/shibukawa/hwnpu/cmd/npuprobe@latest bench    # speed only
//	go run github.com/shibukawa/hwnpu/cmd/npuprobe@latest limit    # size limit only
//
// The models have the shape of LLM decoding (internal/probe): one token through Llama 3.2 1B
// sized layers with random weights, so every call reads every weight once, as generating one
// token does.
//
// bench runs 8 and 16 layers (0.5 B and 1 B weights) in each weight format and gives the time per
// token, the weight bandwidth and, where the device has a readable power meter (Intel and AMD
// on Linux as root), the energy per token. On Apple it also runs each model on the CPU and the GPU
// through Core ML; for energy there use cmd/powerbench -decode.
//
// limit grows the model in each format until it leaves the NPU, bisects to the boundary, then
// loads several models just under it at once to tell a limit per model from a limit on the total.
//
// Placement comes from Core ML's compute plan on Apple; on Intel the driver compiler either builds
// the whole graph for the NPU or fails. The report also gives the process CPU time per wall time
// while a model runs: near 0 when the NPU does the work, 1 or more when the CPU does.
//
// The report is written to npuprobe-<machine>-<time>.md and .json.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"runtime"
	"runtime/debug"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/probe"
)

type report struct {
	Date      string   `json:"date"`
	CPU       string   `json:"cpu"`
	Model     string   `json:"model"`
	OS        string   `json:"os"`
	RAMBytes  int64    `json:"ram_bytes"`
	Device    string   `json:"device"`
	Backend   string   `json:"backend"`
	Driver    string   `json:"driver,omitempty"`
	Firmware  string   `json:"firmware,omitempty"`
	Version   string   `json:"hwnpu_version"`
	Go        string   `json:"go"`
	IdleW     float64  `json:"idle_w,omitempty"`
	Meter     string   `json:"meter,omitempty"`
	MaxBytes  int64    `json:"max_bytes"`
	Bench     []result `json:"bench,omitempty"`
	Sweep     []result `json:"sweep,omitempty"`
	Several   []result `json:"several,omitempty"`
	SeveralOf string   `json:"several_of,omitempty"`
	Notes     []string `json:"notes,omitempty"`
}

func main() {
	mode := "all"
	args := os.Args[1:]
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		mode, args = args[0], args[1:]
	}
	fs := flag.NewFlagSet("npuprobe", flag.ExitOnError)
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "usage: npuprobe [all|bench|limit] [flags]")
		fs.PrintDefaults()
	}
	formats := fs.String("formats", "", "weight formats (default: fp16,int8,lut4 and int4 on the CPU and GPU on Apple; fp16,int8,int4 on Intel)")
	layers := fs.String("layers", "8,16", "bench: layer counts (61 M weights each)")
	baselines := fs.String("baselines", "cpu,gpu", "bench on Apple: also run each model on these Core ML units (empty for none)")
	maxGB := fs.Float64("max-gb", 0, "limit: largest weight size to try, GiB (default: a quarter of the RAM on Apple, a sixth on Intel, at most 8)")
	several := fs.Bool("several", true, "limit: after the sweep, load several models just under the limit at once")
	minRun := fs.Duration("run", 3*time.Second, "time each model for at least this long")
	work := fs.String("dir", "", "directory for generated models (default: a temporary directory; Apple needs free space of about twice -max-gb)")
	out := fs.String("o", "", "report file (default npuprobe-<machine>-<time>.md, and .json beside it)")
	fs.Parse(args)
	if mode != "all" && mode != "bench" && mode != "limit" {
		fs.Usage()
		os.Exit(2)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if err := run(ctx, mode, *formats, *layers, *baselines, *maxGB, *several, *minRun, *work, *out); err != nil {
		fmt.Fprintln(os.Stderr, "npuprobe:", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, mode, formats, layerList, baselines string, maxGB float64, several bool, minRun time.Duration, work, out string) error {
	infos, err := hwnpu.Enumerate(ctx)
	if err != nil {
		return err
	}
	var info *hwnpu.DeviceInfo
	for i := range infos {
		if (infos[i].Backend == "coreml" || infos[i].Backend == "levelzero") && infos[i].Caps&hwnpu.CapGraph != 0 {
			info = &infos[i]
			break
		}
	}
	if info == nil {
		return errors.New("no NPU that runs models was found (Apple: macOS 14.4 or later on Apple Silicon; Intel: the NPU driver and Level Zero, see INSTALL_INTEL.md; run cmd/report for details)")
	}
	dev, err := hwnpu.OpenDevice(ctx, *info, hwnpu.WithPlacement(hwnpu.PlacementPrefer))
	if err != nil {
		return err
	}
	defer dev.Close()
	gr, err := dev.Graph()
	if err != nil {
		return err
	}

	rep := report{Date: time.Now().Format(time.RFC3339), Device: info.Name, Backend: info.Backend, Driver: info.Driver, Firmware: info.Firmware, Go: runtime.Version()}
	rep.CPU, rep.Model, rep.OS, rep.RAMBytes = machine()
	if bi, ok := debug.ReadBuildInfo(); ok {
		rep.Version = bi.Main.Version
		for _, s := range bi.Settings {
			if s.Key == "vcs.revision" {
				rep.Version += " " + s.Value
			}
		}
	}
	p := &prober{ctx: ctx, dev: dev, gr: gr, coreml: info.Backend == "coreml", minRun: minRun}

	benchFormats, limitFormats := split(formats), split(formats)
	if formats == "" {
		benchFormats, limitFormats = []string{"fp16", "int8", "int4"}, []string{"fp16", "int8", "int4"}
		if p.coreml {
			benchFormats, limitFormats = []string{"fp16", "int8", "lut4", "int4"}, []string{"fp16", "int8", "lut4"}
		}
	}
	for _, f := range benchFormats {
		if _, ok := probe.BytesPerParam[f]; !ok || (!p.coreml && f == "lut4") {
			return fmt.Errorf("format %q is not available on %s", f, info.Backend)
		}
	}
	limitFormats = slices.DeleteFunc(limitFormats, func(f string) bool { return p.coreml && f == "int4" })
	var layers []int
	for _, s := range split(layerList) {
		n, err := strconv.Atoi(s)
		if err != nil || n < 1 {
			return fmt.Errorf("-layers: %q is not a layer count", s)
		}
		layers = append(layers, n)
	}
	var units []string
	if p.coreml {
		for _, u := range split(baselines) {
			if u != "cpu" && u != "gpu" {
				return fmt.Errorf("-baselines: %q is not cpu or gpu", u)
			}
			units = append(units, u)
		}
	}

	rep.MaxBytes = int64(maxGB * (1 << 30))
	if rep.MaxBytes == 0 {
		div := int64(6)
		if p.coreml {
			div = 4
		}
		rep.MaxBytes = min(rep.RAMBytes/div, 8<<30)
		if rep.MaxBytes == 0 {
			rep.MaxBytes = 2 << 30
		}
	}
	if work == "" {
		if work, err = os.MkdirTemp("", "hwnpu-npuprobe-"); err != nil {
			return err
		}
		defer os.RemoveAll(work)
	}
	p.dir = work
	if out == "" {
		name := rep.Model
		if name == "" {
			name = runtime.GOOS
		}
		name = strings.Map(func(r rune) rune {
			if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '-' {
				return r
			}
			return '-'
		}, name)
		out = fmt.Sprintf("npuprobe-%s-%s.md", name, time.Now().Format("20060102-1504"))
	}
	save := func() { write(out, &rep) } // keep what was measured if the run is cut short

	fmt.Printf("%s, %s, %s, RAM %.0f GiB\nNPU: %s (%s)\n", rep.CPU, rep.Model, rep.OS, probe.GiB(rep.RAMBytes), info.Name, info.Backend)
	if info.Caps&hwnpu.CapPowerMeter != 0 {
		if err := p.measureIdle(); err != nil {
			rep.Notes = append(rep.Notes, "power meter not readable ("+short(err.Error(), 120)+"); on Linux run as root for energy")
		} else {
			rep.IdleW = p.idleW
			if s, err := dev.PowerSample(); err == nil {
				rep.Meter = s.Source
			}
			fmt.Printf("power meter %s, idle %.2f W\n", rep.Meter, p.idleW)
		}
	}
	fmt.Println()

	if mode != "limit" {
		p.bench(&rep, benchFormats, layers, units, save)
		fmt.Println()
	}
	if mode != "bench" && ctx.Err() == nil {
		fmt.Printf("limit: models up to %.1f GiB of weights\n", probe.GiB(rep.MaxBytes))
		p.limits(&rep, limitFormats, several, save)
	}
	if ctx.Err() != nil {
		rep.Notes = append(rep.Notes, "interrupted")
	}
	if err := write(out, &rep); err != nil {
		return err
	}
	fmt.Printf("\nreport: %s (and %s)\n", out, strings.TrimSuffix(out, ".md")+".json")
	return nil
}

func split(s string) []string {
	var out []string
	for _, f := range strings.Split(s, ",") {
		if f = strings.TrimSpace(f); f != "" {
			out = append(out, f)
		}
	}
	return out
}

func write(out string, rep *report) error {
	if err := os.WriteFile(out, []byte(markdown(rep)), 0o644); err != nil {
		return err
	}
	b, _ := json.MarshalIndent(rep, "", "  ")
	return os.WriteFile(strings.TrimSuffix(out, ".md")+".json", append(b, '\n'), 0o644)
}
