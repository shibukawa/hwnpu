package probe

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
)

// AMD's NPU runs no programs of hwnpu's own (no graph backend): its models are IRON exports. On
// it the probe exports PROBE.md's shapes with tools/amd/iron_probe.py (random weights, the
// operators hwnpu's exports use), in IRON's environment (HWNPU_IRON_ENV, as hwnpu-chat's first
// run), and times them through internal/llm/iron, as hwnpu runs a model:
//
//	bench   the decode step's linears of 8 and 16 layers (layers), and the same weights in GEMVs
//	        of one shape back to back (stream: no operator loaded again between them, the ceiling
//	        of a step run as one resident array)
//	verify  the same layers on K rows: a call checking drafts (2 to 8 rows, QGEMVRows) and a
//	        prompt chunk (multiples of 256 rows, the GEMM), in TFLOPS
//	attn    the image DiTs' attention (FLUX.2 klein's: 24 heads of 128, rows queries over rows keys)

// amdResult is one graph exported and timed.
type amdResult struct {
	Probe       string    `json:"probe"` // bench, verify, attn
	Part        string    `json:"part"`  // layers, stream, attn
	Format      string    `json:"format,omitempty"`
	Layers      int       `json:"layers,omitempty"`
	Rows        int       `json:"rows"`
	Heads       int       `json:"heads,omitempty"`
	Dim         int       `json:"dim,omitempty"`
	Params      int64     `json:"params,omitempty"`
	WeightBytes int64     `json:"weight_bytes,omitempty"`
	FLOPs       float64   `json:"flops"`
	Commands    int       `json:"commands"`
	ExportS     float64   `json:"export_s"`
	LoadS       float64   `json:"load_s"`
	Calls       int       `json:"calls,omitempty"`
	MsPerCall   float64   `json:"ms_per_call,omitempty"`
	CPUUse      float64   `json:"cpu_use,omitempty"`
	NetW        float64   `json:"net_w,omitempty"`
	MJPerCall   float64   `json:"mj_per_call,omitempty"`
	Power       *amdPower `json:"power,omitempty"`
	Error       string    `json:"error,omitempty"`
}

func (r amdResult) gbps() float64 {
	if r.MsPerCall == 0 || r.WeightBytes == 0 {
		return 0
	}
	return float64(r.WeightBytes) / (r.MsPerCall / 1000) / 1e9
}

func (r amdResult) tflops() float64 {
	if r.MsPerCall == 0 {
		return 0
	}
	return r.FLOPs / (r.MsPerCall / 1000) / 1e12
}

func (r amdResult) name() string {
	if r.Part == "attn" {
		return fmt.Sprintf("attn-r%d-h%d-d%d", r.Rows, r.Heads, r.Dim)
	}
	return fmt.Sprintf("%s-%s-l%d-r%d", r.Part, r.Format, r.Layers, r.Rows)
}

// amdRows says whether a call of rows rows runs on the path of hwnpu's exports in format f, and
// why not: one row is a decode step (QGEMV; bfp16 is packed for the GEMM and takes rows), 2 to 8
// a call checking drafts (QGEMVRows, q8 and q6 only), more a prompt chunk, whose GEMM takes
// multiples of 256 rows (flm.ProjectionGEMM; a chunk is padded to it).
func amdRows(f string, rows int) string {
	switch {
	case rows == 1 && f == "bfp16":
		return "bfp16 is packed for the GEMM, which takes rows"
	case rows == 1:
		return ""
	case rows <= 8 && f != "q8" && f != "q6":
		return "a call checking drafts runs q8 and q6 only (QGEMVRows)"
	case rows <= 8:
		return ""
	case f == "bf16":
		return "a prompt chunk of bf16 weights is not one hwnpu's exports use"
	case rows%256 != 0:
		return "a prompt chunk's GEMM takes multiples of 256 rows"
	}
	return ""
}

type amdOpts struct {
	python                   []string
	formats, verifyFormats   []string
	layers, verifyLayers     []int
	rows, attnRows           []int
	attnHeads, attnDim       int
	bench, verify, attention bool
}

// amdProbe runs the AMD measurements of mode on dev.
func (p *prober) amdProbe(rep *report, o amdOpts, save func()) {
	if o.bench {
		fmt.Println("bench: one row (a decode step's linears)")
		for _, n := range o.layers {
			for _, f := range o.formats {
				for _, part := range []string{"layers", "stream"} {
					if p.ctx.Err() != nil {
						return
					}
					rep.AMD = append(rep.AMD, p.amdRun(o.python, amdResult{Probe: "bench", Part: part, Format: f, Layers: n, Rows: 1}))
					save()
				}
			}
		}
		fmt.Println()
	}
	if o.verify {
		fmt.Println("verify: K rows a call (checking drafts, a prompt chunk)")
		for _, n := range o.verifyLayers {
			for _, f := range o.verifyFormats {
				for _, k := range o.rows {
					if p.ctx.Err() != nil {
						return
					}
					if amdRows(f, k) != "" {
						continue
					}
					rep.AMD = append(rep.AMD, p.amdRun(o.python, amdResult{Probe: "verify", Part: "layers", Format: f, Layers: n, Rows: k}))
					save()
				}
			}
		}
		fmt.Println()
	}
	if o.attention {
		fmt.Println("attn: the image DiTs' attention")
		for _, n := range o.attnRows {
			if p.ctx.Err() != nil {
				return
			}
			rep.AMD = append(rep.AMD, p.amdRun(o.python, amdResult{Probe: "attn", Part: "attn", Rows: n, Heads: o.attnHeads, Dim: o.attnDim}))
			save()
		}
		fmt.Println()
	}
}

// amdRun exports r's graph into the work directory, loads it, times it and removes it.
func (p *prober) amdRun(python []string, r amdResult) amdResult {
	name := r.name()
	out := filepath.Join(p.dir, name)
	args := []string{chat.ExportOut, r.Part}
	if r.Part == "attn" {
		args = append(args, "--rows", strconv.Itoa(r.Rows), "--heads", strconv.Itoa(r.Heads), "--dim", strconv.Itoa(r.Dim))
	} else {
		args = append(args, "--fmt", r.Format, "--layers", strconv.Itoa(r.Layers), "--rows", strconv.Itoa(r.Rows))
	}
	defer func() {
		os.RemoveAll(out)
		for _, s := range []string{".done", ".log", ".part"} {
			os.RemoveAll(out + s)
		}
	}()
	t0 := time.Now()
	if err := chat.RunIRON(p.ctx, python, name, "iron_probe.py", args, out, io.Discard, nil); err != nil {
		r.Error = "export: " + err.Error()
		logAMD(r)
		return r
	}
	r.ExportS = time.Since(t0).Seconds()
	if err := p.amdTime(out, &r); err != nil {
		r.Error = err.Error()
	}
	logAMD(r)
	return r
}

func (p *prober) amdTime(dir string, r *amdResult) error {
	var m struct {
		Meta struct {
			Weights     int64   `json:"weights"`
			WeightBytes int64   `json:"weight_bytes"`
			FLOPs       float64 `json:"flops"`
		} `json:"meta"`
	}
	b, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return err
	}
	if err := json.Unmarshal(b, &m); err != nil {
		return err
	}
	r.Params, r.WeightBytes, r.FLOPs = m.Meta.Weights, m.Meta.WeightBytes, m.Meta.FLOPs
	t0 := time.Now()
	g, err := iron.OpenGraphWith(p.ctx, p.dev, dir, iron.GraphOptions{})
	if err != nil {
		return fmt.Errorf("load: %w", err)
	}
	defer g.Close()
	r.LoadS = time.Since(t0).Seconds()
	r.Commands = len(g.Manifest.Graph.Kernels)
	var vals map[string]int64
	if b, err := os.ReadFile(filepath.Join(dir, "check", "values.json")); err == nil {
		if err := json.Unmarshal(b, &vals); err != nil {
			return err
		}
	}
	for _, h := range g.Inputs() {
		b, err := os.ReadFile(filepath.Join(dir, "check", "in_"+h.Name+".bin"))
		if err != nil {
			return err
		}
		if err := g.SetInput(h.Name, b); err != nil {
			return err
		}
	}
	var t result
	for range 2 { // measure's warm-up, here before the sampler starts
		if err := g.Run(p.ctx, vals); err != nil {
			return fmt.Errorf("run: %w", err)
		}
	}
	smp := startAMDSampler(p.amdMetrics)
	err = p.measure(func() error { return g.Run(p.ctx, vals) }, &t, false)
	r.Power = smp.stop()
	if err != nil {
		return fmt.Errorf("run: %w", err)
	}
	r.Calls, r.MsPerCall, r.CPUUse, r.NetW, r.MJPerCall = t.Calls, t.MsPerCall, t.CPUUse, t.NetW, t.MJPerCall
	return nil
}

func logAMD(r amdResult) {
	what := fmt.Sprintf("%-6s %-5s %2d layers %4d rows", r.Part, r.Format, r.Layers, r.Rows)
	if r.Part == "attn" {
		what = fmt.Sprintf("attn   %5d rows, %d heads of %d", r.Rows, r.Heads, r.Dim)
	}
	line := fmt.Sprintf("  %-36s export %5.1fs load %4.1fs", what, r.ExportS, r.LoadS)
	if r.MsPerCall > 0 {
		line += fmt.Sprintf("  %8.2f ms/call", r.MsPerCall)
		if g := r.gbps(); g > 0 && r.Rows == 1 {
			line += fmt.Sprintf(", %.1f GB/s", g)
		} else {
			line += fmt.Sprintf(", %.2f TFLOPS", r.tflops())
		}
		if w := r.Power; w != nil {
			line += fmt.Sprintf(", package %.1f W (NPU %.1f, rest %.1f), DRAM %.0f GB/s", w.SocketW, w.NPUW, w.RestW, w.DRAMRead+w.DRAMWrite)
		}
	}
	if r.Error != "" {
		line += "  error: " + short(r.Error, 300)
	}
	fmt.Println(line)
}

// amdDevice is the AMD NPU of infos, when there is one.
func amdDevice(infos []hwnpu.DeviceInfo) *hwnpu.DeviceInfo {
	for i := range infos {
		if infos[i].Vendor == hwnpu.VendorAMD {
			return &infos[i]
		}
	}
	return nil
}

// amdMarkdown is the report's AMD sections.
func amdMarkdown(sb *strings.Builder, rows []amdResult, idle *amdPower) {
	var bench, verify, attn []amdResult
	for _, r := range rows {
		switch r.Probe {
		case "attn":
			attn = append(attn, r)
		case "bench":
			bench = append(bench, r)
		case "verify":
			verify = append(verify, r)
		}
	}
	if len(bench) > 0 {
		sb.WriteString("\n## Speed of a decode step (AMD, IRON)\n\n")
		sb.WriteString("| weights | format | form | commands | ms/token | tokens/s | weights GB/s | G weights/s | CPU use | net W | mJ/token | export s | error |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
		for _, r := range bench {
			tps := ""
			if r.MsPerCall > 0 {
				tps = fmt.Sprintf("%.1f", 1000/r.MsPerCall)
			}
			fmt.Fprintf(sb, "| %.2f B (%d layers), %.2f GiB | %s | %s | %d | %s | %s | %s | %s | %s | %s | %s | %.0f | %s |\n",
				float64(r.Params)/1e9, r.Layers, float64(r.WeightBytes)/(1<<30), r.Format, r.Part, r.Commands,
				cell(r.MsPerCall, "%.1f"), tps, cell(r.gbps(), "%.0f"), cell(gweights(r.Params, r.MsPerCall), "%.0f"), cell(r.CPUUse, "%.2f"), cell(r.NetW, "%.2f"), cell(r.MJPerCall, "%.1f"), r.ExportS, short(r.Error, 120))
		}
		sb.WriteString("\nlayers: a layer's seven linears one after another (QGEMV for q4, q6 and q8, IRON's GEMV for bf16) and the SwiGLU's product, as hwnpu's exports of the other models than gpt-oss run them: every operator that follows another is loaded onto the cores again. stream: the same bytes in GEMVs of one shape (8192 by 2048) back to back, not loaded again: the time is the weights' transfer, the ceiling of a decode step run as one resident array (gpt-oss-20b's GptOssStep, docs/design/moe.md).\n")
	}
	if len(verify) > 1 {
		sb.WriteString("\n## K rows per call (AMD, IRON)\n\n| weights | format | K | path | ms/call | × K=1 | ms/token | tokens/s | TFLOPS | export s | error |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
		for _, r := range verify {
			path := "QGEMV (decode step)"
			switch {
			case r.Rows > 1 && r.Rows <= 8:
				path = "QGEMVRows (checking drafts)"
			case r.Rows > 8 && r.Format == "bfp16":
				path = "GEMM, weights packed on the host"
			case r.Rows > 8:
				path = "dequantize and GEMM (prompt chunk)"
			case r.Format == "bf16":
				path = "IRON's GEMV"
			}
			ratio, per, tps := "", "", ""
			for _, b := range verify {
				if b.Format == r.Format && b.Layers == r.Layers && b.Rows == 1 && b.MsPerCall > 0 && r.MsPerCall > 0 {
					ratio = fmt.Sprintf("%.2f", r.MsPerCall/b.MsPerCall)
				}
			}
			if r.MsPerCall > 0 {
				per = fmt.Sprintf("%.3f", r.MsPerCall/float64(r.Rows))
				tps = fmt.Sprintf("%.0f", 1000*float64(r.Rows)/r.MsPerCall)
			}
			fmt.Fprintf(sb, "| %.2f B (%d layers) | %s | %d | %s | %s | %s | %s | %s | %s | %.0f | %s |\n",
				float64(r.Params)/1e9, r.Layers, r.Format, r.Rows, path, cell(r.MsPerCall, "%.1f"), ratio, per, tps, cell(r.tflops(), "%.2f"), r.ExportS, short(r.Error, 120))
		}
		sb.WriteString("\n× K=1 is the time per call over that of one row in the same format. A call checking 2 to 8 drafts streams the weights once through all its rows (q8, q6); a prompt chunk of q4 or q8 dequantizes its weights on the device at every call, bfp16 (the image DiTs' weights) is read by the GEMM as it is. TFLOPS counts a multiply and an add per weight and row.\n")
	}
	if len(attn) > 0 {
		sb.WriteString("\n## The image DiTs' attention (AMD, IRON)\n\n| queries and keys | heads | ms/call | TFLOPS | export s | error |\n|---|---|---|---|---|---|\n")
		for _, r := range attn {
			fmt.Fprintf(sb, "| %d | %d of %d | %s | %s | %.0f | %s |\n", r.Rows, r.Heads, r.Dim, cell(r.MsPerCall, "%.1f"), cell(r.tflops(), "%.2f"), r.ExportS, short(r.Error, 120))
		}
		sb.WriteString("\niron_mha.ImageMHA with QKᵀ and P·V through bfp16 and the softmax's exponential on one bf16 limb (FLUX.2 klein's export, docs/design/image-amd.md). TFLOPS counts the two products (4 × rows² × heads × dim).\n")
	}
	amdPowerMarkdown(sb, rows, idle)
}

// amdPowerMarkdown writes where the power of each run went, from gpu_metrics.
func amdPowerMarkdown(sb *strings.Builder, rows []amdResult, idle *amdPower) {
	if idle == nil || !slices.ContainsFunc(rows, func(r amdResult) bool { return r.Power != nil }) {
		return
	}
	sb.WriteString("\n## Where the power goes (AMD, gpu_metrics)\n\n")
	fmt.Fprintf(sb, "At rest: package %.2f W (NPU %.2f, CPU cores %.2f, GPU %.2f, the rest %.2f), DRAM %.1f GB/s read and %.1f written.\n\n",
		idle.SocketW, idle.NPUW, idle.CoresW, idle.GfxW, idle.RestW, idle.DRAMRead, idle.DRAMWrite)
	sb.WriteString("| run | ms/call | package W | above rest | NPU W | CPU cores | the rest | NPU's share | the rest's share | DRAM GB/s | NPU busy | NPU MHz |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n")
	for _, r := range rows {
		w := r.Power
		if w == nil {
			continue
		}
		up := w.SocketW - idle.SocketW
		share := func(x float64) string {
			if up <= 0.1 {
				return ""
			}
			return fmt.Sprintf("%.0f %%", 100*x/up)
		}
		dn, dc, dr := w.NPUW-idle.NPUW, w.CoresW-idle.CoresW, w.RestW-idle.RestW
		fmt.Fprintf(sb, "| %s %s | %s | %.2f | %+.2f | %.2f | %+.2f | %+.2f | %s | %s | %.1f | %.0f %% | %.0f |\n",
			r.Probe, r.name(), cell(r.MsPerCall, "%.1f"), w.SocketW, up, w.NPUW, dc, dr, share(dn), share(dr), w.DRAMRead+w.DRAMWrite, w.NPUBusy, w.NPUClock)
	}
	sb.WriteString("\nThe SMU's own split (amdgpu's gpu_metrics, the means of a sample every 50 ms while the run is timed). NPU W is the NPU's rail; the rest is the package less the NPU, the CPU cores and the GPU: the data fabric, the memory controller and its PHY, and I/O, the part of moving the weights that is on the package. The DRAM chips are outside it and not measured. The shares are of the package's power above rest.\n")
}

// amdFlags are the flags that mean something on AMD, and which were given.
type amdFlags struct {
	formats, layers, tokens string
	set                     map[string]bool
	imageRows               string
	imageHeads, imageDim    int
}

// runAMD probes a machine whose NPU is AMD's: bench, verify and attn (all of them for all).
func runAMD(ctx context.Context, info *hwnpu.DeviceInfo, mode string, af amdFlags, minRun time.Duration, work, out string) error {
	o := amdOpts{attnHeads: af.imageHeads, attnDim: af.imageDim}
	switch mode {
	case "all":
		o.bench, o.verify, o.attention = true, true, true
	case "bench":
		o.bench = true
	case "verify":
		o.verify = true
	case "attn":
		o.attention = true
	default:
		return fmt.Errorf("%s: not on AMD's NPU (bench, verify and attn are)", mode)
	}
	o.formats, o.verifyFormats = []string{"q4", "q6", "q8", "bf16"}, []string{"q8", "q6", "q4", "bfp16"}
	if af.set["formats"] {
		o.formats, o.verifyFormats = split(af.formats), split(af.formats)
	}
	for _, f := range o.formats {
		if !slices.Contains([]string{"q4", "q6", "q8", "bf16", "bfp16"}, f) {
			return fmt.Errorf("-formats: %q is not one of q4, q6, q8, bf16, bfp16 on AMD's NPU", f)
		}
	}
	o.formats = slices.DeleteFunc(slices.Clone(o.formats), func(f string) bool { return f == "bfp16" }) // rows only
	var err error
	if o.layers, err = counts("-layers", af.layers); err != nil {
		return err
	}
	o.verifyLayers = o.layers[:1] // the prompt chunks' exports are large: one size unless asked
	if af.set["layers"] {
		o.verifyLayers = o.layers
	}
	tokens := "1,2,4,8,256,1024,4096"
	if af.set["tokens"] {
		tokens = af.tokens
	}
	if o.rows, err = counts("-tokens", tokens); err != nil {
		return err
	}
	if o.attnRows, err = counts("-image-rows", af.imageRows); err != nil {
		return err
	}
	if o.python, err = chat.IRONPython(ctx); err != nil {
		return fmt.Errorf("AMD's NPU: the probe's graphs are exported with IRON: %w (set %s, see INSTALL_AMD.md)", err, chat.IRONEnv)
	}
	dev, err := hwnpu.OpenDevice(ctx, *info)
	if err != nil {
		return err
	}
	defer dev.Close()
	rep := newReport(info)
	if work == "" {
		if work, err = os.MkdirTemp("", "hwnpu-npuprobe-"); err != nil {
			return err
		}
		defer os.RemoveAll(work)
	}
	if out == "" {
		out = reportName(&rep)
	}
	save := func() { write(out, &rep) }
	p := &prober{ctx: ctx, dev: dev, dir: work, minRun: minRun, amdMetrics: amdMetricsPath()}
	fmt.Printf("%s, %s, %s, RAM %.0f GiB\nNPU: %s (%s)\n", rep.CPU, rep.Model, rep.OS, float64(rep.RAMBytes)/(1<<30), info.Name, info.Backend)
	if info.Caps&hwnpu.CapPowerMeter != 0 {
		if err := p.measureIdle(); err != nil {
			rep.Notes = append(rep.Notes, "power meter not readable ("+short(err.Error(), 120)+")")
		} else {
			rep.IdleW = p.idleW
			if s, err := dev.PowerSample(); err == nil {
				rep.Meter = s.Source
			}
			fmt.Printf("power meter %s, idle %.2f W\n", rep.Meter, p.idleW)
		}
	}
	if p.amdMetrics != "" {
		smp := startAMDSampler(p.amdMetrics)
		time.Sleep(4 * time.Second)
		if rep.AMDIdle = smp.stop(); rep.AMDIdle != nil {
			i := rep.AMDIdle
			fmt.Printf("gpu_metrics at rest: package %.2f W (NPU %.2f, cores %.2f, GPU %.2f, the rest %.2f), DRAM %.1f GB/s\n",
				i.SocketW, i.NPUW, i.CoresW, i.GfxW, i.RestW, i.DRAMRead+i.DRAMWrite)
		}
	}
	fmt.Println("Other programs on the NPU or the CPU spoil the times (pgrep -fa 'hwnpu|iron_').")
	fmt.Println()
	p.amdProbe(&rep, o, save)
	if ctx.Err() != nil {
		rep.Notes = append(rep.Notes, "interrupted")
	}
	if err := write(out, &rep); err != nil {
		return err
	}
	fmt.Printf("report: %s (and %s)\n", out, strings.TrimSuffix(out, ".md")+".json")
	return nil
}
