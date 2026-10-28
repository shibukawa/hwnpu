package main

import (
	"fmt"
	"slices"
	"strings"

	"github.com/shibukawa/hwnpu/internal/probe"
)

func markdown(rep *report) string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "# hwnpu npuprobe\n\n%s\n\n", rep.Date)
	fmt.Fprintf(&sb, "| | |\n|---|---|\n| CPU | %s |\n| machine | %s |\n| OS | %s |\n| RAM | %.0f GiB |\n| NPU | %s (%s) |\n",
		rep.CPU, rep.Model, rep.OS, probe.GiB(rep.RAMBytes), rep.Device, rep.Backend)
	if rep.Driver != "" || rep.Firmware != "" {
		fmt.Fprintf(&sb, "| driver | %s %s |\n", rep.Driver, rep.Firmware)
	}
	if rep.Meter != "" {
		fmt.Fprintf(&sb, "| power meter | %s, idle %.2f W |\n", rep.Meter, rep.IdleW)
	}
	fmt.Fprintf(&sb, "| hwnpu | %s, %s |\n\n", rep.Version, rep.Go)
	sb.WriteString("Models: one token through Llama 3.2 1B sized layers (61 M weights per layer, random values), so a call is the work of generating one token without the attention over the context.\n")

	if len(rep.Bench) > 0 {
		sb.WriteString("\n## Speed\n\n| weights | format | unit | placement | ms/token | tokens/s | weights GB/s | CPU use | net W | mJ/token | compile s | error |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n")
		for _, r := range rep.Bench {
			tps := ""
			if r.MsPerCall > 0 {
				tps = fmt.Sprintf("%.1f", 1000/r.MsPerCall)
			}
			fmt.Fprintf(&sb, "| %.2f B, %.2f GiB | %s | %s | %s | %s | %s | %s | %s | %s | %s | %.1f | %s |\n",
				float64(r.Params)/1e9, probe.GiB(r.WeightBytes), r.Format, strings.ToUpper(r.Units), r.placement(),
				cell(r.MsPerCall, "%.1f"), tps, cell(r.gbps(), "%.0f"), cell(r.CPUUse, "%.2f"), cell(r.NetW, "%.2f"), cell(r.MJPerCall, "%.1f"), r.CompileS, errCell(r))
		}
	}

	if len(rep.Sweep) > 0 {
		fmt.Fprintf(&sb, "\n## Size limit (one model at a time, up to %.1f GiB)\n\n", probe.GiB(rep.MaxBytes))
		sb.WriteString("| format | layers | weights | placement | ms/call | weights GB/s | CPU use | compile s | error |\n|---|---|---|---|---|---|---|---|---|\n")
		rows := slices.Clone(rep.Sweep)
		slices.SortStableFunc(rows, func(a, b result) int {
			if a.Format != b.Format {
				return strings.Compare(a.Format, b.Format)
			}
			return a.Layers - b.Layers
		})
		for _, r := range rows {
			fmt.Fprintf(&sb, "| %s | %d | %.2f GiB | %s | %s | %s | %s | %.1f | %s |\n", r.Format, r.Layers, probe.GiB(r.WeightBytes), r.placement(),
				cell(r.MsPerCall, "%.1f"), cell(r.gbps(), "%.0f"), cell(r.CPUUse, "%.2f"), r.CompileS, errCell(r))
		}
		sb.WriteString("\n")
		for _, f := range formatsOf(rep.Sweep) {
			pass, fail := 0, 0
			for _, r := range rep.Sweep {
				if r.Format != f {
					continue
				}
				if r.OnNPU && r.Layers > pass {
					pass = r.Layers
				}
				if !r.OnNPU && (fail == 0 || r.Layers < fail) {
					fail = r.Layers
				}
			}
			switch {
			case fail == 0:
				fmt.Fprintf(&sb, "- %s: every size tried ran on the NPU, up to %d layers (%.2f GiB).\n", f, pass, probe.GiB(probe.WeightBytes(f, pass)))
			case pass == 0:
				fmt.Fprintf(&sb, "- %s: not on the NPU even at %d layers (%.2f GiB).\n", f, fail, probe.GiB(probe.WeightBytes(f, fail)))
			default:
				fmt.Fprintf(&sb, "- %s: %d layers (%.2f GiB) on the NPU, %d layers (%.2f GiB) not.\n", f, pass, probe.GiB(probe.WeightBytes(f, pass)), fail, probe.GiB(probe.WeightBytes(f, fail)))
			}
		}
	}

	if len(rep.Several) > 0 {
		var total int64
		for _, r := range rep.Several {
			if r.MsPerCall > 0 {
				total += r.WeightBytes
			}
		}
		fmt.Fprintf(&sb, "\n## Several models at once (%s each, %.2f GiB loaded in total)\n\n", rep.SeveralOf, probe.GiB(total))
		sb.WriteString("| model | weights | placement | ms/call after loading it | CPU use | ms/call with all loaded | CPU use with all loaded | compile s | error |\n|---|---|---|---|---|---|---|---|---|\n")
		for _, r := range rep.Several {
			fmt.Fprintf(&sb, "| %d | %.2f GiB | %s | %s | %s | %s | %s | %.1f | %s |\n", r.Copy, probe.GiB(r.WeightBytes), r.placement(),
				cell(r.MsPerCall, "%.1f"), cell(r.CPUUse, "%.2f"), cell(r.MsAllLoaded, "%.1f"), cell(r.CPUAllLoaded, "%.2f"), r.CompileS, errCell(r))
		}
	}
	if len(rep.Notes) > 0 {
		sb.WriteString("\n")
		for _, n := range rep.Notes {
			fmt.Fprintf(&sb, "- %s\n", n)
		}
	}
	sb.WriteString(`
Placement: Apple, Core ML's compute plan (MLComputePlan) for every operation; Intel, NPU when the
driver compiler built the graph (it has no fallback). CPU use is the process CPU time divided by
the wall time while the model runs: near 0 when the NPU does the work, 1 or more when the CPU
does. weights GB/s is the weight bytes times calls per second. net W is the device's power meter
while running minus idle, when it has one.
`)
	return sb.String()
}

func cell(v float64, f string) string {
	if v == 0 {
		return ""
	}
	return fmt.Sprintf(f, v)
}

func errCell(r result) string { return strings.ReplaceAll(short(r.Error, 200), "|", "/") }

func formatsOf(rows []result) []string {
	var fs []string
	for _, r := range rows {
		if !slices.Contains(fs, r.Format) {
			fs = append(fs, r.Format)
		}
	}
	return fs
}
