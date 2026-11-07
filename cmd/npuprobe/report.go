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

	if len(rep.Verify) > 0 {
		sb.WriteString("\n## K tokens per call (checking K drafted tokens)\n\n| weights | format | K | placement | ms/call | × K=1 | ms/token | tokens/s | weights GB/s | TFLOPS | CPU use | compile s | error |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
		for _, r := range rep.Verify {
			ratio, perToken, tps := "", "", ""
			if base := baseMs(rep.Verify, r); base > 0 && r.MsPerCall > 0 {
				ratio = fmt.Sprintf("%.2f", r.MsPerCall/base)
			}
			if r.MsPerCall > 0 {
				perToken = fmt.Sprintf("%.1f", r.MsPerCall/float64(r.Tokens))
				if r.MsPerCall < float64(r.Tokens) {
					perToken = fmt.Sprintf("%.3f", r.MsPerCall/float64(r.Tokens))
				}
				tps = fmt.Sprintf("%.1f", 1000*float64(r.Tokens)/r.MsPerCall)
			}
			fmt.Fprintf(&sb, "| %.2f B, %.2f GiB | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %.1f | %s |\n",
				float64(r.Params)/1e9, probe.GiB(r.WeightBytes), r.Format, r.Tokens, r.placement(),
				cell(r.MsPerCall, "%.1f"), ratio, perToken, tps, cell(r.gbps(), "%.0f"), cell(r.tflops(), "%.2f"), cell(r.CPUUse, "%.2f"), r.CompileS, errCell(r))
		}
		sb.WriteString("\n× K=1 is the time per call over that of one token. Near 1, the call is bound by reading the weights and checking K drafted tokens costs about as much as generating one; tokens/s is then the most speculative decoding could reach if every draft were accepted. TFLOPS counts a multiply and an add per weight and token.\n\n")
		for _, line := range knees(rep.Verify) {
			fmt.Fprintf(&sb, "- %s\n", line)
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
	if len(rep.Split) > 0 {
		sb.WriteString("\n## A model split into parts run one after another\n\n")
		sb.WriteString("| format | weights | parts | layout | placement of the parts | ms/token | parts alone added up | tokens/s | weights GB/s | CPU use | peak RSS | Neural Engine | mJ/token | error after each part | error |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
		for _, s := range rep.Split {
			var place, errs []string
			for _, r := range s.Parts {
				place = append(place, fmt.Sprintf("%d layers %.2f GiB %s", r.Layers, probe.GiB(r.WeightBytes), r.placement()))
			}
			for _, e := range s.RelErr {
				errs = append(errs, fmt.Sprintf("%.2f%%", 100*e))
			}
			against := ""
			if len(errs) > 0 {
				against = "vs " + strings.ToUpper(s.Ref) + ": " + strings.Join(errs, ", ")
			}
			tps := ""
			if s.MsPerToken > 0 {
				tps = fmt.Sprintf("%.1f", 1000/s.MsPerToken)
			}
			layout := "files"
			if s.Layout != "" {
				layout = s.Layout
			}
			rss := ""
			if s.MaxRSS > 0 {
				rss = fmt.Sprintf("%.2f GiB", probe.GiB(s.MaxRSS))
			}
			ane := ""
			if s.NPUWired > 0 {
				ane = fmt.Sprintf("%.2f GiB", probe.GiB(s.NPUWired))
			}
			fmt.Fprintf(&sb, "| %s | %d layers, %.2f GiB | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n", s.Format, s.Layers, probe.GiB(s.WeightBytes), len(s.Parts), layout,
				strings.Join(place, "<br>"), cell(s.MsPerToken, "%.1f"), cell(s.MsSumParts, "%.1f"), tps, cell(s.gbps(), "%.0f"), cell(s.CPUUse, "%.2f"), rss, ane, cell(s.MJPerToken, "%.1f"),
				against, strings.ReplaceAll(short(s.Error, 200), "|", "/"))
		}
		sb.WriteString("\nEvery part is loaded before any is timed. ms/token runs each part once, the output buffer of one bound as the input of the next; parts alone added up is the sum of each part's own time. Layout functions writes the parts as the functions of one package, loaded one function at a time. Peak RSS is the largest resident memory of the process that loaded and ran the parts (it leaves out the system's own Core ML and Neural Engine processes). Neural Engine is what loading the parts and running them once added to the wired kernel memory of the Neural Engine's driver (com.apple.driver.AppleH11ANEInterface, read with mach_memory_info): where the weights on the Neural Engine are, in no process's RSS. The error after each part compares the hidden state with the same parts run one at a time on the reference units (rms of the difference over rms of the reference).\n")
	}
	if len(rep.KV) > 0 {
		sb.WriteString("\n## KV cache: host buffers against Core ML states\n\n")
		sb.WriteString("| cache | KV bytes | mode | placement | ms/token | CPU use | error |\n|---|---|---|---|---|---|---|\n")
		for _, k := range rep.KV {
			fmt.Fprintf(&sb, "| %d | %.1f MiB | %s | %s | %s | %s | %s |\n", k.Cache, float64(k.CacheBytes)/(1<<20), k.Mode, k.placement(), cell(k.MsPerCall, "%.2f"), cell(k.CPUUse, "%.2f"), errCell(k.result))
		}
		sb.WriteString("\nOne token through lut4 layers of the bench size with attention over a cache of the given length. host binds the cache as inputs and outputs the token's keys and values; state keeps it in Core ML states (MLState) and writes the token in with a one-hot row.\n")
	}
	if len(rep.Attn) > 0 {
		sb.WriteString("\n## Attention per chunk of a KV cache, merged on the host\n\n")
		sb.WriteString("| form | Q | chunk | placement | ms/call alone | CPU use | compile s | error |\n|---|---|---|---|---|---|---|---|\n")
		for _, a := range rep.Attn {
			fmt.Fprintf(&sb, "| %s | %d | %d | %s | %s | %s | %.1f | %s |\n", a.Form, a.Tokens, a.Chunk, a.placement(), cell(a.MsPerCall, "%.3f"), cell(a.CPUUse, "%.2f"), a.CompileS, errCell(a.result))
		}
		sb.WriteString("\n| form | Q | chunk | placement | used | layers × chunks | ms/step | ms/call in the step | of it the merge, ms | ms per 1K used | KV | peak RSS | CPU use | error |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
		for _, a := range rep.Attn {
			for _, s := range a.Passes {
				per1K := 0.0
				if s.MsPerPass > 0 {
					per1K = s.MsPerPass / (float64(s.Used) / 1024)
				}
				rss := ""
				if s.MaxRSS > 0 {
					rss = fmt.Sprintf("%.2f GiB", probe.GiB(s.MaxRSS))
				}
				fmt.Fprintf(&sb, "| %s | %d | %d | %s | %d | %d × %d | %s | %s | %s | %s | %.0f MiB | %s | %s | %s |\n", a.Form, a.Tokens, a.Chunk, a.placement(), s.Used, s.Layers, s.Chunks,
					cell(s.MsPerPass, "%.2f"), cell(s.msPerCall(), "%.3f"), cell(s.MsMerge, "%.2f"), cell(per1K, "%.3f"), float64(s.KVBytes)/(1<<20), rss, cell(s.CPUUse, "%.2f"), strings.ReplaceAll(short(s.Error, 200), "|", "/"))
			}
		}
		sb.WriteString("\nThe program is attention without weights over one chunk of the cache (keys and values of 4 kv heads × 128, 16 query heads, Q query rows), returning the chunk's normalized output, largest scores and sums of exps. A step calls it once per chunk holding the used tokens in each layer, every chunk bound from buffers of its own, and merges the outputs on the host in float32. ms/call alone binds one chunk; ms/call in the step is a step's time over its calls, the merge included, so a difference between the two is the cost of binding many buffers. Peak RSS is the measuring process's largest resident memory after the step (it leaves out Core ML's and the Neural Engine's own processes); a peak well over the KV bytes means the inputs were copied.\n")
	}
	if len(rep.Seam) > 0 {
		sb.WriteString("\n## A model split at its attention: weighted parts and the chunk attention interleaved\n\n")
		sb.WriteString("| weights | parts | Q | attention | placement of the parts | of the attention | used | chunks | ms/step weights alone | attention alone | interleaved | ms per seam | seams | CPU use | peak RSS | error |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
		for _, s := range rep.Seam {
			head := fmt.Sprintf("| %s %d layers, %.2f GiB | %d | %d | %s, %d layers, chunks of %d | %s | %s |", s.Format, s.Layers, probe.GiB(s.WeightBytes), s.AttnLayers+1, s.Tokens, s.Form, s.AttnLayers, s.Chunk,
				partsPlacement(s.Parts), s.Attn.placement())
			if len(s.Runs) == 0 {
				fmt.Fprintf(&sb, "%s | | | | | | | | | %s |\n", head, strings.ReplaceAll(short(s.Error, 200), "|", "/"))
			}
			for _, r := range s.Runs {
				rss := ""
				if r.MaxRSS > 0 {
					rss = fmt.Sprintf("%.2f GiB", probe.GiB(r.MaxRSS))
				}
				e := r.Error
				if e == "" {
					e = s.Error
				}
				seam := ""
				if r.MsMixed > 0 {
					seam = fmt.Sprintf("%+.3f", r.msPerSeam())
				}
				fmt.Fprintf(&sb, "%s %d | %d | %s | %s | %s | %s | %d | %s | %s | %s |\n", head, r.Used, r.Chunks, cell(r.MsWeights, "%.2f"), cell(r.MsAttn, "%.2f"), cell(r.MsMixed, "%.2f"),
					seam, r.Seams, cell(r.CPUUse, "%.2f"), rss, strings.ReplaceAll(short(e, 200), "|", "/"))
			}
		}
		sb.WriteString("\nThe weighted parts are the bench layers split into one more part than there are attention layers, every part loaded with the attention program at once. A step runs the parts one after another (weights alone), the attention of every layer over the used chunks with the host's merge (attention alone), or the two interleaved as a model split at its attention would run them: part, attention, part, … (interleaved). Each is the fastest of two rounds. ms per seam is what interleaving adds over the two apart, divided by the switches between a part and the attention; near 0, the Neural Engine keeps both kinds of program, and a reload would show as milliseconds.\n")
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

func errCell(r result) string {
	e := r.Error
	if e == "" && r.PlanError != "" {
		e = "placement not verified: " + r.PlanError
	}
	return strings.ReplaceAll(short(e, 200), "|", "/")
}

func formatsOf(rows []result) []string {
	var fs []string
	for _, r := range rows {
		if !slices.Contains(fs, r.Format) {
			fs = append(fs, r.Format)
		}
	}
	return fs
}
