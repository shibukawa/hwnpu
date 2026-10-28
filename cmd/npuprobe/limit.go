package main

import (
	"fmt"

	"github.com/shibukawa/hwnpu/internal/probe"
)

var ladder = []int{2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256}

// sweep grows the model until it leaves the NPU or reaches maxBytes, then bisects. It returns
// the largest layer count on the NPU and the smallest not on it (0 when not found).
func (p *prober) sweep(format string, maxBytes int64, rows *[]result) (pass, fail int) {
	try := func(layers int) bool {
		r := p.isolated(childSpec{Kind: "try", Format: format, Layers: layers, Units: "npu", Quick: true})[0]
		logResult(r)
		*rows = append(*rows, r)
		return r.OnNPU
	}
	for _, layers := range ladder {
		if probe.WeightBytes(format, layers) > maxBytes || p.ctx.Err() != nil {
			break
		}
		if !try(layers) {
			fail = layers
			break
		}
		pass = layers
	}
	if pass == 0 || fail == 0 {
		return pass, fail
	}
	for fail-pass > 1 && p.ctx.Err() == nil {
		mid := (pass + fail) / 2
		if try(mid) {
			pass = mid
		} else {
			fail = mid
		}
	}
	return pass, fail
}

// several loads n models of the given size at once, timing each right after it is loaded and
// again when all are.
func (p *prober) several(format string, layers, n int) []result {
	var rows []result
	var models []*loaded
	defer func() {
		for _, l := range models {
			l.close()
		}
	}()
	for i := range n {
		if p.ctx.Err() != nil {
			break
		}
		r := result{Copy: i + 1}
		l := p.load(format, layers, "npu", uint64(1000+i), &r)
		if l != nil {
			if err := p.time(l, &r, false); err != nil {
				r.Error = "run: " + err.Error()
			}
			models = append(models, l)
		}
		r.OnNPU = r.Error == "" && r.Verified && r.NPUOps > 0 && r.OtherOps == 0
		rows = append(rows, r)
		if l == nil {
			break
		}
	}
	for i, l := range models {
		var r result
		if err := p.time(l, &r, false); err != nil {
			rows[i].Error = "run with all loaded: " + err.Error()
			continue
		}
		rows[i].MsAllLoaded, rows[i].CPUAllLoaded = r.MsPerCall, r.CPUUse
	}
	return rows
}

// limits sweeps every format, then loads several models of the largest size that stayed on the
// NPU, enough to exceed twice the smallest size that did not.
func (p *prober) limits(rep *report, formats []string, several bool, save func()) {
	type bound struct {
		format     string
		pass, fail int
	}
	var bounds []bound
	for _, f := range formats {
		fmt.Printf("limit, %s:\n", f)
		pass, fail := p.sweep(f, rep.MaxBytes, &rep.Sweep)
		bounds = append(bounds, bound{f, pass, fail})
		save()
	}
	if !several || p.ctx.Err() != nil {
		return
	}
	for _, b := range bounds {
		if b.pass == 0 || b.fail == 0 {
			continue
		}
		each := probe.WeightBytes(b.format, b.pass)
		n := int(min(max((2*probe.WeightBytes(b.format, b.fail)+each-1)/each, 2), 8))
		for n > 2 && int64(n)*each > rep.MaxBytes {
			n--
		}
		if int64(n)*each > rep.MaxBytes {
			rep.Notes = append(rep.Notes, fmt.Sprintf("several models: skipped, %d x %.2f GiB is above -max-gb", n, probe.GiB(each)))
			return
		}
		fmt.Printf("\nlimit, several %s models of %d layers (%.2f GiB each) at once:\n", b.format, b.pass, probe.GiB(each))
		rep.SeveralOf = fmt.Sprintf("%s, %d layers", b.format, b.pass)
		rep.Several = p.isolated(childSpec{Kind: "several", Format: b.format, Layers: b.pass, N: n})
		for _, r := range rep.Several {
			logResult(r)
			if r.MsAllLoaded > 0 {
				fmt.Printf("    with all %d loaded: %8.1f ms/call, CPU use %.2f\n", len(rep.Several), r.MsAllLoaded, r.CPUAllLoaded)
			}
		}
		return
	}
	rep.Notes = append(rep.Notes, "several models: skipped, no format left the NPU below -max-gb")
}
