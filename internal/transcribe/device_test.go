package transcribe

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/check"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
)

// TestEncoderOnDevice runs encoder packages on the Neural Engine and on Core ML's CPU
// (HWNPU_COREML_UNITS=cpu, fp32 there) and compares both with transformers' fp32 encoder output,
// the checklist's 3.3 check for arithmetic the CPU interpreter does not show (the ANE's own
// layer_norm, rounding inside its operations). HWNPU_WHISPER_REF is a prefix written by
// tools/ref/ref_whisper.py (PREFIX.x.bin the mel frames, PREFIX.logits.bin the output);
// HWNPU_WHISPER_ENCODERS is a comma separated list of encoder .mlpackage directories.
func TestEncoderOnDevice(t *testing.T) {
	prefix, pkgs := os.Getenv("HWNPU_WHISPER_REF"), os.Getenv("HWNPU_WHISPER_ENCODERS")
	if prefix == "" || pkgs == "" {
		t.Skip("HWNPU_WHISPER_REF and HWNPU_WHISPER_ENCODERS are not set")
	}
	ref, err := check.LoadRef(prefix)
	if err != nil {
		t.Fatal(err)
	}
	mel, ok, err := ref.Extra("x")
	if err != nil || !ok {
		t.Fatalf("%s.x.bin: %v", prefix, err)
	}
	want := ref.Logits
	if l := os.Getenv("HWNPU_WHISPER_LAYER"); l != "" { // packages of the first N layers (hwnpu-convert -layers)
		n, err := strconv.Atoi(l)
		if err != nil {
			t.Fatal(err)
		}
		width := len(ref.Logits) / ref.Meta.Seq
		want = ref.Hidden[n*ref.Meta.Seq*width : (n+1)*ref.Meta.Seq*width]
	}
	// the positions (20 ms) of the clip's samples; the rest of the 30 seconds is zeros
	speech := ref.Meta.Seq
	if raw, err := os.ReadFile(prefix[:len(prefix)-len(filepath.Ext(prefix))] + ".samples.bin"); err == nil {
		speech = min(speech, (len(raw)/4+319)/320)
	}
	ctx := context.Background()
	cache, _ := os.UserCacheDir()
	dev, _, err := llm.OpenDevice(ctx, "", filepath.Join(cache, "hwnpu"))
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	units := []string{"npu", "cpu"}
	if u := os.Getenv("HWNPU_WHISPER_UNITS"); u != "" {
		units = strings.Split(u, ",")
	}
	for _, pkg := range strings.Split(pkgs, ",") {
		var outs [][]float32
		for _, u := range units {
			if u != "npu" {
				os.Setenv("HWNPU_COREML_UNITS", u)
			}
			p, err := runner.Open(ctx, dev, pkg)
			os.Unsetenv("HWNPU_COREML_UNITS")
			if err != nil {
				t.Fatalf("%s on %s: %v", pkg, u, err)
			}
			var out []float32
			var best time.Duration
			for i := range 4 { // the first call loads the program onto the device
				t0 := time.Now()
				if out, err = p.Run(ctx, map[string][]float32{"x": mel}); err != nil {
					t.Fatal(err)
				}
				if d := time.Since(t0); i > 0 && (best == 0 || d < best) {
					best = d
				}
			}
			npu := p.OnNPU()
			p.Close()
			rel, cos, worst, at, each := comparePositions(out, want, len(ref.Logits)/ref.Meta.Seq)
			in, after := below(each, speech)
			t.Logf("%s on %s (on the NPU: %v): %.1f ms; against HF fp32: relative RMS %.4f, cosine %.6f, worst position %d %.4f, under 0.99: %d of the %d speech positions, %d of the padding's",
				filepath.Base(pkg), u, npu, float64(best.Microseconds())/1000, rel, cos, at, worst, in, speech, after)
			outs = append(outs, out)
		}
		for i := 1; i < len(outs); i++ {
			rel, cos, worst, at := compare(outs[0], outs[i], len(ref.Logits)/ref.Meta.Seq)
			t.Logf("%s: %s against %s: relative RMS %.4f, cosine %.6f, worst position %d %.4f",
				filepath.Base(pkg), units[0], units[i], rel, cos, at, worst)
		}
	}
}

// compare is got's relative RMS error and cosine against want, and the least cosine of a
// position (rows of width values) with its index.
func compare(got, want []float32, width int) (rel, cos, worst float64, at int) {
	rel, cos, worst, at, _ = comparePositions(got, want, width)
	return
}

// comparePositions is compare with each position's cosine.
func comparePositions(got, want []float32, width int) (rel, cos, worst float64, at int, each []float64) {
	var dd, ww, gw, gg float64
	worst = 2
	for r := 0; r*width < len(want); r++ {
		var pd, pw, pg float64
		for i := r * width; i < (r+1)*width; i++ {
			g, w := float64(got[i]), float64(want[i])
			dd += (g - w) * (g - w)
			pd += g * w
			pw += w * w
			pg += g * g
		}
		ww += pw
		gg += pg
		gw += pd
		c := pd / math.Sqrt(pw*pg)
		if c < worst {
			worst, at = c, r
		}
		each = append(each, c)
	}
	return math.Sqrt(dd / ww), gw / math.Sqrt(ww*gg), worst, at, each
}

// below counts the cosines under 0.99, in the first n positions and after them.
func below(each []float64, n int) (in, after int) {
	for i, c := range each {
		if c < 0.99 {
			if i < n {
				in++
			} else {
				after++
			}
		}
	}
	return in, after
}
