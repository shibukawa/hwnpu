package coreai

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestBuildLeavesVAEToDevice: Build passes over a VAE's and a text encoder's programs
// (LeftToDevice), whose prebuilt ones come out wrong on the device, and Prebuilt does not take one
// that lies there already.
func TestBuildLeavesVAEToDevice(t *testing.T) {
	export := filepath.Join(t.TempDir(), "m.coreai")
	for _, n := range []string{"dit_0", "vae_s0", "vae", "text_0"} {
		d := filepath.Join(export, n+".aimodel")
		os.MkdirAll(d, 0o755)
		os.WriteFile(filepath.Join(d, "main.mlirb"), []byte("bytecode "+n), 0o644)
		os.WriteFile(filepath.Join(d, "main.hash"), []byte("hash "+n), 0o644)
	}
	var log bytes.Buffer
	rs, err := Build(context.Background(), []string{export}, BuildOptions{Archs: []string{"h13g"}, DryRun: true, Log: &log})
	if err != nil {
		t.Fatal(err)
	}
	if len(rs) != 1 || filepath.Base(rs[0].Asset) != "dit_0.aimodel" {
		t.Errorf("compiled: %v", rs)
	}
	if strings.Count(log.String(), "left to the device") != 3 {
		t.Errorf("log:\n%s", log.String())
	}

	// a prebuilt VAE program of the same source beside it (from an earlier hwnpu) is not taken
	vae := filepath.Join(export, "vae_s0.aimodel")
	pb := PrebuiltPath(vae, "h13g")
	os.MkdirAll(pb, 0o755)
	os.WriteFile(filepath.Join(pb, "main.hash"), []byte("hash vae_s0"), 0o644)
	if p := Prebuilt(vae); p != "" {
		t.Errorf("a VAE's prebuilt program was taken: %s", p)
	}
	text := filepath.Join(export, "text_0.aimodel")
	pb = PrebuiltPath(text, "h13g")
	os.MkdirAll(pb, 0o755)
	os.WriteFile(filepath.Join(pb, "main.hash"), []byte("hash text_0"), 0o644)
	if p := Prebuilt(text); p != "" {
		t.Errorf("a text encoder's prebuilt program was taken: %s", p)
	}
}

func TestLeftToDevice(t *testing.T) {
	for dir, want := range map[string]bool{
		"/x/m.coreai/vae_s3.aimodel":              true,
		"/x/m.coreai/vae.aimodel/":                true,
		"/x/m.coreai/VAE_decoder.aimodel":         true,
		"/x/m.coreai/dit_single_0.aimodel":        false,
		"/x/vae.coreai/text_0.aimodel":            true,
		"/x/m.coreai/text.aimodel":                true,
		"/x/text.coreai/mods.aimodel":             false,
		"/x/m.coreai/ctx.aimodel":                 false,
		"/x/m.coreai/dit_double_0_device.aimodel": true,
		"/x/m_device.coreai/dit_double_0.aimodel": false,
	} {
		if got := LeftToDevice(dir); got != want {
			t.Errorf("%s: %v", dir, got)
		}
	}
}

// The reason of a failed compile is the log's lines of failures, their messages alone, the last
// three without repeats.
func TestReasonLines(t *testing.T) {
	log := `Timestamp               Ty Process[PID:TID]
2026-11-22 10:00:00.000 I  coreai-build[123:456] [com.apple.ane:compiler] Configuring bonded mode region_K
2026-11-22 10:00:01.000 E  coreai-build[123:456] [com.apple.ane:compiler] RegAlloc: failed to allocate tensors with fixed L2 hint
2026-11-22 10:00:01.000 E  coreai-build[123:456] [com.apple.ane:compiler] RegAlloc: failed to allocate tensors with fixed L2 hint
2026-11-22 10:00:02.000 E  coreai-build[123:456] [com.apple.ane:compiler] Failed to compile function func_s1152x768
`
	want := "RegAlloc: failed to allocate tensors with fixed L2 hint; Failed to compile function func_s1152x768"
	if got := reasonLines(log); got != want {
		t.Errorf("reason %q, want %q", got, want)
	}
	if got := reasonLines("Timestamp\n"); got != "" {
		t.Errorf("no failure: %q", got)
	}
}
