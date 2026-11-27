package coreai

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestBuildLeavesDeviceProgramsToDevice: Build passes over a program named *_device
// (LeftToDevice) and builds a VAE's and a text encoder's (the recipe leaves those it must to the
// device), and Prebuilt does not take a *_device program's that lies there already.
func TestBuildLeavesDeviceProgramsToDevice(t *testing.T) {
	export := filepath.Join(t.TempDir(), "m.coreai")
	for _, n := range []string{"dit_0", "vae_s0", "text_0", "dit_1_device"} {
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
	var got []string
	for _, r := range rs {
		got = append(got, filepath.Base(r.Asset))
	}
	if strings.Join(got, " ") != "dit_0.aimodel text_0.aimodel vae_s0.aimodel" {
		t.Errorf("compiled: %v", got)
	}
	if strings.Count(log.String(), "left to the device") != 1 {
		t.Errorf("log:\n%s", log.String())
	}

	// a prebuilt program of the same source beside it (from an earlier hwnpu) is not taken
	dev := filepath.Join(export, "dit_1_device.aimodel")
	pb := PrebuiltPath(dev, "h13g")
	os.MkdirAll(pb, 0o755)
	os.WriteFile(filepath.Join(pb, "main.hash"), []byte("hash dit_1_device"), 0o644)
	if p := Prebuilt(dev); p != "" {
		t.Errorf("a program left to the device was taken prebuilt: %s", p)
	}
}

func TestLeftToDevice(t *testing.T) {
	for dir, want := range map[string]bool{
		"/x/m.coreai/vae_s3.aimodel":              false,
		"/x/m.coreai/vae.aimodel/":                false,
		"/x/m.coreai/dit_single_0.aimodel":        false,
		"/x/vae.coreai/text_0.aimodel":            false,
		"/x/m.coreai/text.aimodel":                false,
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
