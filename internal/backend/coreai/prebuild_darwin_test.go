package coreai

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestBuildLeavesVAEToDevice: Build passes over a text encoder's programs on every architecture
// and a VAE's on M1's (LeftToDevice), whose prebuilt ones come out wrong there, and builds a VAE
// for M2; Prebuilt does not take a text encoder's that lies there already.
func TestBuildLeavesVAEToDevice(t *testing.T) {
	export := filepath.Join(t.TempDir(), "m.coreai")
	for _, n := range []string{"dit_0", "vae_s0", "vae", "text_0"} {
		d := filepath.Join(export, n+".aimodel")
		os.MkdirAll(d, 0o755)
		os.WriteFile(filepath.Join(d, "main.mlirb"), []byte("bytecode "+n), 0o644)
		os.WriteFile(filepath.Join(d, "main.hash"), []byte("hash "+n), 0o644)
	}
	var log bytes.Buffer
	rs, err := Build(context.Background(), []string{export}, BuildOptions{Archs: []string{"h13g", "h14g"}, DryRun: true, Log: &log})
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, r := range rs {
		got = append(got, strings.TrimSuffix(filepath.Base(r.Asset), ".aimodel")+" "+r.Arch)
	}
	want := []string{"dit_0 h13g", "dit_0 h14g", "vae h14g", "vae_s0 h14g"}
	if strings.Join(got, ", ") != strings.Join(want, ", ") {
		t.Errorf("compiled: %v, want %v", got, want)
	}
	if strings.Count(log.String(), "left to the device") != 4 { // vae, vae_s0 on h13g; text_0 on both
		t.Errorf("log:\n%s", log.String())
	}

	// a prebuilt text encoder's program of the same source beside it (from an earlier hwnpu) is
	// not taken
	if arch := Arch(); arch != "" {
		text := filepath.Join(export, "text_0.aimodel")
		pb := PrebuiltPath(text, arch)
		os.MkdirAll(pb, 0o755)
		os.WriteFile(filepath.Join(pb, "main.hash"), []byte("hash text_0"), 0o644)
		if p := Prebuilt(text); p != "" {
			t.Errorf("a text encoder's prebuilt program was taken: %s", p)
		}
	}
}

func TestLeftToDevice(t *testing.T) {
	for _, c := range []struct {
		dir, arch string
		want      bool
	}{
		{"/x/m.coreai/vae_s3.aimodel", "h13g", true},
		{"/x/m.coreai/vae.aimodel/", "h13g", true},
		{"/x/m.coreai/VAE_decoder.aimodel", "h13c", true},
		{"/x/m.coreai/vae_s3.aimodel", "", true},
		{"/x/m.coreai/vae_s3.aimodel", "h14g", false},
		{"/x/m.coreai/VAE_decoder.aimodel", "h15g", false},
		{"/x/m.coreai/dit_single_0.aimodel", "h13g", false},
		{"/x/vae.coreai/text_0.aimodel", "h14g", true},
		{"/x/m.coreai/text.aimodel", "h15g", true},
		{"/x/text.coreai/mods.aimodel", "h13g", false},
		{"/x/m.coreai/ctx.aimodel", "h14g", false},
		{"/x/m.coreai/dit_double_0_device.aimodel", "h14g", true},
		{"/x/m_device.coreai/dit_double_0.aimodel", "h13g", false},
	} {
		if got := LeftToDevice(c.dir, c.arch); got != c.want {
			t.Errorf("%s on %q: %v", c.dir, c.arch, got)
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
