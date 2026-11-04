package iron

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu"
)

// LFM3-VL's vision tower on the NPU gives the soft tokens Hugging Face's does: an export in
// HWNPU_IRON_VL, and in HWNPU_IRON_VL_CHECK an image's patches in raster order (patches.bin,
// float32) and the soft tokens Hugging Face computed from them (soft.bin, float32).
func TestLfm3VlTowerOnDevice(t *testing.T) {
	dir, check := os.Getenv("HWNPU_IRON_VL"), os.Getenv("HWNPU_IRON_VL_CHECK")
	if dir == "" || check == "" {
		t.Skip("HWNPU_IRON_VL and HWNPU_IRON_VL_CHECK are not set")
	}
	ctx := context.Background()
	dev, err := hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD))
	if err != nil {
		t.Skip(err)
	}
	defer dev.Close()
	m, err := Open(ctx, dev, dir, t.Logf)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	mm := m.Manifest.Multimodal
	if mm == nil || mm.Kind != "lfm3vl" || !m.HasVision() {
		t.Fatalf("%s is no LFM3-VL export", dir)
	}
	f32 := func(name string) []float32 {
		b, err := os.ReadFile(filepath.Join(check, name))
		if err != nil {
			t.Fatal(err)
		}
		out := make([]float32, len(b)/4)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
		}
		return out
	}
	patches, want := f32("patches.bin"), f32("soft.bin")
	buf := make([]byte, 2*len(patches))
	for i, x := range patches {
		binary.LittleEndian.PutUint16(buf[2*i:], bf16(x))
	}
	if err := m.runTower(ctx, m.vision, towerInput(mm, mm.ImageToken), buf, map[string]int64{}, 0); err != nil {
		t.Fatal(err)
	}
	merged, _, err := m.Buffer("vision", "merged")
	if err != nil {
		t.Fatal(err)
	}
	width := m.Manifest.Config.HiddenSize
	at := m.Manifest.Config.PrefillChunk // the soft tokens follow a prompt chunk's rows (Multimodal.at)
	got := make([]float32, len(want))
	for i := range got {
		h := binary.LittleEndian.Uint16(merged[2*(at*width+i):])
		got[i] = math.Float32frombits(uint32(h) << 16)
	}
	worst := 1.0
	for r := 0; r < len(want)/width; r++ {
		g, w := got[r*width:(r+1)*width], want[r*width:(r+1)*width]
		worst = min(worst, cosine(g, w))
	}
	if worst < 0.99 {
		t.Fatalf("worst soft token's cosine with Hugging Face's: %.5f", worst)
	}
	t.Logf("%d soft tokens, worst cosine %.5f, all %.5f", len(want)/width, worst, cosine(got, want))

	// the prompt with the image's placeholders (ids.bin) on the placed chunk, and the logits
	// after it as Hugging Face's (logits.bin), when the check has them
	raw, err := os.ReadFile(filepath.Join(check, "ids.bin"))
	if err != nil {
		return
	}
	ids := make([]int32, len(raw)/4)
	for i := range ids {
		ids[i] = int32(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	logits, err := m.PromptPlaced(ctx, ids, 0, []int32{0})
	if err != nil {
		t.Fatal(err)
	}
	wl := f32("logits.bin")
	if cos := cosine(logits[:len(wl)], wl); cos < 0.99 {
		t.Fatalf("logits after the image: cosine %.5f with Hugging Face's", cos)
	} else {
		t.Logf("logits after %d tokens with the image: cosine %.5f, top %d (Hugging Face %d)", len(ids), cos, argmax(logits[:len(wl)]), argmax(wl))
	}
}

func argmax(x []float32) int {
	best := 0
	for i, v := range x {
		if v > x[best] {
			best = i
		}
	}
	return best
}

// LFM3-Audio's audio encoder on the NPU gives the soft tokens Hugging Face's does: an export in
// HWNPU_IRON_AUDIO, and in HWNPU_IRON_AUDIO_CHECK a clip's log-mel features (features.bin,
// float32, frames by mels) and the soft tokens Hugging Face computed from them (soft.bin).
func TestLfm3AudioTowerOnDevice(t *testing.T) {
	dir, check := os.Getenv("HWNPU_IRON_AUDIO"), os.Getenv("HWNPU_IRON_AUDIO_CHECK")
	if dir == "" || check == "" {
		t.Skip("HWNPU_IRON_AUDIO and HWNPU_IRON_AUDIO_CHECK are not set")
	}
	ctx := context.Background()
	dev, err := hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD))
	if err != nil {
		t.Skip(err)
	}
	defer dev.Close()
	m, err := Open(ctx, dev, dir, t.Logf)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	mm := m.Manifest.Multimodal
	if mm == nil || mm.Kind != "lfm3audio" || !m.HasAudio() {
		t.Fatalf("%s is no LFM3-Audio export", dir)
	}
	f32 := func(name string) []float32 {
		b, err := os.ReadFile(filepath.Join(check, name))
		if err != nil {
			t.Fatal(err)
		}
		out := make([]float32, len(b)/4)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
		}
		return out
	}
	mels := mm.Audio.Mels
	feats, want := f32("features.bin"), f32("soft.bin")
	frames := len(feats) / mels
	buf := make([]byte, 2*(mm.Audio.MaxFrames+2)*mels)
	for i, x := range feats {
		binary.LittleEndian.PutUint16(buf[2*(mels+i):], bf16(x))
	}
	n1 := (frames + 1) / 2
	n2 := (n1 + 1) / 2
	tokens := (n2 + 1) / 2
	input := towerInput(mm, mm.AudioToken)
	own := map[string]int64{input + "_n1": int64(n1), input + "_n2": int64(n2), input + "_tokens": int64(tokens)}
	if err := m.runTower(ctx, m.audio, input, buf, own, frames-1); err != nil {
		t.Fatal(err)
	}
	merged, _, err := m.Buffer("audio", "merged")
	if err != nil {
		t.Fatal(err)
	}
	width := m.Manifest.Config.HiddenSize
	at := m.Manifest.Config.PrefillChunk
	if len(want) != tokens*width {
		t.Fatalf("%d soft values for %d tokens", len(want), tokens)
	}
	worst := 1.0
	for r := range tokens {
		g := make([]float32, width)
		for i := range g {
			g[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(merged[2*((at+r)*width+i):])) << 16)
		}
		c := cosine(g, want[r*width:(r+1)*width])
		worst = min(worst, c)
		if testing.Verbose() {
			t.Logf("soft token %d: cosine %.5f", r, c)
		}
	}
	if worst < 0.98 {
		t.Fatalf("worst soft token's cosine with Hugging Face's: %.5f", worst)
	}
	t.Logf("%d frames, %d soft tokens, worst cosine %.5f", frames, tokens, worst)
}
