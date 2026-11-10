package sfx

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/internal/llm"
)

// On the Neural Engine (HWNPU_SFX_DEVICE), the prepared programs (HWNPU_SFX_DIR, the model's
// directory, prepared by hwnpu-audio generate -m sfx) generate the reference's sound from its noise in 4 steps
// (tools/ref/ref_moss_sfx.py run --steps 4 --seconds 10, HWNPU_SFX_REF; the prompt in
// HWNPU_SFX_PROMPT).
func TestGenerateOnDevice(t *testing.T) {
	dir, ref := refDir(t)
	prompt := os.Getenv("HWNPU_SFX_PROMPT")
	if prompt == "" || os.Getenv("HWNPU_SFX_DEVICE") == "" {
		t.Skip("HWNPU_SFX_PROMPT and HWNPU_SFX_DEVICE are not set")
	}
	ctx := context.Background()
	dev, _, err := llm.OpenDevice(ctx, "", "")
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	p, err := Prepare(ctx, Setup{Dir: dir, Repo: "OpenMOSS-Team/MOSS-SoundEffect-v2.0", TextFormat: os.Getenv("HWNPU_SFX_TEXT_FORMAT")})
	if err != nil {
		t.Fatal(err)
	}
	g, err := Open(p, dev)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	text, _, err := g.Encode(ctx, prompt, 10, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := g.Load(ctx, nil); err != nil {
		t.Fatal(err)
	}
	noise := rows(readF32(t, ref+".noise.bin"), LatentDim, Frames)
	t0 := time.Now()
	got, err := g.Generate(ctx, text, 10, Options{Steps: 4, Noise: noise})
	if err != nil {
		t.Fatal(err)
	}
	want := readF32(t, ref+".samples.bin")
	t.Logf("%s (%s): in %s, samples %.3g off the reference, cosine %.4f", prompt, p, time.Since(t0).Round(time.Millisecond), relRMS(got, want), cosine(got, want))
}
