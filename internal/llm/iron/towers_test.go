package iron

import "testing"

// testMultimodal is Gemma 4 E2B's processor settings.
func testMultimodal() *Model {
	mm := &Multimodal{}
	mm.Vision.PoolingKernelSize, mm.Vision.PatchSize, mm.Vision.ImageTokens = 3, 16, 280
	mm.Audio.SampleRate, mm.Audio.Frame, mm.Audio.Hop, mm.Audio.MaxSamples, mm.Audio.MaxTokens = 16000, 320, 160, 480000, 768
	return &Model{Manifest: &Manifest{Multimodal: mm}, vision: &version{}, audio: &version{}}
}

// TestImageSize compares with IRON's VisionConfig.size and tokens (Gemma 4's processor).
func TestImageSize(t *testing.T) {
	m := testMultimodal()
	for _, c := range []struct{ w, h, ow, oh, tokens int }{
		{640, 480, 912, 672, 266}, {1920, 1080, 1056, 576, 264}, {300, 2000, 288, 2064, 258},
		{48, 48, 768, 768, 256}, {4000, 100, 5040, 96, 210}, {1000, 1000, 768, 768, 256},
	} {
		ow, oh, n, err := m.ImageSize(c.w, c.h)
		if err != nil || ow != c.ow || oh != c.oh || n != c.tokens {
			t.Errorf("%dx%d: %dx%d %d tokens (%v), IRON %dx%d %d", c.w, c.h, ow, oh, n, err, c.ow, c.oh, c.tokens)
		}
	}
}

// TestAudioTokens compares with IRON's AudioConfig.frames and tokens.
func TestAudioTokens(t *testing.T) {
	m := testMultimodal()
	for _, c := range []struct{ n, frames, tokens int }{{16000, 99, 25}, {22050, 137, 35}, {480000, 2999, 750}, {600000, 2999, 750}, {500, 3, 1}} {
		f, n, err := m.AudioTokens(c.n)
		if err != nil || f != c.frames || n != c.tokens {
			t.Errorf("%d samples: %d frames %d tokens (%v), IRON %d %d", c.n, f, n, err, c.frames, c.tokens)
		}
	}
	if _, _, err := m.AudioTokens(100); err == nil {
		t.Error("a clip of no frame was accepted")
	}
}
