package ai

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/internal/llm/lfm3audio"
)

// TestOnDevice prepares and runs the uses of HWNPU_AI_USES (comma separated: transcribe,
// translate, speech) in the store HWNPU_AI_STORE on the machine's NPU: Plan, Run (which
// downloads and converts what is missing, as hwnpu-chat and hwnpu-audio do), Open, and a call of
// each. HWNPU_AI_WAV is speech for transcribe (default: the speech speech makes, or skipped).
// Run it while no other session uses the Neural Engine:
//
//	HWNPU_AI_STORE=~/Library/Caches/hwnpu/models HWNPU_AI_USES=translate go test ./ai -run OnDevice -v
func TestOnDevice(t *testing.T) {
	dir, uses := os.Getenv("HWNPU_AI_STORE"), os.Getenv("HWNPU_AI_USES")
	if dir == "" || uses == "" {
		t.Skip("HWNPU_AI_STORE and HWNPU_AI_USES are not set")
	}
	ctx := context.Background()
	store := Store{Dir: dir}
	names := strings.Split(uses, ",")
	p, err := NewRegistry().Require(names...).Plan(ctx, store)
	if err != nil {
		t.Fatal(err)
	}
	for _, it := range p.Items {
		t.Logf("%s: %s ready %v, about %.2f GB to download, %.1f GB running %s", it.Use, it.Model, it.Ready, float64(it.Download)/1e9, it.MemoryGB, it.Why)
	}
	t0 := time.Now()
	if err := p.Run(ctx, func(pr Progress) {
		if pr.Message != "" {
			t.Logf("[%d/%d %s %s] %s", pr.Item, pr.Items, pr.Use, pr.Stage, pr.Message)
		}
	}); err != nil {
		t.Fatal(err)
	}
	t.Logf("prepared in %.1f s", time.Since(t0).Seconds())

	rt, err := Open(ctx, store, WithLog(testLog{t}))
	if err != nil {
		t.Fatal(err)
	}
	defer rt.Close()
	var spoken []float32
	for _, use := range names {
		t0 := time.Now()
		switch it := p.Items[indexOf(p, use)]; it.Role {
		case RoleTranslate:
			tr, err := rt.Translator(ctx, use)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("translator loaded in %.1f s, on the NPU %v", time.Since(t0).Seconds(), tr.OnNPU())
			t0 = time.Now()
			en, err := tr.Translate(ctx, "今日は良い天気ですね。", TranslateOptions{To: "en"})
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("translated in %.2f s: %q", time.Since(t0).Seconds(), en)
			if en == "" {
				t.Error("empty translation")
			}
		case RoleSpeech:
			sp, err := rt.Speaker(ctx, use)
			if err != nil {
				t.Fatal(err)
			}
			t0 = time.Now()
			a, err := sp.Speak(ctx, "Hello, this is a test.", SpeakOptions{Seed: 1})
			if err != nil {
				t.Fatal(err)
			}
			secs := float64(len(a.Samples)) / float64(a.Rate)
			t.Logf("spoke %.2f s of audio in %.2f s", secs, time.Since(t0).Seconds())
			spoken = resample(a.Samples, a.Rate, TranscribeRate)
		case RoleTranscribe:
			stt, err := rt.Transcriber(ctx, use)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("transcriber loaded in %.1f s, on the NPU %v", time.Since(t0).Seconds(), stt.OnNPU())
			pcm := spoken
			if wav := os.Getenv("HWNPU_AI_WAV"); wav != "" {
				raw, err := os.ReadFile(wav)
				if err != nil {
					t.Fatal(err)
				}
				if pcm, err = lfm3audio.ReadWAV(raw); err != nil {
					t.Fatal(err)
				}
			}
			if pcm == nil {
				t.Log("no speech to transcribe (HWNPU_AI_WAV, or speech before transcribe)")
				continue
			}
			t0 = time.Now()
			tr, err := stt.Transcribe(ctx, pcm, TranscribeOptions{})
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("transcribed in %.2f s (%s): %q", time.Since(t0).Seconds(), tr.Lang, tr.Text)
		}
	}
}

type testLog struct{ t *testing.T }

func (l testLog) Write(b []byte) (int, error) {
	if s := strings.TrimSpace(stripANSI(string(b))); s != "" {
		l.t.Log(s)
	}
	return len(b), nil
}

func indexOf(p *Plan, use string) int {
	for i, it := range p.Items {
		if it.Use == use {
			return i
		}
	}
	return -1
}

// resample is linear resampling, enough to hand speech to a transcriber in a test.
func resample(x []float32, from, to int) []float32 {
	n := len(x) * to / from
	out := make([]float32, n)
	for i := range out {
		p := float64(i) * float64(from) / float64(to)
		j := int(p)
		f := float32(p - float64(j))
		if j+1 < len(x) {
			out[i] = x[j]*(1-f) + x[j+1]*f
		} else if j < len(x) {
			out[i] = x[j]
		}
	}
	return out
}
