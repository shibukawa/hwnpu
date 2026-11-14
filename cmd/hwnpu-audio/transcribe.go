package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3audio"
	"github.com/shibukawa/hwnpu/internal/llm/whisper"
	"github.com/shibukawa/hwnpu/internal/mem"
	"github.com/shibukawa/hwnpu/internal/transcribe"
)

// TranscribeCmd is "hwnpu-audio transcribe": speech to text with Whisper on the Neural Engine.
// The first run downloads the checkpoint (1.6 GB for whisper-large-v3-turbo) into --dir and
// converts its three programs (the encoder, the cross attention's keys and values, the decoder),
// in 8 bit tables by default. A WAV of any rate and channels is read (averaged to mono and
// resampled to 16 kHz), 30 seconds at a time; each window's language is detected unless
// --language says it, and the text is greedy (internal/transcribe).
type TranscribeCmd struct {
	Files      []string `arg:"" type:"existingfile" name:"FILE.wav" help:"The WAV files of speech."`
	Model      string   `short:"m" default:"transcribe" placeholder:"NAME" help:"The Whisper model: a name, a use or a family of models.yaml (transcribe, whisper:large-v3-turbo), or a Hugging Face repository OWNER/NAME."`
	Language   string   `short:"l" placeholder:"CODE" help:"The language of the speech (ja, en, ...); none: detected in each 30 seconds."`
	Translate  bool     `help:"Turn the speech into English text instead of its own language's."`
	Timestamps bool     `help:"Print each segment with its start and end."`
	Format     string   `placeholder:"FORMAT" help:"Weight format of the programs: lut8 (default), lut6, lut4 or fp16."`
	MaxTokens  int      `name:"max-tokens" placeholder:"N" help:"Most tokens of text per 30 seconds (0: as many as the decoder holds, 444)."`
	Verbose    bool     `short:"v" help:"Log each window's language and times."`
	Dir        string   `type:"path" placeholder:"DIR" help:"Where the models are kept (as hwnpu-chat's --dir): the checkpoint and its programs in NAME/."`
	Cache      string   `placeholder:"DIR" help:"Compile cache (default: hwnpu in the user cache directory)."`
	Models     string   `placeholder:"FILES" help:"${models_usage}."`
	CPU        bool     `name:"cpu" help:"Run on the CPU interpreter instead of the Neural Engine (slow)."`
}

// Run writes the text of each file.
func (c *TranscribeCmd) Run(ctx context.Context) error {
	lw := console.Log(os.Stderr)
	logf := func(format string, args ...any) { fmt.Fprintf(lw, format+"\n", args...) }
	m, err := resolveKind(c.Model, chat.KindTranscribe, c.Models, lw)
	if err != nil {
		return err
	}
	s := transcribe.Setup{Dir: filepath.Join(modelsDir(c.Dir), m.DirName()), Repo: m.Source(), Format: c.Format, Log: lw}
	o := transcribe.Options{Language: c.Language, Translate: c.Translate, MaxTokens: c.MaxTokens}
	var dev *hwnpu.Device
	if c.CPU {
		s.Backend = llm.BackendCoreML // what the CPU interpreter runs
	} else {
		if dev, s.Backend, err = openAudioDevice(ctx, cacheDir(c.Cache), "--cpu runs Whisper on the CPU"); err != nil {
			return err
		}
		defer dev.Close()
	}
	cfg, err := transcribe.Prepare(ctx, s)
	if err != nil {
		return err
	}
	wired0, wired := mem.NPUWired()
	t0 := time.Now()
	t, err := transcribe.Load(ctx, dev, cfg)
	if err != nil {
		return err
	}
	defer t.Close()
	logf("loaded %s in %.1f s, %s", m.Name, time.Since(t0).Seconds(), placement(c.CPU, t.OnNPU()))
	console.Running()
	for _, p := range c.Files {
		raw, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		samples, err := lfm3audio.ReadWAV(raw) // mono, at 16 kHz: the rate of both models
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		start := time.Now()
		res, err := t.Transcribe(ctx, samples, o)
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		if len(c.Files) > 1 {
			fmt.Printf("%s: ", p)
		}
		if c.Timestamps {
			fmt.Println()
			for _, w := range res.Windows {
				for _, seg := range w.Segments {
					fmt.Printf("[%s - %s] %s\n", clock(w.Start+seg.Start), clock(w.Start+seg.End), strings.TrimSpace(t.SegmentText(seg)))
				}
			}
		} else {
			fmt.Println(res.Text)
		}
		secs := float64(len(samples)) / whisper.SampleRate
		el := time.Since(start).Seconds()
		if c.Verbose {
			for _, w := range res.Windows {
				logf("  %v %s: %d tokens; features %.0f ms, encoder and cross %.0f ms, decoding %.0f ms (%.1f tokens/s)",
					w.Start, w.Language, len(w.Tokens), ms(w.Features), ms(w.Encode), ms(w.Decode), float64(len(w.Tokens)+4)/w.Decode.Seconds())
			}
		}
		logf("%s: %.1f s of audio in %.2f s (%.1fx real time)", p, secs, el, secs/el)
	}
	if w, ok := mem.NPUWired(); wired && ok && c.Verbose {
		// programs compiled just now come onto the Neural Engine at their first call
		logf("the Neural Engine holds %.2f GiB for the programs, kernel memory of its driver outside this process's footprint", float64(w-wired0)/(1<<30))
	}
	return nil
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

// clock writes d as minutes, seconds and hundredths.
func clock(d time.Duration) string {
	cs := d.Milliseconds() / 10
	return fmt.Sprintf("%02d:%02d.%02d", cs/6000, cs/100%60, cs%100)
}
