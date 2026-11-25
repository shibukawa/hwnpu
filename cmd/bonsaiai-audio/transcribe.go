package main

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/shibukawa/bonsaiai/ai"
	"github.com/shibukawa/bonsaiai/ai/speech"
	"github.com/shibukawa/bonsaiai/internal/console"
	"github.com/shibukawa/bonsaiai/internal/mem"
	"github.com/shibukawa/bonsaiai/internal/transcribe"
)

// TranscribeCmd is "bonsaiai-audio transcribe": speech to text with Whisper on the Neural Engine.
// The first run downloads the checkpoint (1.6 GB for whisper-large-v3-turbo) and
// converts its three programs (the encoder, the cross attention's keys and values, the decoder),
// in 8 bit tables by default. A WAV of any rate and channels is read (averaged to mono and
// resampled to 16 kHz), 30 seconds at a time; each window's language is detected unless
// --language says it, and the text is greedy (internal/transcribe).
type TranscribeCmd struct {
	Files      []string `arg:"" type:"existingfile" name:"FILE.wav" help:"The WAV files of speech."`
	Model      string   `short:"m" default:"transcribe" placeholder:"NAME" help:"The Whisper model: a name, a use or a family of models.yaml (transcribe, whisper:large-v3-turbo), a Hugging Face repository OWNER/NAME, or on AMD's NPU an export's directory (tools/amd/iron_whisper.py)."`
	Language   string   `short:"l" placeholder:"CODE" help:"The language of the speech (ja, en, ...); none: detected in each 30 seconds."`
	Translate  bool     `help:"Turn the speech into English text instead of its own language's."`
	Timestamps bool     `help:"Print each segment with its start and end."`
	Format     string   `placeholder:"FORMAT" help:"Weight format of the programs: lut8 (default), lut6, lut4 or fp16."`
	MaxTokens  int      `name:"max-tokens" placeholder:"N" help:"Most tokens of text per 30 seconds (0: as many as the decoder holds, 444)."`
	Verbose    bool     `short:"v" help:"Log each window's language and times."`
	Models     string   `placeholder:"FILES" help:"${models_usage}."`
	CPU        bool     `name:"cpu" help:"Run on the CPU interpreter instead of the Neural Engine (slow)."`
}

// Run writes the text of each file.
func (c *TranscribeCmd) Run(ctx context.Context) error {
	lw := console.Log(os.Stderr)
	logf := func(format string, args ...any) { fmt.Fprintf(lw, format+"\n", args...) }
	st := store(c.Models)
	use, defs, err := c.use(st)
	if err != nil {
		return err
	}
	st.Definitions = defs
	app := ai.NewApp("bonsaiai-audio", use)
	app.Store = st
	rt, err := open(ctx, app, lw)
	if err != nil {
		return err
	}
	defer rt.Close()
	wired0, wired := mem.NPUWired()
	t, err := use.Get(ctx, rt)
	if err != nil {
		return err
	}
	logf("%s", loadedLine(rt, t.OnCPU(), t.OnNPU()))
	console.Running()
	o := speech.TranscribeOptions{Language: c.Language, Translate: c.Translate, MaxTokens: c.MaxTokens}
	for _, p := range c.Files {
		raw, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		samples, err := speech.ReadWAV(raw, speech.TranscribeRate) // mono, at 16 kHz
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		res, err := t.Transcribe(ctx, samples, o)
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		if len(c.Files) > 1 {
			fmt.Printf("%s: ", p)
		}
		if c.Timestamps {
			fmt.Println()
			for _, seg := range res.Segments {
				fmt.Printf("[%s - %s] %s\n", clock(seg.Start), clock(seg.End), seg.Text)
			}
		} else {
			fmt.Println(res.Text)
		}
		secs := float64(len(samples)) / speech.TranscribeRate
		el := res.Took.Seconds()
		if c.Verbose {
			for _, w := range res.Windows {
				logf("  %v %s: %d tokens; features %.0f ms, encoder and cross %.0f ms, decoding %.0f ms (%.1f tokens/s)",
					w.Start, w.Lang, w.Tokens, ms(w.Features), ms(w.Encode), ms(w.Decode), float64(w.Tokens+4)/w.Decode.Seconds())
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

// use is the use of the model --model names, as the flags make it: a model of its own for
// --format and --cpu, or for an export's directory.
func (c *TranscribeCmd) use(st ai.Store) (ai.Use[*speech.Transcriber], []ai.Definition, error) {
	var d ai.Definition
	name := c.Model
	if dir := expandHome(c.Model); transcribe.IsIronExport(dir) {
		// an export of tools/amd/iron_whisper.py, for AMD's NPU: a model like the definitions' own
		d.Programs = map[string][]string{"iron": {absolute(dir)}}
		name = "transcribe"
	}
	use, defs, err := useFor(st, speech.Transcribe(name))
	if err != nil {
		return use, nil, err
	}
	if c.Format != "" {
		d.Settings = map[string]string{"format": c.Format}
	}
	if c.CPU {
		d.Device = "cpu"
	}
	if d.Programs == nil && d.Settings == nil && d.Device == "" {
		return use, defs, nil
	}
	d.Like = use.Name()
	return speech.Transcribe(d.ID()), append(defs, d), nil
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

// clock writes d as minutes, seconds and hundredths.
func clock(d time.Duration) string {
	cs := d.Milliseconds() / 10
	return fmt.Sprintf("%02d:%02d.%02d", cs/6000, cs/100%60, cs%100)
}
