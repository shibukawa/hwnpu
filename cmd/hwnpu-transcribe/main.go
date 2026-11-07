// hwnpu-transcribe turns speech into text with Whisper on the Neural Engine.
//
//	hwnpu-transcribe speech.wav                      # whisper-large-v3-turbo, the language detected
//	hwnpu-transcribe -language ja a.wav b.wav
//	hwnpu-transcribe -translate speech.wav           # into English
//	hwnpu-transcribe -use kotoba-tech/kotoba-whisper-v2.2 speech.wav
//
// The first run downloads the checkpoint (1.6 GB for whisper-large-v3-turbo) into -dir and
// converts its three programs (the encoder, the cross attention's keys and values, the decoder),
// in 8 bit tables by default. A WAV of any rate and channels is read (averaged to mono and
// resampled to 16 kHz), 30 seconds at a time; each window's language is detected unless -language
// says it, and the text is greedy, without timestamps (internal/transcribe).
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3audio"
	"github.com/shibukawa/hwnpu/internal/llm/whisper"
	"github.com/shibukawa/hwnpu/internal/mem"
	"github.com/shibukawa/hwnpu/internal/transcribe"
)

func main() {
	use := flag.String("use", transcribe.DefaultRepo, "the Whisper checkpoint: a Hugging Face repository OWNER/NAME")
	dir := flag.String("dir", chat.ModelsDir(), "where the models are kept (as hwnpu-chat's -dir): the checkpoint and its programs in NAME/")
	language := flag.String("language", "", "language code of the speech (ja, en, ...); none: detected in each 30 seconds")
	translate := flag.Bool("translate", false, "turn the speech into English text instead of its own language's")
	format := flag.String("format", "", "weight format of the programs: lut8 (default), lut6, lut4 or fp16")
	maxTokens := flag.Int("max-tokens", 0, "most tokens of text per 30 seconds (0: as many as the decoder holds, 444)")
	cpu := flag.Bool("cpu", false, "run on the CPU interpreter instead of the Neural Engine (slow)")
	cacheDir := flag.String("cache", "", "compile cache (default: hwnpu in the user cache directory)")
	verbose := flag.Bool("v", false, "print each window's language and times")
	stamps := flag.Bool("timestamps", false, "print each segment with its start and end (a clip over 30 seconds is decoded in segments)")
	flag.Parse()
	if flag.NArg() == 0 || strings.Count(*use, "/") != 1 {
		flag.Usage()
		os.Exit(2)
	}
	if *cacheDir == "" {
		if d, err := os.UserCacheDir(); err == nil {
			*cacheDir = filepath.Join(d, "hwnpu")
		}
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	s := transcribe.Setup{Dir: filepath.Join(*dir, filepath.Base(*use)), Repo: *use, Format: *format, Log: os.Stderr}
	o := transcribe.Options{Language: *language, Translate: *translate, MaxTokens: *maxTokens}
	if err := run(ctx, s, o, *cpu, *cacheDir, *verbose, *stamps, flag.Args()); err != nil {
		fmt.Fprintln(os.Stderr, "hwnpu-transcribe:", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, s transcribe.Setup, o transcribe.Options, cpu bool, cacheDir string, verbose, stamps bool, paths []string) error {
	c, err := transcribe.Prepare(ctx, s)
	if err != nil {
		return err
	}
	var dev *hwnpu.Device
	if !cpu {
		if dev, _, err = llm.OpenDevice(ctx, "", cacheDir); err != nil {
			return err
		}
		defer dev.Close()
	}
	wired0, wired := mem.NPUWired()
	t0 := time.Now()
	t, err := transcribe.Load(ctx, dev, c)
	if err != nil {
		return err
	}
	defer t.Close()
	fmt.Fprintf(os.Stderr, "loaded %s in %.1f s (on the NPU: %v)\n", filepath.Base(s.Dir), time.Since(t0).Seconds(), t.OnNPU())
	for _, p := range paths {
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
		if len(paths) > 1 {
			fmt.Printf("%s: ", p)
		}
		if stamps {
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
		if verbose {
			for _, w := range res.Windows {
				fmt.Fprintf(os.Stderr, "  %v %s: %d tokens; features %.0f ms, encoder and cross %.0f ms, decoding %.0f ms (%.1f tokens/s)\n",
					w.Start, w.Language, len(w.Tokens), ms(w.Features), ms(w.Encode), ms(w.Decode), float64(len(w.Tokens)+4)/w.Decode.Seconds())
			}
		}
		fmt.Fprintf(os.Stderr, "%s: %.1f s of audio in %.2f s (%.1fx real time)\n", p, secs, el, secs/el)
	}
	if w, ok := mem.NPUWired(); wired && ok && verbose {
		// programs compiled just now come onto the Neural Engine at their first call
		fmt.Fprintf(os.Stderr, "the Neural Engine holds %.2f GiB for the programs, kernel memory of its driver outside this process's footprint\n", float64(w-wired0)/(1<<30))
	}
	return nil
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

// clock writes d as minutes, seconds and hundredths.
func clock(d time.Duration) string {
	cs := d.Milliseconds() / 10
	return fmt.Sprintf("%02d:%02d.%02d", cs/6000, cs/100%60, cs%100)
}
