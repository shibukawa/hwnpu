// hwnpu-pitch estimates the pitch (F0) of a voice or an instrument in a WAV file with SwiftF0
// (lars76/swift-f0) on the NPU: a pitch and a voicing confidence every 16 ms.
//
//	hwnpu-pitch voice.wav                 # CSV: timestamp, pitch_hz, confidence, voiced, loudness_db
//	hwnpu-pitch -fmin 80 -fmax 800 a.wav  # candidates within the band only
//	hwnpu-pitch -json a.wav               # JSON
//
// The model (140 KB) is downloaded and its program converted the first time. The front end (the
// three spectrograms) and the decoding of the logits run on the host; the CNN on the NPU.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3audio"
	"github.com/shibukawa/hwnpu/internal/pitch"
)

func main() {
	use := flag.String("use", "pitch", "the model: a name, a use or a family of -list (pitch, swiftf0)")
	dir := flag.String("dir", defaultDir(), "where the models are kept, each in a directory of its name")
	fmin := flag.Float64("fmin", 0, "the lowest pitch to consider, in Hz (0: 46.875)")
	fmax := flag.Float64("fmax", 0, "the highest pitch to consider, in Hz (0: 2093.75)")
	threshold := flag.Float64("threshold", 0.5, "the confidence at and over which a frame is voiced")
	asJSON := flag.Bool("json", false, "write JSON instead of CSV")
	cpu := flag.Bool("cpu", false, "run the program on the CPU interpreter instead of the NPU")
	cacheDir := flag.String("cache", "", "compile cache (default: hwnpu in the user cache directory)")
	list := flag.Bool("list", false, "list the pitch models of the definitions and exit")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "usage: hwnpu-pitch [flags] FILE.wav...\n\nThe pitch every 16 ms of each file (any WAV; resampled to 16 kHz, channels averaged).\n\n")
		flag.PrintDefaults()
	}
	flag.Parse()
	if *list {
		chat.List(os.Stdout, *dir, chat.KindPitch)
		return
	}
	if flag.NArg() == 0 {
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
	if err := run(ctx, *use, *dir, *cacheDir, *cpu, *fmin, *fmax, *threshold, *asJSON, flag.Args()); err != nil {
		fmt.Fprintln(os.Stderr, "hwnpu-pitch:", err)
		os.Exit(1)
	}
}

func defaultDir() string {
	d, err := os.UserCacheDir()
	if err != nil {
		return "models"
	}
	return filepath.Join(d, "hwnpu", "models")
}

func run(ctx context.Context, use, dir, cacheDir string, cpu bool, fmin, fmax, threshold float64, asJSON bool, paths []string) error {
	m, err := chat.ResolveFor(use, chat.KindPitch, os.Stderr)
	if err != nil {
		return err
	}
	backend := llm.BackendCoreML
	var dev *hwnpu.Device
	if !cpu {
		var device string
		if dev, device, err = llm.OpenDevice(ctx, "", cacheDir); err != nil {
			return err
		}
		defer dev.Close()
		if backend = llm.ProgramBackend(device); backend == "" {
			return fmt.Errorf("%s runs neither Core ML packages nor OpenVINO IR (AMD's NPU has no SwiftF0 export yet; -cpu runs it on the CPU)", device)
		}
	}
	model, program, err := pitch.Prepare(ctx, pitch.Setup{Dir: filepath.Join(dir, m.DirName()), Repo: m.Source(), Backend: backend, Log: os.Stderr})
	if err != nil {
		return err
	}
	t0 := time.Now()
	d, err := pitch.Open(ctx, dev, model, program)
	if err != nil {
		return err
	}
	defer d.Close()
	fmt.Fprintf(os.Stderr, "loaded %s in %.1f s (on the NPU: %v)\n", m.Name, time.Since(t0).Seconds(), d.OnNPU())
	for _, p := range paths {
		raw, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		x, err := lfm3audio.ReadWAV(raw) // mono at 16 kHz, SwiftF0's rate too
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		start := time.Now()
		r, err := d.Pitch(ctx, x, fmin, fmax)
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		el := time.Since(start)
		if err := write(p, r, threshold, asJSON, len(paths) > 1); err != nil {
			return err
		}
		secs := float64(len(x)) / pitch.SampleRate
		fmt.Fprintf(os.Stderr, "%s: %.1f s of audio, %d frames in %.3f s (%.0fx real time)\n", p, secs, len(r.PitchHz), el.Seconds(), secs/el.Seconds())
	}
	return nil
}

func write(path string, r pitch.Result, threshold float64, asJSON, named bool) error {
	period := float64(pitch.Hop) / pitch.SampleRate
	if asJSON {
		type frame struct {
			Time       float64 `json:"timestamp"`
			Pitch      float64 `json:"pitch_hz"`
			Confidence float64 `json:"confidence"`
			Voiced     bool    `json:"voiced"`
			Loudness   float64 `json:"loudness_db"`
		}
		out := struct {
			File   string  `json:"file,omitempty"`
			Frames []frame `json:"frames"`
		}{}
		if named {
			out.File = path
		}
		for i := range r.PitchHz {
			out.Frames = append(out.Frames, frame{float64(i) * period, r.PitchHz[i], r.Confidence[i], r.Confidence[i] >= threshold, r.LoudnessDB[i]})
		}
		enc := json.NewEncoder(os.Stdout)
		return enc.Encode(out)
	}
	if named {
		fmt.Printf("# %s\n", path)
	}
	fmt.Println("timestamp,pitch_hz,confidence,voiced,loudness_db")
	for i := range r.PitchHz {
		fmt.Printf("%.4f,%.2f,%.4f,%v,%.1f\n", float64(i)*period, r.PitchHz[i], r.Confidence[i], r.Confidence[i] >= threshold, r.LoudnessDB[i])
	}
	return nil
}
