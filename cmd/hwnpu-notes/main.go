// hwnpu-notes transcribes music in a WAV file into notes with Basic Pitch (spotify/basic-pitch)
// on the NPU, and writes them as MIDI.
//
//	hwnpu-notes song.wav                 # the notes as CSV: start, end, pitch, velocity
//	hwnpu-notes -midi song.mid song.wav  # also a MIDI file, with pitch bends
//	hwnpu-notes -json song.wav           # JSON, with the pitch bends
//
// The model (230 KB) is downloaded and converted the first time. The constant-Q transform runs
// on the host, the network on the NPU, the making of notes on the host.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"os/signal"
	"path/filepath"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3audio"
	"github.com/shibukawa/hwnpu/internal/notes"
)

func main() {
	use := flag.String("use", "notes", "the model: a name, a use or a family of -list (notes, basic-pitch)")
	dir := flag.String("dir", defaultDir(), "where the models are kept, each in a directory of its name")
	o := notes.DefaultOptions
	flag.Float64Var(&o.OnsetThreshold, "onset", o.OnsetThreshold, "the least onset activation of a note's start")
	flag.Float64Var(&o.FrameThreshold, "frame", o.FrameThreshold, "the least note activation that keeps a note on")
	flag.Float64Var(&o.MinNoteMS, "min-note", o.MinNoteMS, "the shortest note, in milliseconds")
	flag.Float64Var(&o.MinHz, "fmin", 0, "drop the notes below this frequency, in Hz (0: none)")
	flag.Float64Var(&o.MaxHz, "fmax", 0, "drop the notes above this frequency, in Hz (0: none)")
	flag.BoolVar(&o.Melodia, "melodia", o.Melodia, "also take notes from the strong activations left without an onset (basic-pitch's melodia trick)")
	midi := flag.String("midi", "", "write a MIDI file (one input)")
	asJSON := flag.Bool("json", false, "write JSON instead of CSV")
	cpu := flag.Bool("cpu", false, "run the program on the CPU interpreter instead of the NPU")
	cacheDir := flag.String("cache", "", "compile cache (default: hwnpu in the user cache directory)")
	list := flag.Bool("list", false, "list the note transcription models of the definitions and exit")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "usage: hwnpu-notes [flags] FILE.wav...\n\nThe notes of each file (any WAV; resampled to 22050 Hz, channels averaged).\n\n")
		flag.PrintDefaults()
	}
	flag.Parse()
	if *list {
		chat.List(os.Stdout, *dir, chat.KindNotes)
		return
	}
	if flag.NArg() == 0 || *midi != "" && flag.NArg() != 1 {
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
	if err := run(ctx, *use, *dir, *cacheDir, *cpu, o, *midi, *asJSON, flag.Args()); err != nil {
		fmt.Fprintln(os.Stderr, "hwnpu-notes:", err)
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

func run(ctx context.Context, use, dir, cacheDir string, cpu bool, o notes.Options, midi string, asJSON bool, paths []string) error {
	m, err := chat.ResolveFor(use, chat.KindNotes, os.Stderr)
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
			return fmt.Errorf("%s runs neither Core ML packages nor OpenVINO IR (AMD's NPU has no Basic Pitch export yet; -cpu runs it on the CPU)", device)
		}
	}
	model, program, err := notes.Prepare(ctx, notes.Setup{Dir: filepath.Join(dir, m.DirName()), Repo: m.Source(), SHA256: m.SHA256[notes.ModelFile], Backend: backend, Log: os.Stderr})
	if err != nil {
		return err
	}
	t0 := time.Now()
	tr, err := notes.Open(ctx, dev, model, program)
	if err != nil {
		return err
	}
	defer tr.Close()
	fmt.Fprintf(os.Stderr, "loaded %s in %.1f s (on the NPU: %v)\n", m.Name, time.Since(t0).Seconds(), tr.OnNPU())
	for _, p := range paths {
		raw, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		x, err := lfm3audio.ReadWAVAt(raw, notes.SampleRate) // mono
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		start := time.Now()
		a, err := tr.Run(ctx, x)
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		ns := notes.Notes(a, o)
		secs := float64(len(x)) / notes.SampleRate
		fmt.Fprintf(os.Stderr, "%s: %.1f s of audio, %d notes in %.2f s (%.0fx real time)\n", p, secs, len(ns), time.Since(start).Seconds(), secs/time.Since(start).Seconds())
		switch {
		case asJSON:
			if err := json.NewEncoder(os.Stdout).Encode(struct {
				File  string       `json:"file"`
				Notes []notes.Note `json:"notes"`
			}{p, ns}); err != nil {
				return err
			}
		default:
			if len(paths) > 1 {
				fmt.Printf("# %s\n", p)
			}
			fmt.Println("start,end,pitch,velocity")
			for _, n := range ns {
				fmt.Printf("%.3f,%.3f,%d,%d\n", n.Start, n.End, n.Pitch, int(math.RoundToEven(127*n.Amplitude)))
			}
		}
		if midi != "" {
			f, err := os.Create(midi)
			if err != nil {
				return err
			}
			if err := notes.WriteMIDI(f, ns); err != nil {
				f.Close()
				return err
			}
			if err := f.Close(); err != nil {
				return err
			}
		}
	}
	return nil
}
