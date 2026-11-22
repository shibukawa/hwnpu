package supertonic

import (
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/onnx"
)

// Arch names the programs in their metadata; Revision counts the changes to them that call for
// converting them again.
const (
	Arch     = "supertonic3"
	Revision = 4
)

// Size is a program's text (T, characters with the language's tokens) and latent (L, frames
// of Chunk samples: 70 ms).
type Size struct{ T, L int }

// Sizes are the programs made for the NPU, the smallest that holds a text runs it: up to
// about 2.2, 4.5 and 8.9 seconds of speech. A text longer than the largest is spoken a
// sentence at a time (Speaker).
var Sizes = []Size{{64, 32}, {128, 64}, {256, 128}}

// Voices are the voices of the model, voice_styles/NAME.json.
var Voices = []string{"F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"}

// files are the files of the model it reads, with their sha256 at the revision of models.yaml.
var files = map[string]string{
	"onnx/duration_predictor.onnx": "c3eb91414d5ff8a7a239b7fe9e34e7e2bf8a8140d8375ffb14718b1c639325db",
	"onnx/text_encoder.onnx":       "c7befd5ea8c3119769e8a6c1486c4edc6a3bc8365c67621c881bbb774b9902ff",
	"onnx/vector_estimator.onnx":   "883ac868ea0275ef0e991524dc64f16b3c0376efd7c320af6b53f5b780d7c61c",
	"onnx/vocoder.onnx":            "085de76dd8e8d5836d6ca66826601f615939218f90e519f70ee8a36ed2a4c4ba",
	"onnx/tts.json":                "42078d3aef1cd43ab43021f3c54f47d2d75ceb4e75f627f118890128b06a0d09",
	"onnx/unicode_indexer.json":    "9bf7346e43883a81f8645c81224f786d43c5b57f3641f6e7671a7d6c493cb24f",
}

// Setup says what Prepare downloads and converts.
type Setup struct {
	// Repo is where the model is downloaded from (chat.Fetch's OWNER/NAME@REVISION); Dir is
	// where its programs are kept.
	Repo, Dir string
	// Backend is the programs' format (llm.Options.Backend); "" for none: the CPU alone.
	Backend string
	Log     io.Writer
}

// Config is what Load reads: the model's directory (onnx/, voice_styles/) and the programs
// of Sizes, of each stage (none for the CPU alone: ONNX Runtime).
type Config struct {
	Dir                                       string
	Durations, Encoders, Estimators, Vocoders []string
	// AMD is the export of PrepareAMD (the graphs), run on Device, AMD's NPU.
	AMD     string
	Device  *hwnpu.Device // nil: the CPU
	Threads int           // ONNX Runtime's threads on the CPU (0: 4)
}

// stage is a graph made into programs of Sizes.
type stage struct {
	name, file string
	program    func(g *onnx.Graph, s Size) *onnx.Program
	paths      func(c *Config) *[]string
	// text says the program is of the text alone (its name has no L): one of each T
	text bool
}

var stages = []stage{
	{"duration", Graphs.Duration, func(g *onnx.Graph, s Size) *onnx.Program { return DurationProgram(g, s.T) }, func(c *Config) *[]string { return &c.Durations }, true},
	{"encoder", Graphs.Encoder, func(g *onnx.Graph, s Size) *onnx.Program { return EncoderProgram(g, s.T) }, func(c *Config) *[]string { return &c.Encoders }, true},
	{"estimator", Graphs.Estimator, func(g *onnx.Graph, s Size) *onnx.Program { return EstimatorProgram(g, s.T, s.L) }, func(c *Config) *[]string { return &c.Estimators }, false},
	{"vocoder", Graphs.Vocoder, func(g *onnx.Graph, s Size) *onnx.Program { return VocoderProgram(g, s.L) }, func(c *Config) *[]string { return &c.Vocoders }, false},
}

// Prepare downloads the model's files unless they are there and checks them, and converts the
// programs of Sizes for the backend unless they are.
func Prepare(ctx context.Context, s Setup) (Config, error) {
	log := cmp.Or[io.Writer](s.Log, io.Discard)
	var c Config
	names := []string{}
	for f := range files {
		names = append(names, f)
	}
	for _, v := range Voices {
		names = append(names, "voice_styles/"+v+".json")
	}
	for _, f := range names {
		p, err := chat.Fetch(ctx, s.Repo, f, log)
		if err != nil {
			return c, err
		}
		if want, ok := files[f]; ok {
			if err := checkSHA(p, want); err != nil {
				return c, err
			}
		}
		c.Dir = strings.TrimSuffix(p, filepath.FromSlash(f))
	}
	if s.Backend == "" {
		return c, nil
	}
	for _, st := range stages {
		var g *onnx.Graph
		for _, size := range Sizes {
			name := fmt.Sprintf("%s-r%d-%s-t%d-l%d-fp16", Arch, Revision, st.name, size.T, size.L)
			about := fmt.Sprintf("a text of %d characters and %d frames", size.T, size.L)
			switch {
			case st.text:
				name, about = fmt.Sprintf("%s-r%d-%s-t%d-fp16", Arch, Revision, st.name, size.T), fmt.Sprintf("a text of %d characters", size.T)
			case st.name == "vocoder":
				name, about = fmt.Sprintf("%s-r%d-%s-l%d-fp16", Arch, Revision, st.name, size.L), fmt.Sprintf("%d frames", size.L)
			}
			program, done, export := llm.ProgramPath(s.Dir, name, s.Backend)
			paths := st.paths(&c)
			*paths = append(*paths, program)
			if llm.ProgramReady(ctx, s.Dir, name, done, export) {
				continue
			}
			if g == nil {
				var err error
				if g, err = onnx.Load(filepath.Join(c.Dir, "onnx", st.file)); err != nil {
					return c, err
				}
			}
			var sin []byte // the estimator's hostSin
			if st.name == "estimator" {
				h, err := sinOf(g)
				if err != nil {
					return c, err
				}
				sin, _ = json.Marshal(h)
			}
			p := st.program(g, size)
			bar := console.NewBar(log, "converting "+name)
			if err := llm.StartProgram(ctx, program, export); err != nil {
				bar.Done("failed")
				return c, err
			}
			if _, err := llm.WritePart(program, p, nil, llm.Options{Format: "fp16", Backend: s.Backend}, func(pkg *mlpackage.Package) {
				pkg.Description = fmt.Sprintf("Supertonic 3's %s, %s, fp16 (hwnpu)", st.name, about)
				pkg.Metadata[llm.MetaArch] = Arch
				if sin != nil {
					pkg.Metadata[metaSin] = string(sin)
				}
			}); err != nil {
				bar.Done("failed")
				return c, err
			}
			bar.Done(console.Duration(bar.Elapsed()))
			if err := llm.FinishProgram(ctx, s.Dir, name, done, export); err != nil {
				return c, err
			}
		}
	}
	return c, nil
}

func checkSHA(path, want string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return err
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != want {
		return fmt.Errorf("supertonic: %s is not the file of models.yaml's revision (sha256 %s, want %s)", path, got, want)
	}
	return nil
}

// Load opens the model: its programs on the device, with no ONNX Runtime (a text longer than
// they hold is ErrTooLong), or on AMD's NPU the export c.AMD, or its graphs on the CPU with ONNX
// Runtime without them.
func Load(ctx context.Context, c Config) (*Synth, error) {
	ix, err := LoadIndexer(filepath.Join(c.Dir, "onnx", "unicode_indexer.json"))
	if err != nil {
		return nil, err
	}
	s := &Synth{Indexer: ix, Dir: c.Dir}
	if c.AMD != "" {
		if c.Device == nil {
			return nil, fmt.Errorf("supertonic: the export %s is for AMD's NPU, and no device is open", c.AMD)
		}
		if s.Stages, err = OpenAMD(ctx, c.Device, c, c.AMD); err != nil {
			return nil, err
		}
		return s, nil
	}
	if c.Device != nil && len(c.Estimators) > 0 {
		if s.Stages, err = OpenNPU(ctx, c.Device, nil, c); err != nil {
			return nil, err
		}
		return s, nil
	}
	if s.Stages, err = OpenCPU(filepath.Join(c.Dir, "onnx"), cmp.Or(c.Threads, 4)); err != nil {
		return nil, err
	}
	return s, nil
}

// Voice is a voice of the model by its name (F1 ... M5), or female (F1) and male (M1).
func (s *Synth) Voice(name string) (Style, error) {
	switch strings.ToLower(name) {
	case "", "female":
		name = "F1"
	case "male":
		name = "M1"
	}
	name = strings.ToUpper(name)
	return LoadStyle(filepath.Join(s.Dir, "voice_styles", name+".json"))
}
