package speech

import (
	"cmp"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime/debug"
	"slices"
	"sync"

	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3gen"
)

// DefaultCache is the KV cache of the text model's programs Prepare converts: the TTS prompt and
// about two minutes of audio frames (80 ms each, one position per frame).
const DefaultCache = 2048

// TablesFile holds the tables the host reads while it speaks (the text model's token embedding,
// the depthformer's code tables, the feedback table and the mask embedding), so that it runs
// without the checkpoint. It is apart from chat.TablesFile, which hwnpu-chat writes in the same
// directory for the same checkpoint's text model.
const TablesFile = "speech-tables.safetensors"

// Setup says what Prepare downloads and converts.
type Setup struct {
	// Dir is where the checkpoint and the programs are kept; Repo is the Hugging Face repository
	// it is downloaded from when it is not there (OWNER/NAME, or OWNER/NAME@REVISION).
	Dir, Repo string
	// Format is the weight format of the text model's programs (empty: the architecture's, 8
	// bit tables for LFM3-Audio, which 4 bit ones spoil); the depthformer and the codec decoder
	// stay in fp16.
	Format string
	// Cache is the text model's KV cache (0: DefaultCache).
	Cache int
	Log   io.Writer
}

// files are the checkpoint's files besides the weights; generation_config.json may be missing.
var files = []string{"config.json", "tokenizer.json", "audio_tokenizer/config.json"}

// Prepare downloads the checkpoint into s.Dir unless it is there, converts the programs hwnpu-audio tts
// runs unless they are (the text model with a KV cache and its final hidden state out, 1 and 64
// tokens per call; the depthformer; the codec decoder of 64 frames per call), and writes the
// tables the host reads. The checkpoint is downloaded only when something is converted or written
// from it.
func Prepare(ctx context.Context, s Setup) (Config, error) {
	log := cmp.Or[io.Writer](s.Log, io.Discard)
	if err := os.MkdirAll(filepath.Join(s.Dir, "audio_tokenizer"), 0o755); err != nil {
		return Config{}, err
	}
	for _, f := range files {
		if err := chat.Fetch(ctx, s.Repo, f, filepath.Join(s.Dir, filepath.FromSlash(f)), log); err != nil {
			return Config{}, err
		}
	}
	if err := chat.Fetch(ctx, s.Repo, "generation_config.json", filepath.Join(s.Dir, "generation_config.json"), log); err != nil && !chat.NotFound(err) {
		return Config{}, err
	}
	config := filepath.Join(s.Dir, "config.json")
	codecConfig := filepath.Join(s.Dir, "audio_tokenizer", "config.json")
	a, tm, err := llm.LoadConfig(config, lfm3.Arch.Name)
	if err != nil {
		return Config{}, err
	}
	_, dm, err := llm.LoadConfig(config, lfm3gen.DepthArch.Name)
	if err != nil {
		return Config{}, err
	}
	checkpoint := sync.OnceValues(func() (string, error) {
		return filepath.Join(s.Dir, "model.safetensors"), chat.FetchWeights(ctx, s.Repo, s.Dir, log)
	})
	codecWeights := sync.OnceValues(func() (string, error) {
		p := filepath.Join(s.Dir, "audio_tokenizer", "model.safetensors")
		return p, chat.Fetch(ctx, s.Repo, "audio_tokenizer/model.safetensors", p, log)
	})
	cache := cmp.Or(s.Cache, DefaultCache)
	format := cmp.Or(s.Format, a.DefaultFormat(tm))
	c := Config{Dir: s.Dir}
	type prog struct {
		config  string
		weights func() (string, error)
		name    string
		p       chat.Program
	}
	text := func(seq int) prog {
		return prog{config, checkpoint, fmt.Sprintf("tts-%s%s-%s-c%d-q%d", a.Name, a.Rev(), format, cache, seq),
			chat.Program{Arch: lfm3.Arch.Name, Format: format, Seq: seq, Cache: cache, Params: map[string]string{"hidden-out": "1"}}}
	}
	depth, codec := lfm3gen.DepthArch, lfm3gen.CodecArch
	progs := []prog{
		text(1), text(64),
		{config, checkpoint, "tts-" + depth.Name + depth.Rev() + "-fp16", chat.Program{Arch: depth.Name, Format: "fp16"}},
		{codecConfig, codecWeights, "tts-" + codec.Name + codec.Rev() + "-fp16-f64", chat.Program{Arch: codec.Name, Format: "fp16", Params: map[string]string{"codec-frames": "64"}}},
	}
	var paths []string
	for _, p := range progs {
		out := filepath.Join(s.Dir, p.name+".mlpackage")
		weights := ""
		if !chat.Converted(out) {
			if weights, err = p.weights(); err != nil {
				return Config{}, err
			}
		}
		path, err := chat.ConvertProgram(ctx, p.config, weights, out, p.p, log)
		if err != nil {
			return Config{}, err
		}
		paths = append(paths, path)
	}
	c.Text, c.Depth, c.Codec = paths[:2], paths[2], paths[3]
	d := dm.(*lfm3gen.Depth)
	names := append(llm.HostTables(tm), d.Feedback(), d.MaskEmbedding())
	for cb := range d.Codebooks {
		names = append(names, d.Table(cb))
	}
	slices.Sort(names)
	names = slices.Compact(names)
	if c.Weights, err = chat.Tables(filepath.Join(s.Dir, TablesFile), lfm3.Arch, names, checkpoint, log); err != nil {
		return Config{}, err
	}
	// a conversion reads the checkpoint into memory: give it back now
	debug.FreeOSMemory()
	return c, nil
}
