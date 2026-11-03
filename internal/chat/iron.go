package chat

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

// ironSession is the sequence of a model IRON compiled for AMD's NPU (internal/llm/iron). The
// model embeds the tokens itself and its caches are on the device, written by position: going
// back to a mark is setting the position (the export refuses models whose state cannot restart
// anywhere, iron.Manifest.Restartable). A copy of the whole cache (Save) is not kept.
type ironSession struct {
	m   *iron.Model
	pos int
}

var _ session = (*ironSession)(nil)

func (s *ironSession) Q() int    { return s.m.Manifest.Config.PrefillChunk }
func (s *ironSession) Qs() []int { return []int{1, s.Q()} } // no drafts: the decode step reads 1
func (s *ironSession) Cache() int {
	return s.m.Manifest.Config.MaxSeqLen
}
func (s *ironSession) Pos() int              { return s.pos }
func (s *ironSession) Reset()                { s.pos = 0 }
func (s *ironSession) Mark() *decode.Mark    { return decode.NewMark(s.pos, nil) }
func (s *ironSession) Save() *decode.Mark    { return nil }
func (s *ironSession) Rewind(m *decode.Mark) { s.pos = m.Pos() }
func (s *ironSession) Migrate(decode.Program) error {
	return errors.New("chat: an IRON model has one cache")
}
func (s *ironSession) Add(decode.Program) error {
	return errors.New("chat: an IRON model has one cache")
}

func (s *ironSession) Resume([]byte, *decode.Mark) error {
	return errors.New("chat: an IRON model keeps no prompts on disk")
}

func (s *ironSession) Step([][]float32) ([]float32, error) {
	return nil, errors.New("chat: an IRON model reads token ids, not embeddings")
}

func (s *ironSession) Speculate([][]float32, func(func(int) []float32) int) ([]float32, int, error) {
	return nil, 0, errors.New("chat: an IRON model checks no drafts")
}

// StepTokens runs one token on the decode step, more on the prompt chunk.
func (s *ironSession) StepTokens(toks []int32) ([]float32, error) {
	if s.pos+len(toks) > s.Cache() {
		return nil, fmt.Errorf("%w: %d tokens past %d, the cache holds %d", ErrContext, len(toks), s.pos, s.Cache())
	}
	var logits []float32
	var err error
	if len(toks) == 1 {
		logits, err = s.m.Step(context.Background(), toks[0], s.pos)
	} else {
		logits, err = s.m.Prompt(context.Background(), toks, s.pos)
	}
	if err != nil {
		return nil, err
	}
	s.pos += len(toks)
	return logits, nil
}

// archOf is the chat format of an IRON model by its class.
func archOf(model string) (string, error) {
	switch model {
	case "Gemma4":
		return "gemma4", nil
	}
	return "", fmt.Errorf("chat: no chat format for the IRON model %s", model)
}

// loadIRON loads a model IRON compiled for AMD's NPU (Config.IRON).
func loadIRON(ctx context.Context, c Config) (*Engine, error) {
	logf := func(format string, args ...any) {
		if c.Log != nil {
			fmt.Fprintf(c.Log, format+"\n", args...)
		}
	}
	name := c.Name
	if name == "" {
		name = strings.TrimSuffix(filepath.Base(c.IRON), ".amd-npu2")
	}
	logf("loading %s onto AMD's NPU", name)
	dev, err := hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD))
	if err != nil {
		return nil, err
	}
	m, err := iron.Open(ctx, dev, c.IRON, func(f string, a ...any) { logf("  "+f, a...) })
	if err != nil {
		dev.Close()
		return nil, err
	}
	e := &Engine{Stops: map[int32]bool{}, emb: map[int32][]float32{}, nextID: -1, log: c.Log, dev: dev, ir: m}
	e.Name = name
	fail := func(err error) (*Engine, error) {
		e.Close()
		return nil, err
	}
	arch, err := archOf(m.Manifest.Model)
	if err != nil {
		return fail(err)
	}
	if e.Template, err = TemplateFor(arch); err != nil {
		return fail(err)
	}
	tok := c.Tokenizer
	if tok == "" {
		tok = filepath.Join(c.IRON, "tokenizer.json")
	}
	if e.Tok, err = tokenizer.Load(tok); err != nil {
		return fail(err)
	}
	for _, t := range e.Template.Stops() {
		if id, ok := e.Tok.ID(t); ok {
			e.Stops[id] = true
		}
	}
	e.sess = &ironSession{m: m}
	logf("ready: context %d tokens, up to %d tokens per call", e.Context(), e.sess.Q())
	return e, nil
}
