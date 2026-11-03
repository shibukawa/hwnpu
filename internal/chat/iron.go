package chat

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"slices"
	"strings"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

// ironSession is the sequence of a model IRON compiled for AMD's NPU (internal/llm/iron). The
// model embeds the tokens itself and its state is its KV caches, written by position: going
// back to a mark is setting the position, and the rows of the positions (iron.Model.ExportRows)
// are the whole sequence, which Save copies and the prompt store keeps. Models whose state
// cannot restart anywhere (rings, recurrent layers) are not exported.
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
func (s *ironSession) Pos() int           { return s.pos }
func (s *ironSession) Reset()             { s.pos = 0 }
func (s *ironSession) Mark() *decode.Mark { return decode.NewMark(s.pos, nil) }

// savedRows is the key of a saved sequence's rows in its mark's buffers.
const savedRows = "iron-rows"

// Save copies the rows of the whole sequence (18 KiB a token for Gemma 4 E2B).
func (s *ironSession) Save() *decode.Mark {
	rows, err := s.m.ExportRows(0, s.pos)
	if err != nil {
		return nil
	}
	return decode.NewMark(s.pos, map[string][]byte{savedRows: rows})
}

// Rewind goes back to a mark of this sequence, or brings back a saved one.
func (s *ironSession) Rewind(m *decode.Mark) {
	if rows, ok := m.Bufs()[savedRows]; ok {
		if err := s.m.ImportRows(0, rows); err != nil {
			s.pos = 0 // the next prompt is read from the start
			return
		}
	}
	s.pos = m.Pos()
}

func (s *ironSession) RowBytes() int { return s.m.RowBytes() }

func (s *ironSession) ExportRows(from, to int) ([]byte, error) {
	if to > s.pos {
		return nil, fmt.Errorf("chat: rows to %d of a sequence of %d", to, s.pos)
	}
	return s.m.ExportRows(from, to)
}

var _ rowSource = (*ironSession)(nil)

func (s *ironSession) Migrate(decode.Program) error {
	return errors.New("chat: an IRON model has one cache")
}
func (s *ironSession) Add(decode.Program) error {
	return errors.New("chat: an IRON model has one cache")
}

// Resume brings an empty sequence to a point of the prompt store: the rows of the positions
// before the mark.
func (s *ironSession) Resume(rows []byte, m *decode.Mark) error {
	rb := s.m.RowBytes()
	if rb == 0 || len(rows) != m.Pos()*rb {
		return fmt.Errorf("chat: %d bytes of rows for a mark at %d (%d a row)", len(rows), m.Pos(), rb)
	}
	if err := s.m.ImportRows(0, rows); err != nil {
		return err
	}
	s.pos = m.Pos()
	return nil
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
	if c.PromptDir != "" {
		var sizes []int64
		for _, b := range m.Manifest.Banks {
			sizes = append(sizes, int64(b.Bytes))
		}
		slices.Sort(sizes)
		dir := filepath.Join(c.PromptDir, storeKey(name, []string{filepath.Join(c.IRON, "manifest.json")}, sizes))
		if e.store, err = openStore(dir); err != nil {
			logf("prompt store: %v (prompts will not be kept)", err)
			e.store = nil
		} else {
			logf("prompt store: %s, %d prompts kept", dir, len(e.store.entries))
		}
	}
	logf("ready: context %d tokens, up to %d tokens per call", e.Context(), e.sess.Q())
	return e, nil
}
