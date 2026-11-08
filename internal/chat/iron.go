package chat

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

// ironSession is the sequence of a model IRON compiled for AMD's NPU (internal/llm/iron). The
// model embeds the tokens itself and its state is its KV caches, written by position, and for a
// model with recurrent layers (LFM3's short convolutions) their states, a slot per prompt-chunk
// boundary: a mark holds the slot its position reads (iron.Model.StateAt), so going back to it
// is setting the position and writing that slot back. The rows of the positions
// (iron.Model.ExportRows) and a mark's slot are the whole sequence, which Save copies and the
// prompt store keeps. Models with ring caches are not loaded.
type ironSession struct {
	m   *iron.Model
	pos int
	// media finds the image or clip of a negative id (Engine.media); placed is the first id of
	// the media whose soft tokens each tower last left on the device, by placeholder token.
	media  func(t int32) (*encoded, int, bool)
	placed map[int32]int32
	// step is the last decode step's time and chunk the shortest prompt chunk's: a few tokens
	// run as decode steps when they cost less than a chunk (Qwen3.6's chunk takes 2.2 s however
	// short, a step 0.17 s), once both have been seen.
	step, chunk time.Duration
}

var _ session = (*ironSession)(nil)

func (s *ironSession) Q() int    { return s.m.Manifest.Config.PrefillChunk }
func (s *ironSession) Qs() []int { return []int{1, s.Q()} } // no drafts: the decode step reads 1
func (s *ironSession) Cache() int {
	return s.m.Manifest.Config.MaxSeqLen
}
func (s *ironSession) Pos() int { return s.pos }
func (s *ironSession) Reset()   { s.pos = 0 }

// savedRows is the key of a saved sequence's rows in its mark's buffers, savedState that of the
// recurrent states' slot at its position.
const (
	savedRows  = "iron-rows"
	savedState = "iron-state"
)

// Mark is the position, with the recurrent states' slot there for a model that has them.
func (s *ironSession) Mark() *decode.Mark {
	if len(s.m.Manifest.Checkpoints) == 0 {
		return decode.NewMark(s.pos, nil)
	}
	state, err := s.m.StateAt(s.pos)
	if err != nil {
		return decode.NewMark(0, nil) // a mark at the start, which needs no state
	}
	return decode.NewMark(s.pos, map[string][]byte{savedState: state})
}

// Save copies the rows of the whole sequence (18 KiB a token for Gemma 4 E2B) and its mark.
func (s *ironSession) Save() *decode.Mark {
	rows, err := s.m.ExportRows(0, s.pos)
	if err != nil {
		return nil
	}
	m := s.Mark()
	bufs := map[string][]byte{savedRows: rows}
	if st, ok := m.Bufs()[savedState]; ok {
		bufs[savedState] = st
	}
	return decode.NewMark(m.Pos(), bufs)
}

// Rewind goes back to a mark of this sequence, or brings back a saved one.
func (s *ironSession) Rewind(m *decode.Mark) {
	if err := s.restore(m); err != nil {
		s.pos = 0 // the next prompt is read from the start
		return
	}
	s.pos = m.Pos()
}

// restore writes back what a mark holds: the rows of a saved sequence, the recurrent states'
// slot of a model with them. A mark of such a model past the start without its slot is refused.
func (s *ironSession) restore(m *decode.Mark) error {
	if rows, ok := m.Bufs()[savedRows]; ok {
		if err := s.m.ImportRows(0, rows); err != nil {
			return err
		}
	}
	if len(s.m.Manifest.Checkpoints) == 0 || m.Pos() == 0 {
		return nil
	}
	state, ok := m.Bufs()[savedState]
	if !ok {
		return fmt.Errorf("chat: a mark at %d without the recurrent states", m.Pos())
	}
	return s.m.SetStateAt(m.Pos(), state)
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
	if err := s.restore(m); err != nil {
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

// StepTokens runs one token on the decode step, more on the prompt chunk; a piece holding an
// image's or a clip's soft tokens (negative ids) runs on the placed prompt chunk.
func (s *ironSession) StepTokens(toks []int32) ([]float32, error) {
	if s.pos+len(toks) > s.Cache() {
		return nil, fmt.Errorf("%w: %d tokens past %d, the cache holds %d", ErrContext, len(toks), s.pos, s.Cache())
	}
	if slices.ContainsFunc(toks, func(t int32) bool { return t < 0 }) {
		return s.stepMedia(toks)
	}
	var logits []float32
	var err error
	t0 := time.Now()
	switch {
	case len(toks) == 1:
		logits, err = s.m.Step(context.Background(), toks[0], s.pos)
		s.step = time.Since(t0)
	case s.step > 0 && s.chunk > 0 && time.Duration(len(toks))*s.step < s.chunk:
		for i, t := range toks {
			if logits, err = s.m.Step(context.Background(), t, s.pos+i); err != nil {
				break
			}
		}
	default:
		logits, err = s.m.Prompt(context.Background(), toks, s.pos)
		if d := time.Since(t0); s.chunk == 0 || d < s.chunk {
			s.chunk = d
		}
	}
	if err != nil {
		return nil, err
	}
	s.pos += len(toks)
	return logits, nil
}

// stepMedia runs tokens holding soft tokens, in pieces of at most one image and one clip: each
// tower leaves one media's soft tokens on the device at a time, and a placed chunk reads the
// run of each from before, its place in the media's run (IRON's Merge).
func (s *ironSession) stepMedia(toks []int32) ([]float32, error) {
	mm := s.m.Manifest.Multimodal
	if mm == nil || s.media == nil {
		return nil, fmt.Errorf("%w: this model reads no images", ErrImage)
	}
	var logits []float32
	for len(toks) > 0 {
		// the piece: up to a second image or a second clip
		var cur [2]*encoded // image, clip
		var from [2]int
		n := 0
		for n < len(toks) {
			t := toks[n]
			if t < 0 {
				im, r, ok := s.media(t)
				if !ok {
					return nil, fmt.Errorf("%w: an image of the prompt is no longer kept; send the request again", ErrImage)
				}
				k := 0
				if im.audio {
					k = 1
				}
				if cur[k] == nil {
					cur[k], from[k] = im, r
				} else if cur[k] != im {
					break
				}
			}
			n++
		}
		ids := make([]int32, n)
		for i, t := range toks[:n] {
			ids[i] = t
			if t < 0 {
				if im, _, _ := s.media(t); im.audio {
					ids[i] = mm.AudioToken
				} else {
					ids[i] = mm.ImageToken
				}
			}
		}
		for k, im := range cur {
			if im == nil {
				continue
			}
			token := mm.ImageToken
			if im.audio {
				token = mm.AudioToken
			}
			if s.placed[token] == im.first {
				continue
			}
			var err error
			if im.audio {
				_, err = s.m.RunAudio(context.Background(), im.clip)
			} else {
				_, err = s.m.RunVision(context.Background(), im.img)
			}
			if err != nil {
				return nil, err
			}
			s.placed[token] = im.first
			_ = k
		}
		before := make([]int32, len(mm.Placeholders))
		for i, p := range mm.Placeholders {
			switch {
			case p == mm.ImageToken && cur[0] != nil:
				before[i] = int32(from[0])
			case p == mm.AudioToken && cur[1] != nil:
				before[i] = int32(from[1])
			}
		}
		var err error
		if logits, err = s.m.PromptPlaced(context.Background(), ids, s.pos, before); err != nil {
			return nil, err
		}
		s.pos += n
		toks = toks[n:]
	}
	return logits, nil
}

// archOf is the chat format of an IRON model by its class.
func archOf(model string) (string, error) {
	switch model {
	case "Gemma4":
		return "gemma4", nil
	case "Lfm3", "Lfm3Vl", "Lfm3Audio", "Lfm3Omni":
		return "lfm3", nil
	case "Plamo3":
		return "plamo3", nil
	case "Qwen35", "Qwen35Moe":
		return "qwen35", nil
	}
	return "", fmt.Errorf("chat: no chat format for the IRON model %s", model)
}

// qwen36Template reports whether an export's chat_template.jinja is Qwen3.6's (it has
// preserve_thinking), which the model's class does not tell from Qwen3.5's.
func qwen36Template(dir string) bool {
	b, err := os.ReadFile(filepath.Join(dir, "chat_template.jinja"))
	return err == nil && bytes.Contains(b, []byte("preserve_thinking"))
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
	iron.ShareExport(c.IRON, func(f string, a ...any) { logf("  "+f, a...) })
	m, err := iron.Open(ctx, dev, c.IRON, func(f string, a ...any) { logf("  "+f, a...) })
	if err != nil {
		dev.Close()
		return nil, err
	}
	if !m.Resumable() {
		m.Close()
		dev.Close()
		return nil, fmt.Errorf("chat: %s has ring caches (or is an export from before recurrent states were listed), which the chat engine cannot go back in; export it again, with a prompt chunk that keeps every cache whole (a chunk of 1024 or more at 4096 tokens)", c.IRON)
	}
	e := &Engine{Stops: map[int32]bool{}, emb: map[int32][]float32{}, nextID: -1, log: c.Log, dev: dev, ir: m,
		idle: c.Idle, devName: llm.DeviceName(dev)}
	e.Name = name
	fail := func(err error) (*Engine, error) {
		e.Close()
		return nil, err
	}
	arch, err := archOf(m.Manifest.Model)
	if err != nil {
		return fail(err)
	}
	if arch == "qwen35" && qwen36Template(c.IRON) {
		arch = "qwen36"
	}
	if e.Template, err = TemplateFor(arch); err != nil {
		return fail(err)
	}
	tok := c.Tokenizer
	if tok == "" {
		tok = filepath.Join(c.IRON, "tokenizer.json")
		if _, err := os.Stat(tok); err != nil {
			jsonl := filepath.Join(c.IRON, "tokenizer.jsonl") // PLaMo's
			if _, err := os.Stat(jsonl); err == nil {
				tok = jsonl
			}
		}
	}
	if e.Tok, err = tokenizer.Load(tok); err != nil {
		return fail(err)
	}
	for _, t := range e.Template.Stops() {
		if id, ok := e.Tok.ID(t); ok {
			e.Stops[id] = true
		}
	}
	e.turnStart = -1
	for _, t := range []string{"<|im_start|>", "<start_of_turn>"} {
		if id, ok := e.Tok.ID(t); ok {
			e.turnStart = id
			break
		}
	}
	e.turnEnd = -1
	if ids := e.Tok.Encode("\n"); len(ids) == 1 {
		e.turnEnd = ids[0]
	}
	e.markTurns = len(m.Manifest.Checkpoints) > 0 && !m.Manifest.Restartable
	e.sess = &ironSession{m: m, media: e.media, placed: map[int32]int32{}}
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
	e.touch() // released after Config.Idle even when no request comes
	return e, nil
}

// irMultimodal is the multimodal settings of an IRON model with towers, or nil.
func (e *Engine) irMultimodal() *iron.Multimodal {
	if e.ir == nil || !(e.ir.HasVision() || e.ir.HasAudio()) {
		return nil
	}
	return e.ir.Manifest.Multimodal
}

// AMDExport is the IRON export hwnpu-chat and hwnpu-agent run when -iron is not given and the only
// NPU is AMD's, which runs no programs of hwnpu-convert: ~/models/NAME.amd-npu2-q4, else NAME.amd-npu2, for the
// model -use names. It is "" when a backend is named (backend or HWNPU_BACKEND), an Apple or
// Intel NPU is found, or no AMD one.
func AMDExport(ctx context.Context, use, backend string) (string, error) {
	if backend != "" || strings.TrimSpace(os.Getenv(llm.BackendEnv)) != "" {
		return "", nil
	}
	if _, _, err := llm.FindDevice(ctx, ""); err == nil {
		return "", nil
	}
	infos, _ := hwnpu.Enumerate(ctx)
	amd := false
	for _, info := range infos {
		amd = amd || info.Vendor == hwnpu.VendorAMD
	}
	if !amd {
		return "", nil
	}
	home, _ := os.UserHomeDir()
	root := filepath.Join(home, "models")
	name := strings.TrimSpace(use)
	if strings.Contains(name, ",") {
		return "", fmt.Errorf("-use %s: on AMD's NPU, one chat model runs at a time; -iron names its export (tools/amd/iron_export_lm.py)", use)
	}
	name = filepath.Base(name) // OWNER/NAME
	for _, suffix := range []string{".amd-npu2-q4", ".amd-npu2"} {
		d := filepath.Join(root, name+suffix)
		if _, err := os.Stat(filepath.Join(d, "manifest.json")); err == nil {
			return d, nil
		}
	}
	found, _ := filepath.Glob(filepath.Join(root, "*.amd-npu2*"))
	var names []string
	for _, d := range found {
		if _, err := os.Stat(filepath.Join(d, "manifest.json")); err == nil {
			names = append(names, filepath.Base(d))
		}
	}
	msg := fmt.Sprintf("AMD's NPU runs exports of tools/amd/iron_export_lm.py, and %s has none of %s (%s.amd-npu2-q4 or %s.amd-npu2); -iron names one", root, name, name, name)
	if len(names) > 0 {
		msg += ": " + strings.Join(names, ", ")
	}
	return "", errors.New(msg)
}
