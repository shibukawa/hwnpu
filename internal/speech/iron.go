package speech

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"

	"github.com/shibukawa/bonsaiai"
	"github.com/shibukawa/bonsaiai/internal/llm/iron"
	"github.com/shibukawa/bonsaiai/internal/vecops"
)

// HiddenBuffer is the state the export of tools/amd/iron_lfm3tts.py writes the text model's final
// hidden state to (after the final norm, before the head), in a prompt chunk and a decode step.
const HiddenBuffer = "bonsaiai_hidden"

// ironText is the text model on AMD's NPU: an export of tools/amd/iron_lfm3tts.py (iron.Model),
// its token embeddings the host's, so that a decode step reads the row the host writes
// (iron.Model.StepRow: a frame's feedback).
type ironText struct {
	m      *iron.Model
	ctx    context.Context
	hidden []float32
	at     int
	row    []byte
}

func openIronText(ctx context.Context, dev *bonsaiai.Device, dir string, hidden int, logf func(string, ...any)) (*ironText, error) {
	m, err := iron.Open(ctx, dev, dir, logf)
	if err != nil {
		return nil, fmt.Errorf("speech: text model: %w", err)
	}
	if _, ok := m.Buffers("decode")[HiddenBuffer]; !ok {
		m.Close()
		return nil, fmt.Errorf("speech: %s does not write its hidden state (%s): export it with iron_export_lm.py lfm3tts", dir, HiddenBuffer)
	}
	return &ironText{m: m, ctx: ctx, hidden: make([]float32, hidden), row: make([]byte, 2*hidden)}, nil
}

func (t *ironText) Reset()            { t.at = 0 }
func (t *ironText) Q() int            { return 1 }
func (t *ironText) Pos() int          { return t.at }
func (t *ironText) Cache() int        { return t.m.Manifest.Config.MaxSeqLen }
func (t *ironText) Hidden() []float32 { return t.hidden }
func (t *ironText) Close() error      { return t.m.Close() }

// ReadPrompt reads ids from position 0 in prompt chunks.
func (t *ironText) ReadPrompt(ctx context.Context, ids []int32) error {
	t.ctx = ctx
	if _, err := t.m.Prompt(ctx, ids, 0); err != nil {
		return fmt.Errorf("speech: text model: %w", err)
	}
	t.at = len(ids)
	return t.readHidden()
}

// Step reads each row (an embedding) in a decode step.
func (t *ironText) Step(rows [][]float32) ([]float32, error) {
	var logits []float32
	for _, r := range rows {
		if len(r) != len(t.hidden) {
			return nil, fmt.Errorf("speech: a row of %d for the text model's %d", len(r), len(t.hidden))
		}
		putBF16(t.row, r)
		var err error
		if logits, err = t.m.StepRow(t.ctx, t.row, t.at); err != nil {
			return nil, fmt.Errorf("speech: text model: %w", err)
		}
		t.at++
	}
	return logits, t.readHidden()
}

func (t *ironText) readHidden() error {
	b, _, err := t.m.Buffer("decode", HiddenBuffer)
	if err != nil {
		return err
	}
	if len(b) < 2*len(t.hidden) {
		return fmt.Errorf("speech: %s holds %d bytes, not a row of %d", HiddenBuffer, len(b), len(t.hidden))
	}
	vecops.BF16ToF32(t.hidden, b[:2*len(t.hidden)])
	return nil
}

// putBF16 writes xs as bf16 (to the nearest even) into b.
func putBF16(b []byte, xs []float32) {
	for i, x := range xs {
		bits := math.Float32bits(x)
		var h uint16
		if bits&0x7fffffff > 0x7f800000 {
			h = uint16(bits>>16) | 0x40
		} else {
			h = uint16((bits + 0x7fff + (bits>>16)&1) >> 16)
		}
		binary.LittleEndian.PutUint16(b[2*i:], h)
	}
}
