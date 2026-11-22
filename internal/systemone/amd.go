package systemone

import (
	"context"
	"fmt"
	"io"
	"path/filepath"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

// A d1 model on AMD's NPU is an export of tools/amd/iron_export_lm.py d1 (d1omni for d1-Omni,
// tools/amd/iron_d1.py): LFM3-VL's (LFM3-Omni's) export whose prompt chunk reads the whole
// prompt from position 0, the full attention layers both ways, and gives the final hidden state
// of its last row; its towers and placed chunk are LFM3-VL's (LFM3-Omni's). A question is one
// call of the chunk (so a prompt of at most a chunk); the logits of its labels are that hidden
// state times the rows of the token embeddings (the head is tied to them), which the export keeps
// on the host (embed.bin). Each tower holds one run's soft tokens: a request has at most one
// image and one clip (a spoken question's included), which run once for all its questions.

// AMDConfig says where a d1 export for AMD's NPU is.
type AMDConfig struct {
	// Dir is the export: its manifest, and the checkpoint's config.json, tokenizer.json and
	// calibration.json, which the exporter copies there.
	Dir string
	// Name is the model's name in the API (default: the export's).
	Name string
	// Log receives what loading does (nil for none).
	Log io.Writer
}

// LoadAMD loads a d1 export for AMD's NPU.
func LoadAMD(ctx context.Context, c AMDConfig) (*Model, error) {
	logf := func(format string, args ...any) {
		if c.Log != nil {
			fmt.Fprintf(c.Log, format+"\n", args...)
		}
	}
	_, model, err := llm.LoadConfig(filepath.Join(c.Dir, "config.json"), "")
	if err != nil {
		return nil, err
	}
	cfg, ok := model.(*lfm3.Config)
	if !ok || !cfg.D1 {
		return nil, fmt.Errorf("systemone: %s is not a d1 model (LFM3 with architectures D1Model)", c.Dir)
	}
	m := &Model{Name: c.Name, cfg: cfg, emb: map[int32][]float32{}, heads: map[int32][]float32{}}
	if m.Name == "" {
		m.Name = strings.TrimSuffix(filepath.Base(c.Dir), filepath.Ext(c.Dir))
	}
	if m.temperature, err = LoadCalibration(filepath.Join(c.Dir, "calibration.json")); err != nil {
		return nil, err
	}
	tok, err := tokenizer.Load(filepath.Join(c.Dir, "tokenizer.json"))
	if err != nil {
		return nil, err
	}
	m.tok, m.escape = tok, Escaper(tok.Control())
	t0 := time.Now()
	if err := llm.LockNPU(ctx); err != nil { // one process's language model on the NPU at a time
		return nil, err
	}
	if m.dev, err = hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD)); err != nil {
		return nil, err
	}
	iron.ShareExport(c.Dir, func(f string, a ...any) { logf("  "+f, a...) })
	if m.amd, err = iron.Open(ctx, m.dev, c.Dir, func(f string, a ...any) { logf("  "+f, a...) }); err != nil {
		m.Close()
		return nil, err
	}
	mc := m.amd.Manifest.Config
	if mc.HiddenSize != cfg.HiddenSize {
		m.Close()
		return nil, fmt.Errorf("systemone: %s: an export of hidden size %d for a model of %d", c.Dir, mc.HiddenSize, cfg.HiddenSize)
	}
	what := "text"
	if m.amd.HasVision() {
		what += ", images"
	}
	if m.amd.HasAudio() {
		what += ", audio"
	}
	logf("%s: %d tokens per question (%s), on AMD's NPU, loaded in %.1fs", filepath.Base(c.Dir), mc.PrefillChunk, what, time.Since(t0).Seconds())
	return m, nil
}

// decideAMD is Decide on AMD's NPU: the towers run once for the request, a call of the prompt
// chunk a question.
func (m *Model) decideAMD(ctx context.Context, r *Request) (*Response, error) {
	mm := m.amd.Manifest.Multimodal
	// what each of the request's media is in the prompt: its placeholder and its soft tokens
	type placed struct {
		token int32
		count int
	}
	var media []placed
	images, clips := 0, 0
	for _, md := range r.Media {
		if md.Image != nil {
			images++
		} else {
			clips++
		}
	}
	spoken := false
	for _, q := range r.Questions {
		spoken = spoken || q.Spoken != nil
	}
	switch {
	case images > 1 || clips > 1:
		return nil, &Unsupported{"on AMD's NPU a request takes one image and one clip at most"}
	case spoken && clips > 0:
		return nil, &Unsupported{"on AMD's NPU a request takes one clip at most: a spoken question and a clip are two"}
	case images > 0 && !m.amd.HasVision():
		return nil, &Unsupported{"this model does not take images"}
	case (clips > 0 || spoken) && !m.amd.HasAudio():
		return nil, &Unsupported{"this server does not take audio: d1-Omni does"}
	}
	for _, md := range r.Media {
		var n int
		var err error
		token := mm.ImageToken
		if md.Image != nil {
			n, err = m.amd.RunVision(ctx, md.Image)
		} else {
			n, err = m.amd.RunAudio(ctx, md.Audio)
			token = mm.AudioToken
		}
		if err != nil {
			return nil, err
		}
		media = append(media, placed{token, n})
	}
	res := &Response{Model: m.Name}
	for _, q := range r.Questions {
		all := media
		if q.Spoken != nil {
			n, err := m.amd.RunAudio(ctx, q.Spoken)
			if err != nil {
				return nil, err
			}
			all = append(append([]placed(nil), media...), placed{mm.AudioToken, n})
		}
		var counts []int
		for _, p := range all {
			counts = append(counts, p.count)
		}
		ids, groups, err := m.Tokens(r.State, q, counts)
		if err != nil {
			return nil, err
		}
		a, err := m.answerAMD(ctx, q, ids, groups, func(i int) int32 {
			// the i-th soft token of the prompt is of the media whose run holds it
			for _, p := range all {
				if i < p.count {
					return p.token
				}
				i -= p.count
			}
			return mm.ImageToken
		}, len(all) > 0)
		if err != nil {
			return nil, err
		}
		res.IDs = append(res.IDs, q.ID)
		res.Answers = append(res.Answers, a)
		res.InputTokens += len(ids)
	}
	return res, nil
}

// answerAMD answers one question of the prompt ids (the media's places imageToken, each the
// placeholder placeholder(k) of the k-th of them gives): one call of the prompt chunk from 0.
func (m *Model) answerAMD(ctx context.Context, q Question, ids []int32, groups [][]int32, placeholder func(int) int32, withMedia bool) (Answer, error) {
	n, chunk := len(ids), m.amd.Manifest.Config.PrefillChunk
	if n > chunk {
		return Answer{}, bad("questions.%s: the prompt is %d tokens, the model takes %d on AMD's NPU (export it with a larger --prefill-chunk)", q.ID, n, chunk)
	}
	k := 0
	for i, t := range ids {
		if t == imageToken {
			ids[i] = placeholder(k)
			k++
		}
	}
	var h []float32
	var err error
	if withMedia {
		// each placeholder's soft tokens before the call: none, a call from position 0
		h, err = m.amd.PromptPlaced(ctx, ids, 0, make([]int32, len(m.amd.Manifest.Multimodal.Placeholders)))
	} else {
		h, err = m.amd.Prompt(ctx, ids, 0)
	}
	if err != nil {
		return Answer{}, err
	}
	if len(h) != m.cfg.HiddenSize {
		return Answer{}, fmt.Errorf("systemone: the export gave %d values, not the hidden state of %d (an export of tools/amd/iron_export_lm.py d1)", len(h), m.cfg.HiddenSize)
	}
	logits := make([]float64, len(groups))
	for gi, g := range groups {
		best := -1e300
		for _, t := range g {
			row, ok := m.heads[t]
			if !ok {
				if row, err = m.amd.EmbedRow(t); err != nil {
					return Answer{}, err
				}
				m.heads[t] = row
			}
			var dot float64
			for j, v := range row {
				dot += float64(v) * float64(h[j])
			}
			best = max(best, dot)
		}
		logits[gi] = best
	}
	return Decide(q, logits, m.temperature), nil
}
