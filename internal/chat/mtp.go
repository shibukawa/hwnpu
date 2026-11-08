package chat

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/chain"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
)

// mtpDrafter guesses drafts with a checkpoint's multi-token prediction layer (Config.MTP,
// internal/llm/qwen35's MTP): at position t it reads the model's final hidden state h_t and the
// embedding of token t+1 and guesses token t+2. Its decoder layer attends over an entry for each
// position it has read, in a KV cache of its own, so it reads every position the model does:
// after each call of the model, the positions whose next token is now known, and the last one with
// the token just picked, whose logits give the first draft. It guesses further from its own final
// hidden state in place of the model's, one token per call, and then goes back to the positions
// read for real (decode.Session.Truncate: the cache is its only state).
//
// Its positions start at base: the layer's attention, with RoPE, sees relative positions only,
// so it can start anywhere the model's hidden states are known, without the sequence before.
type mtpDrafter struct {
	e      *Engine
	chains []*chain.Chain
	sess   *decode.Session
	k      int     // the most drafts it guesses in a row
	ids    []int32 // the tokens of the logits, in their order (llm.Manifest.OutputIDs); nil: all

	// fed is the sequence the hidden states and the entries are of; hrows the model's final
	// hidden states at positions [hbase, hbase+len(hrows)) of it
	fed   []int32
	hbase int
	hrows [][]float32
	// base is the position of the session's first entry, and real the entries read for real
	// (the session's position after Truncate); -1: the session is empty and starts anywhere
	base, real int
	failed     bool // an error turned it off
}

// mtpKeep is how many hidden states before the entries the drafter keeps, to go back to a mark of
// the sequence without losing the entries before it: the next request's prompt goes back to the
// end of the last one's (engine.go's marks), which is behind the answer.
const mtpKeep = 4096

// mtpStart is the most positions the layer reads before its first draft when it starts anew or
// is that far behind (a long prompt): the model's hidden state at each holds what came before
// it, and the layer's own attention over the positions before these is not worth a call per
// mtpSeq of them.
const mtpStart = 512

// mtpVocab is how many of the most frequent tokens the layer guesses among (qwen35-mtp's
// draft-vocab, HWNPU_MTP_VOCAB at conversion; 0 for the whole vocabulary): its output layer is
// most of its weights (Qwen3.5-2B: 0.36 of 0.40 GiB in lut6, of 248320 tokens), read at every
// draft. The 32768 most frequent cover 98.3% of the tokens of Japanese and English text and code.
var mtpVocab = 32768

// mtpDrafts is the most drafts in a row the layer guesses (HWNPU_MTP_DRAFTS overrides it): each
// after the first costs a call of the layer and is right less often.
var mtpDrafts = 2

// loadMTP opens the program of the multi-token prediction layer (Config.MTP).
func loadMTP(ctx context.Context, e *Engine, c Config) (*mtpDrafter, error) {
	ds, ok := e.sess.(*decode.Session)
	if !ok {
		return nil, fmt.Errorf("chat: drafts from the multi-token prediction layer need a model on the NPU's programs")
	}
	a, _, err := llm.LoadConfig(c.ModelConfig, "")
	if err != nil {
		return nil, err
	}
	_, mm, err := llm.LoadConfig(c.ModelConfig, a.Name+"-mtp")
	if err != nil {
		return nil, err
	}
	cm, ok := mm.(llm.Cached)
	if !ok {
		return nil, fmt.Errorf("chat: %s cannot run with a KV cache", filepath.Base(c.MTP[0]))
	}
	m := &mtpDrafter{e: e, k: mtpDrafts, base: -1}
	if v, err := strconv.Atoi(os.Getenv("HWNPU_MTP_DRAFTS")); err == nil && v > 0 {
		m.k = v
	}
	m.k = min(m.k, ds.Qs()[0]-1)
	man, err := llm.ReadManifest(c.MTP[0])
	if err != nil {
		return nil, err
	}
	m.ids = man.OutputIDs
	for i, pkg := range c.MTP {
		var ch *chain.Chain
		if i == 0 {
			ch, err = chain.Open(ctx, e.dev, pkg)
		} else {
			ch, err = m.chains[0].Share(ctx, e.dev, pkg)
		}
		if err != nil {
			m.close()
			return nil, err
		}
		m.chains = append(m.chains, ch)
		if !ch.OnNPU() {
			e.logf("the multi-token prediction layer's program %s is NOT all on the NPU", filepath.Base(pkg))
		}
	}
	if m.sess, err = decode.New(m.chains[0].Program(context.Background()), cm); err != nil {
		m.close()
		return nil, fmt.Errorf("chat: %s: %w", filepath.Base(c.MTP[0]), err)
	}
	for _, ch := range m.chains[1:] {
		if err := m.sess.Add(ch.Program(context.Background())); err != nil {
			m.close()
			return nil, err
		}
	}
	return m, nil
}

func (m *mtpDrafter) close() {
	if m.sess != nil {
		m.sess.Close()
		m.sess = nil
	}
	for i := len(m.chains) - 1; i >= 0; i-- {
		m.chains[i].Close()
	}
	m.chains = nil
}

// sync keeps what is of seq's beginning: the hidden states and the entries of positions whose
// tokens up to them are still those of seq.
func (m *mtpDrafter) sync(seq []int32) {
	n := 0
	for n < len(m.fed) && n < len(seq) && m.fed[n] == seq[n] {
		n++
	}
	m.fed = m.fed[:n]
	if n < m.hbase {
		m.hbase, m.hrows = n, nil
	} else if n < m.hbase+len(m.hrows) {
		m.hrows = m.hrows[:n-m.hbase]
	}
	// entry t read h_t and token t+1
	if m.base >= 0 && m.base+m.real > n-1 {
		m.real = max(n-1-m.base, 0)
		if m.real == 0 {
			m.base = -1
		}
	}
}

// record takes the model's final hidden states of the tokens fed[pos0:], which a call just kept;
// rows nil (a program without them) leaves a gap the entries cannot cross.
func (m *mtpDrafter) record(fed []int32, pos0 int, rows [][]float32) {
	if m.failed {
		return
	}
	m.sync(fed[:pos0])
	m.fed = append(m.fed, fed[pos0:]...)
	if len(rows) != len(fed)-pos0 || pos0 != m.hbase+len(m.hrows) {
		m.hbase, m.hrows = len(fed), nil
		if len(rows) == len(fed)-pos0 {
			m.hbase, m.hrows = pos0, rows
		}
		return
	}
	m.hrows = append(m.hrows, rows...)
	// drop what is far behind the entries
	keep := m.hbase + len(m.hrows) - mtpKeep
	if m.base >= 0 {
		keep = min(keep, m.base+m.real)
	}
	if d := keep - m.hbase; d > 0 {
		m.hrows = slices.Delete(m.hrows, 0, d)
		m.hbase = keep
	}
}

// draft guesses up to k tokens after seq (the tokens fed and the one just picked), with the
// prompt's guess when that agrees with the layer's first and goes further (LookupDraft).
func (m *mtpDrafter) draft(seq []int32, k int) []int32 {
	look := LookupDraft(seq, k)
	if m.failed {
		return look
	}
	d, err := m.guess(seq, min(k, m.k))
	if err != nil {
		m.failed = true
		m.e.logf("the multi-token prediction layer is off: %v", err)
		return look
	}
	if len(look) > len(d) && (len(d) == 0 || look[0] == d[0]) {
		return look
	}
	return d
}

// guess reads the positions up to seq's last token and guesses k tokens after it.
func (m *mtpDrafter) guess(seq []int32, k int) ([]int32, error) {
	if k <= 0 {
		return nil, nil
	}
	last := len(seq) - 1 // the token picked, not fed yet
	m.sync(seq[:last])
	// entries go up to position last-1, whose token after it is seq[last]
	end := last // the entries to have: [base, end)
	if m.hbase+len(m.hrows) < end || end <= m.hbase {
		return nil, nil // the model's hidden state at last-1 is not known
	}
	if m.base < 0 || m.base+m.real < m.hbase || end-m.base > m.sess.Cache() || end-(m.base+m.real) > mtpStart {
		// start anew from the hidden states known, mtpStart of them at most: a long prompt
		// would take a call per mtpSeq tokens before the first draft
		m.base = max(m.hbase, end-min(mtpStart, m.sess.Cache()/2))
		m.real = 0
		m.sess.Reset()
	}
	if err := m.sess.Truncate(m.real); err != nil {
		return nil, err
	}
	var logits, h []float32
	for t := m.base + m.real; t < end; {
		n := min(m.sess.Q(), end-t)
		rows := make([][]float32, n)
		for i := range rows {
			x, err := m.e.rows(seq[t+i+1 : t+i+2])
			if err != nil {
				return nil, err
			}
			rows[i] = append(append(make([]float32, 0, 2*len(m.hrows[0])), x[0]...), m.hrows[t+i-m.hbase]...)
		}
		var err error
		if logits, err = m.sess.Step(rows); err != nil {
			m.sess.Reset()
			m.base = -1
			return nil, err
		}
		h = m.sess.Hidden()
		t += n
		m.real += n
	}
	if m.ids != nil && len(logits) != len(m.ids) {
		return nil, fmt.Errorf("chat: %d logits of the multi-token prediction layer for %d tokens", len(logits), len(m.ids))
	}
	drafts := []int32{m.token(argmax(logits))}
	for len(drafts) < k {
		x, err := m.e.rows(drafts[len(drafts)-1:])
		if err != nil {
			return nil, err
		}
		if h == nil {
			break
		}
		if logits, err = m.sess.Step([][]float32{append(append([]float32{}, x[0]...), h...)}); err != nil {
			return nil, err
		}
		h = m.sess.Hidden()
		drafts = append(drafts, m.token(argmax(logits)))
	}
	// back to the entries read for real
	return drafts, m.sess.Truncate(m.real)
}

// token is the token of the i-th logit.
func (m *mtpDrafter) token(i int32) int32 {
	if m.ids != nil {
		return m.ids[i]
	}
	return i
}

func argmax(v []float32) int32 {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return int32(best)
}
