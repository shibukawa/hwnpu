package engine

import (
	"context"
	"fmt"
	"image"
	"io"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/ai/vision"
	"github.com/shibukawa/hwnpu/ai/vision/internal/bridge"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/imagedev"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	"github.com/shibukawa/hwnpu/internal/vision/siglip"
)

func init() {
	r := bridge.Roles["label"]
	r.Prepare = prepareLabel
	r.Load = loadLabel
}

func prepareLabel(e airole.PrepareEnv) (f airole.Files, err error) {
	m := e.Model
	dir := program(m, "export")
	if dir == "" {
		format := ""
		switch {
		case interpreter(m):
			format = llm.BackendCoreML // what the CPU interpreter runs
		case !e.NPU.AMD():
			format = e.NPU.Format
		}
		switch format {
		case llm.BackendCoreML, llm.BackendCoreAI:
			dir, err = siglip.Prepare(chat.WithModel(e.Ctx, m), siglip.Setup{Dir: chat.ModelsDir(), Repo: m.Source(), Name: e.Base.Name, Format: format, Log: e.Log})
		case "":
			dir, err = chat.AMDPictureExport(e.Ctx, e.Base, "tools/amd/iron_siglip2.py", 0, e.Log)
		default:
			err = fmt.Errorf("SigLIP 2 runs on AMD's NPU and Apple's Neural Engine so far (this machine's NPU takes %s programs)", format)
		}
		if err != nil {
			return f, err
		}
	}
	x, err := siglip.ReadExport(dir)
	if err != nil {
		return f, err
	}
	if !siglip.IsProgramFormat(x.Format) && e.NPU.AMD() {
		iron.ShareExport(dir, logf(e.Log))
	}
	f.Set("export", dir)
	return f, nil
}

func loadLabel(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
	dir := f.Path("export")
	x, err := siglip.ReadExport(dir)
	if err != nil {
		return nil, err
	}
	var dev *hwnpu.Device
	switch {
	case siglip.IsProgramFormat(x.Format) && interpreter(e.Def):
	case siglip.IsProgramFormat(x.Format):
		if dev, _, err = llm.OpenDevice(ctx, imagedev.ExportDevice(x.Format, e.NPU.Device), e.Cache, hwnpu.WithPlacement(hwnpu.PlacementStrict)); err != nil {
			return nil, err
		}
	default:
		if dev, err = hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD)); err != nil {
			return nil, err
		}
	}
	m, err := siglip.Open(ctx, dev, dir)
	if err != nil {
		if dev != nil {
			dev.Close()
		}
		return nil, err
	}
	m.Logf = logf(e.Log)
	return bridge.Wrap["label"](&labeler{m: m, dev: dev, name: e.Model, dir: dir}), nil
}

// labeler is a label model of the engine (vision.Labeler's).
type labeler struct {
	mu   sync.Mutex
	m    *siglip.Model
	dev  *hwnpu.Device
	name string
	dir  string
}

func (l *labeler) Model() string  { return l.name }
func (l *labeler) Export() string { return l.dir }
func (l *labeler) ImageSize() int { return l.m.Config.Image }
func (l *labeler) Dim() int       { return l.m.Dim() }

func (l *labeler) EmbedPicture(ctx context.Context, img image.Image) (*vision.PictureEmbedding, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	t0 := time.Now()
	e, err := l.m.Image(ctx, img)
	if err != nil {
		return nil, err
	}
	return &vision.PictureEmbedding{Embedding: e, Took: time.Since(t0), Input: l.m.Times[0], NPU: l.m.Times[1]}, nil
}

func (l *labeler) EmbedTexts(ctx context.Context, texts []string) (*vision.TextEmbeddings, error) {
	return l.texts(ctx, texts, "")
}

func (l *labeler) texts(ctx context.Context, texts []string, keep string) (*vision.TextEmbeddings, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	t0 := time.Now()
	l.m.Times = [3]time.Duration{}
	es, err := l.m.CachedTexts(ctx, texts, keep)
	if err != nil {
		return nil, err
	}
	out := &vision.TextEmbeddings{Took: time.Since(t0), NPU: l.m.Times[1]}
	for _, e := range es {
		out.Embeddings = append(out.Embeddings, e)
	}
	return out, nil
}

func (l *labeler) Candidates(ctx context.Context, labels []string, o vision.CandidateOptions) (*vision.Candidates, error) {
	tmpl := o.Template
	if tmpl == "" {
		tmpl = vision.DefaultTemplate
	}
	if !strings.Contains(tmpl, "{}") {
		return nil, fmt.Errorf("vision: the template %q has no {}", tmpl)
	}
	sentences := make([]string, len(labels))
	for i, s := range labels {
		sentences[i] = strings.ReplaceAll(tmpl, "{}", s)
	}
	keep := ""
	if !o.NoCache {
		keep = cache.Sub("siglip")
	}
	t, err := l.texts(ctx, sentences, keep)
	if err != nil {
		return nil, err
	}
	return &vision.Candidates{Labels: append([]string(nil), labels...), TextEmbeddings: *t}, nil
}

func (l *labeler) Rank(e vision.Embedding, c *vision.Candidates, o vision.RankOptions) []vision.Scored {
	texts := make([][]float32, len(c.Embeddings))
	for i, t := range c.Embeddings {
		texts[i] = t
	}
	out := []vision.Scored{}
	for _, s := range l.m.Rank(e, c.Labels, texts) {
		if (o.Top > 0 && len(out) >= o.Top) || s.Score < o.Min {
			break
		}
		out = append(out, vision.Scored{Label: s.Label, Score: s.Score})
	}
	return out
}

func (l *labeler) Probability(picture, text vision.Embedding) float64 {
	return l.m.Probability(picture, text)
}

func (l *labeler) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	err := l.m.Close()
	if l.dev != nil {
		l.dev.Close()
	}
	return err
}
