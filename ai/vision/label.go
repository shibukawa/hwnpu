package vision

import (
	"context"
	"fmt"
	"image"
	"io"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/bonsaiai"
	"github.com/shibukawa/bonsaiai/ai"
	"github.com/shibukawa/bonsaiai/internal/airole"
	"github.com/shibukawa/bonsaiai/internal/airpc"
	"github.com/shibukawa/bonsaiai/internal/cache"
	"github.com/shibukawa/bonsaiai/internal/chat"
	"github.com/shibukawa/bonsaiai/internal/imagedev"
	"github.com/shibukawa/bonsaiai/internal/llm"
	"github.com/shibukawa/bonsaiai/internal/llm/iron"
	"github.com/shibukawa/bonsaiai/internal/vision/siglip"
)

// Label is the use of a model that labels pictures and embeds pictures and texts in one space
// (SigLIP 2) for an app: a model, a family or a use of the definitions ("label", "siglip2:large").
func Label(name string) ai.Use[*Labeler] {
	return ai.Define[*Labeler](labelRole, name)
}

// The definition of a label model may give the settings cpu and the programs export (a
// directory of tools/amd/iron_siglip2.py, or one Prepare made for Apple's Neural Engine)
// (common.go). It runs on AMD's NPU from the export of iron_siglip2.py, and on Apple's Neural
// Engine from the checkpoint converted into Core AI programs on the first run (with settings
// cpu: interpreter, into Core ML packages that the CPU interpreter runs).
var labelRole = &airole.Role{
	Remote: func(h *airpc.Handle) io.Closer { return &Labeler{remote: h} },
	Name:   "label",
	Kinds:  []chat.Kind{chat.KindLabel},
	Prepare: func(e airole.PrepareEnv) (f airole.Files, err error) {
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
	},
	Load: func(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
		dir := f.Path("export")
		x, err := siglip.ReadExport(dir)
		if err != nil {
			return nil, err
		}
		var dev *bonsaiai.Device
		switch {
		case siglip.IsProgramFormat(x.Format) && interpreter(e.Def):
		case siglip.IsProgramFormat(x.Format):
			if dev, _, err = llm.OpenDevice(ctx, imagedev.ExportDevice(x.Format, e.NPU.Device), e.Cache, bonsaiai.WithPlacement(bonsaiai.PlacementStrict)); err != nil {
				return nil, err
			}
		default:
			if dev, err = bonsaiai.Open(ctx, bonsaiai.WithVendor(bonsaiai.VendorAMD)); err != nil {
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
		return &Labeler{m: m, dev: dev, name: e.Model, dir: dir}, nil
	},
}

// DefaultLabels are about 440 common things, places and kinds of picture, the labels a picture
// is labeled with when an app gives none.
func DefaultLabels() []string { return siglip.DefaultLabels() }

// ParseLabels is the labels of a list: one a line, blank lines and lines starting with # left
// out.
func ParseLabels(s string) []string { return siglip.ParseLabels(s) }

// DefaultTemplate is the sentence each label is put in by default, {} the label.
const DefaultTemplate = "this is a photo of {}."

// Cosine is the cosine of two embeddings (their dot product: they are normalized).
func Cosine(a, b []float32) float64 { return siglip.Dot(a, b) }

// Embedding is the embedding of a picture or a text: normalized, Labeler.Dim long.
type Embedding []float32

// PictureEmbedding is a picture's embedding and what making it took.
type PictureEmbedding struct {
	Embedding Embedding
	// Took is how long it all took: Input the picture made the model's input on the host, NPU
	// the model's network.
	Took, Input, NPU time.Duration
}

// TextEmbeddings are texts' embeddings, in their order, and what making them took (NPU: the
// model's network; nothing when they were kept from before).
type TextEmbeddings struct {
	Embeddings []Embedding
	Took, NPU  time.Duration
}

// Candidates are the labels a picture is labeled with, embedded (Labeler.Candidates).
type Candidates struct {
	Labels []string
	TextEmbeddings
}

// CandidateOptions are how labels are made candidates.
type CandidateOptions struct {
	// Template is the sentence each label is put in, {} the label (default DefaultTemplate;
	// "{}" the label alone).
	Template string
	// NoCache embeds the labels again instead of taking the embeddings kept in the user cache
	// directory (bonsaiai/siglip) from before.
	NoCache bool
}

// Scored is a label and the model's probability that it describes the picture.
type Scored struct {
	Label string
	Score float64
}

// RankOptions are which labels Rank keeps.
type RankOptions struct {
	// Top is the most labels kept (0: all), Min the least score of a label kept.
	Top int
	Min float64
}

// Labels are the labels of a picture, the likeliest first, and what labeling it took.
type Labels struct {
	Labels  []Scored
	Picture *PictureEmbedding
}

// Labeler labels pictures and embeds pictures and texts in one space. It runs one call at a
// time; others wait.
type Labeler struct {
	remote *airpc.Handle // the model in bonsaiaid (ai's BONSAIAI_DAEMON), nil when it is here
	mu     sync.Mutex
	m      *siglip.Model
	dev    *bonsaiai.Device
	name   string
	dir    string
}

// Model is the name of the model in the definitions.
func (l *Labeler) Model() string {
	if l.remote != nil {
		return airpc.Value[string](l.remote, "Model")
	}
	return l.name
}

// Export is the directory of the export or the programs the model runs from.
func (l *Labeler) Export() string {
	if l.remote != nil {
		return airpc.Value[string](l.remote, "Export")
	}
	return l.dir
}

// ImageSize is the side of the square pictures are stretched to, Dim the embeddings' length.
func (l *Labeler) ImageSize() int {
	if l.remote != nil {
		return airpc.Value[int](l.remote, "ImageSize")
	}
	return l.m.Config.Image
}
func (l *Labeler) Dim() int {
	if l.remote != nil {
		return airpc.Value[int](l.remote, "Dim")
	}
	return l.m.Dim()
}

// EmbedPicture is a picture's embedding.
func (l *Labeler) EmbedPicture(ctx context.Context, img image.Image) (*PictureEmbedding, error) {
	if l.remote != nil {
		return airpc.Call1[*PictureEmbedding](l.remote, ctx, "EmbedPicture", img)
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	t0 := time.Now()
	e, err := l.m.Image(ctx, img)
	if err != nil {
		return nil, err
	}
	return &PictureEmbedding{Embedding: e, Took: time.Since(t0), Input: l.m.Times[0], NPU: l.m.Times[1]}, nil
}

// EmbedTexts are texts' embeddings (lowercased, as SigLIP 2 was trained).
func (l *Labeler) EmbedTexts(ctx context.Context, texts []string) (*TextEmbeddings, error) {
	if l.remote != nil {
		return airpc.Call1[*TextEmbeddings](l.remote, ctx, "EmbedTexts", texts)
	}
	return l.texts(ctx, texts, "")
}

func (l *Labeler) texts(ctx context.Context, texts []string, keep string) (*TextEmbeddings, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	t0 := time.Now()
	l.m.Times = [3]time.Duration{}
	es, err := l.m.CachedTexts(ctx, texts, keep)
	if err != nil {
		return nil, err
	}
	out := &TextEmbeddings{Took: time.Since(t0), NPU: l.m.Times[1]}
	for _, e := range es {
		out.Embeddings = append(out.Embeddings, e)
	}
	return out, nil
}

// Candidates embeds labels, each put in the options' sentence, to label pictures with (Rank):
// once for any number of pictures. The embeddings are kept in the user cache directory, so that
// a list of labels is embedded once.
func (l *Labeler) Candidates(ctx context.Context, labels []string, o CandidateOptions) (*Candidates, error) {
	if l.remote != nil {
		return airpc.Call1[*Candidates](l.remote, ctx, "Candidates", labels, o)
	}
	tmpl := o.Template
	if tmpl == "" {
		tmpl = DefaultTemplate
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
	return &Candidates{Labels: append([]string(nil), labels...), TextEmbeddings: *t}, nil
}

// Rank is the candidates by their probability for the picture of embedding e, the likeliest
// first: each label's on its own (a sigmoid, not a softmax over the labels), so that they need
// not sum to 1 and a picture may have several or none.
func (l *Labeler) Rank(e Embedding, c *Candidates, o RankOptions) []Scored {
	if l.remote != nil {
		return airpc.Value[[]Scored](l.remote, "Rank", e, c, o)
	}
	texts := make([][]float32, len(c.Embeddings))
	for i, t := range c.Embeddings {
		texts[i] = t
	}
	out := []Scored{}
	for _, s := range l.m.Rank(e, c.Labels, texts) {
		if (o.Top > 0 && len(out) >= o.Top) || s.Score < o.Min {
			break
		}
		out = append(out, Scored{Label: s.Label, Score: s.Score})
	}
	return out
}

// Label labels a picture with the candidates: EmbedPicture, then Rank.
func (l *Labeler) Label(ctx context.Context, img image.Image, c *Candidates, o RankOptions) (*Labels, error) {
	p, err := l.EmbedPicture(ctx, img)
	if err != nil {
		return nil, err
	}
	return &Labels{Labels: l.Rank(p.Embedding, c, o), Picture: p}, nil
}

// Probability is the model's probability that the text of embedding text describes the picture
// of embedding picture.
func (l *Labeler) Probability(picture, text Embedding) float64 {
	if l.remote != nil {
		return airpc.Value[float64](l.remote, "Probability", picture, text)
	}
	return l.m.Probability(picture, text)
}

// Close unloads the model. An app leaves it to the Runtime (its Close, or the use's Release).
func (l *Labeler) Close() error {
	if l.remote != nil {
		return l.remote.Release() // the Runtime in bonsaiaid unloads it
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	err := l.m.Close()
	if l.dev != nil {
		l.dev.Close()
	}
	return err
}

func init() {
	airole.Register(labelRole)
}
