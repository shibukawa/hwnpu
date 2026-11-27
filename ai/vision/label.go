package vision

import (
	"context"
	"image"
	"io"
	"time"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/vision/internal/bridge"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/airpc"
	"github.com/shibukawa/hwnpu/internal/catalog"
	"github.com/shibukawa/hwnpu/internal/visionwire"
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
// cpu: interpreter, into Core ML packages that the CPU interpreter runs). labelRole's Prepare
// and Load are the engine's (ai/vision/engine).
var labelRole = &airole.Role{
	Remote: func(h *airpc.Handle) io.Closer { return &Labeler{remote: h} },
	Name:   "label",
	Kinds:  []catalog.Kind{catalog.KindLabel},
}

// DefaultLabels are about 440 common things, places and kinds of picture, the labels a picture
// is labeled with when an app gives none.
func DefaultLabels() []string { return visionwire.DefaultLabels() }

// ParseLabels is the labels of a list: one a line, blank lines and lines starting with # left
// out.
func ParseLabels(s string) []string { return visionwire.ParseLabels(s) }

// DefaultTemplate is the sentence each label is put in by default, {} the label.
const DefaultTemplate = "this is a photo of {}."

// Cosine is the cosine of two embeddings (their dot product: they are normalized).
func Cosine(a, b []float32) float64 { return visionwire.Dot(a, b) }

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
	// directory (hwnpu/siglip) from before.
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
	remote *airpc.Handle // the model in hwnpud (ai's HWNPU_DAEMON), nil when it is here
	local  labelerImpl   // the engine's (ai/vision/engine), when it is here
}

// labelerImpl is a label model of the engine in this process.
type labelerImpl interface {
	Model() string
	Export() string
	ImageSize() int
	Dim() int
	EmbedPicture(ctx context.Context, img image.Image) (*PictureEmbedding, error)
	EmbedTexts(ctx context.Context, texts []string) (*TextEmbeddings, error)
	Candidates(ctx context.Context, labels []string, o CandidateOptions) (*Candidates, error)
	Rank(e Embedding, c *Candidates, o RankOptions) []Scored
	Probability(picture, text Embedding) float64
	Close() error
}

// Model is the name of the model in the definitions.
func (l *Labeler) Model() string {
	if l.remote != nil {
		return airpc.Value[string](l.remote, "Model")
	}
	return l.local.Model()
}

// Export is the directory of the export or the programs the model runs from.
func (l *Labeler) Export() string {
	if l.remote != nil {
		return airpc.Value[string](l.remote, "Export")
	}
	return l.local.Export()
}

// ImageSize is the side of the square pictures are stretched to, Dim the embeddings' length.
func (l *Labeler) ImageSize() int {
	if l.remote != nil {
		return airpc.Value[int](l.remote, "ImageSize")
	}
	return l.local.ImageSize()
}
func (l *Labeler) Dim() int {
	if l.remote != nil {
		return airpc.Value[int](l.remote, "Dim")
	}
	return l.local.Dim()
}

// EmbedPicture is a picture's embedding.
func (l *Labeler) EmbedPicture(ctx context.Context, img image.Image) (*PictureEmbedding, error) {
	if l.remote != nil {
		return airpc.Call1[*PictureEmbedding](l.remote, ctx, "EmbedPicture", img)
	}
	return l.local.EmbedPicture(ctx, img)
}

// EmbedTexts are texts' embeddings (lowercased, as SigLIP 2 was trained).
func (l *Labeler) EmbedTexts(ctx context.Context, texts []string) (*TextEmbeddings, error) {
	if l.remote != nil {
		return airpc.Call1[*TextEmbeddings](l.remote, ctx, "EmbedTexts", texts)
	}
	return l.local.EmbedTexts(ctx, texts)
}

// Candidates embeds labels, each put in the options' sentence, to label pictures with (Rank):
// once for any number of pictures. The embeddings are kept in the user cache directory, so that
// a list of labels is embedded once.
func (l *Labeler) Candidates(ctx context.Context, labels []string, o CandidateOptions) (*Candidates, error) {
	if l.remote != nil {
		return airpc.Call1[*Candidates](l.remote, ctx, "Candidates", labels, o)
	}
	return l.local.Candidates(ctx, labels, o)
}

// Rank is the candidates by their probability for the picture of embedding e, the likeliest
// first: each label's on its own (a sigmoid, not a softmax over the labels), so that they need
// not sum to 1 and a picture may have several or none.
func (l *Labeler) Rank(e Embedding, c *Candidates, o RankOptions) []Scored {
	if l.remote != nil {
		return airpc.Value[[]Scored](l.remote, "Rank", e, c, o)
	}
	return l.local.Rank(e, c, o)
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
	return l.local.Probability(picture, text)
}

// Close unloads the model. An app leaves it to the Runtime (its Close, or the use's Release).
func (l *Labeler) Close() error {
	if l.remote != nil {
		return l.remote.Release() // the Runtime in hwnpud unloads it
	}
	return l.local.Close()
}

func init() {
	airole.Register(labelRole)
	bridge.Roles[labelRole.Name] = labelRole
	bridge.Wrap[labelRole.Name] = func(l any) io.Closer { return &Labeler{local: l.(labelerImpl)} }
}
