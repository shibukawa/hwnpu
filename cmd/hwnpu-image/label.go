package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/vision"
	"github.com/shibukawa/hwnpu/internal/console"
)

// LabelCmd is "hwnpu-image label": zero-shot labels of pictures with SigLIP 2 (google/siglip2-*,
// Apache-2.0; ai/vision's Label) on AMD's NPU (exports of tools/amd/iron_siglip2.py) or Apple's
// Neural Engine (the checkpoint downloaded and converted into Core AI programs on the first run;
// --cpu: Core ML packages on the CPU interpreter).
//
// Each label is put in --template's sentence ("this is a photo of {}."), and the model gives
// the probability that the sentence describes the picture: each label on its own (a sigmoid,
// not a softmax over the labels), so they need not sum to 1 and a picture may have several or
// none. Without --labels or --labels-file it ranks a list of about 440 common things, places
// and kinds of picture (internal/vision/siglip/labels.txt). The texts' embeddings are kept in
// the user cache directory, so a list is embedded once.
//
// For each picture it prints one line of JSON: the --top labels with a score of at least --min,
// the most likely first.
type LabelCmd struct {
	Pictures    []string `arg:"" type:"existingfile" help:"The pictures (JPEG or PNG)."`
	Model       string   `short:"m" placeholder:"NAME|EXPORT" default:"label" help:"The model: a name, a family or a use of models.yaml (label: siglip2), run from NAME.amd-npu2 (in the cache's entries/amd-xdna2 or ~/models), or an export's directory (tools/amd/iron_siglip2.py)."`
	Labels      []string `short:"l" placeholder:"LABEL" help:"The candidate labels (any language; separated by commas)."`
	LabelsFile  string   `type:"existingfile" placeholder:"FILE" help:"The candidate labels, one a line (# starts a comment)."`
	Template    string   `default:"this is a photo of {}." help:"The sentence each label is put in, {} the label (\"{}\": the label alone)."`
	Top         int      `default:"10" help:"The most labels to print for a picture (0: all)."`
	Min         float64  `default:"0" help:"The least score of a label to print."`
	Repeat      int      `default:"1" help:"Run each picture this many times and report the mean time."`
	NoCache     bool     `help:"Embed the labels again instead of reading them from the cache."`
	siglipFlags `embed:""`
}

// Run ranks the labels for each picture.
func (c *LabelCmd) Run(ctx context.Context) error {
	lw := console.Log(os.Stderr)
	logf := func(format string, args ...any) { fmt.Fprintf(lw, format+"\n", args...) }
	labels, err := c.labels()
	if err != nil {
		return err
	}
	if !strings.Contains(c.Template, "{}") {
		return fmt.Errorf("--template %q has no {}", c.Template)
	}
	m, rt, err := openSiglip(ctx, c.Model, c.siglipFlags, lw, logf)
	if err != nil {
		return err
	}
	defer rt.Close()
	defer npuHeld(logf)()
	console.Running()
	cands, err := m.Candidates(ctx, labels, vision.CandidateOptions{Template: c.Template, NoCache: c.NoCache})
	if err != nil {
		return err
	}
	if cands.Took > 50*time.Millisecond {
		logf("embedded %d labels in %.2f s (NPU %.2f s)", len(labels), cands.Took.Seconds(), cands.NPU.Seconds())
	}
	enc := json.NewEncoder(os.Stdout)
	for _, p := range c.Pictures {
		img, err := readPicture(p)
		if err != nil {
			return err
		}
		var e *vision.PictureEmbedding
		start := time.Now()
		for i := range max(1, c.Repeat) {
			if i == 1 {
				start = time.Now() // the first loads the graph
			}
			if e, err = m.EmbedPicture(ctx, img); err != nil {
				return fmt.Errorf("%s: %w", p, err)
			}
		}
		runs := max(1, c.Repeat-1)
		logf("%s: %.1f ms (input %.1f, NPU %.1f)%s", filepath.Base(p), ms(time.Since(start))/float64(runs),
			ms(e.Input), ms(e.NPU), repeated(c.Repeat))
		out := []labelJSON{}
		for _, l := range m.Rank(e.Embedding, cands, vision.RankOptions{Top: c.Top, Min: c.Min}) {
			out = append(out, labelJSON(l))
		}
		if err := enc.Encode(struct {
			File   string      `json:"file"`
			Labels []labelJSON `json:"labels"`
		}{p, out}); err != nil {
			return err
		}
	}
	return nil
}

// labelJSON is a label as label prints it, with its score.
type labelJSON struct {
	Label string  `json:"label"`
	Score float64 `json:"score"`
}

// labels are --labels, --labels-file's, or the default list.
func (c *LabelCmd) labels() ([]string, error) {
	var out []string
	for _, l := range c.Labels {
		if l = strings.TrimSpace(l); l != "" {
			out = append(out, l)
		}
	}
	if c.LabelsFile != "" {
		b, err := os.ReadFile(c.LabelsFile)
		if err != nil {
			return nil, err
		}
		out = append(out, vision.ParseLabels(string(b))...)
	}
	if len(out) == 0 && (len(c.Labels) > 0 || c.LabelsFile != "") {
		return nil, errors.New("no labels in --labels or --labels-file")
	}
	if len(out) == 0 {
		out = vision.DefaultLabels()
	}
	return out, nil
}

// EmbedCmd is "hwnpu-image embed": embeddings of pictures and texts in SigLIP 2's one space
// (google/siglip2-*, Apache-2.0; exports of tools/amd/iron_siglip2.py on AMD's NPU, Core AI
// programs converted on the first run on Apple's Neural Engine), for image search and
// similarity: the cosine (the dot product: they are normalized) of a picture's and a text's, or
// of two pictures'.
//
// It prints a line of JSON for each picture and each --text: the file or the text, and the
// embedding. With --compare it prints the cosines of every picture with every text (and of the
// pictures with each other without texts) instead.
type EmbedCmd struct {
	Pictures    []string `arg:"" optional:"" type:"existingfile" help:"The pictures (JPEG or PNG)."`
	Model       string   `short:"m" placeholder:"NAME|EXPORT" default:"label" help:"The model: a name, a family or a use of models.yaml (label: siglip2), or an export's directory (tools/amd/iron_siglip2.py)."`
	Text        []string `short:"t" placeholder:"TEXT" sep:"none" help:"A text to embed (repeat for more; lowercased, as SigLIP 2 was trained)."`
	Compare     bool     `help:"Print the cosines instead of the embeddings."`
	siglipFlags `embed:""`
}

// Run embeds the pictures and the texts.
func (c *EmbedCmd) Run(ctx context.Context) error {
	if len(c.Pictures) == 0 && len(c.Text) == 0 {
		return errors.New("no pictures and no --text")
	}
	lw := console.Log(os.Stderr)
	logf := func(format string, args ...any) { fmt.Fprintf(lw, format+"\n", args...) }
	m, rt, err := openSiglip(ctx, c.Model, c.siglipFlags, lw, logf)
	if err != nil {
		return err
	}
	defer rt.Close()
	defer npuHeld(logf)()
	console.Running()
	var imgs []vision.Embedding
	for _, p := range c.Pictures {
		img, err := readPicture(p)
		if err != nil {
			return err
		}
		e, err := m.EmbedPicture(ctx, img)
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		imgs = append(imgs, e.Embedding)
	}
	var texts []vision.Embedding
	if len(c.Text) > 0 {
		t, err := m.EmbedTexts(ctx, c.Text)
		if err != nil {
			return err
		}
		texts = t.Embeddings
	}
	enc := json.NewEncoder(os.Stdout)
	if !c.Compare {
		for i, p := range c.Pictures {
			if err := enc.Encode(struct {
				File      string    `json:"file"`
				Embedding []float32 `json:"embedding"`
			}{p, imgs[i]}); err != nil {
				return err
			}
		}
		for i, t := range c.Text {
			if err := enc.Encode(struct {
				Text      string    `json:"text"`
				Embedding []float32 `json:"embedding"`
			}{t, texts[i]}); err != nil {
				return err
			}
		}
		return nil
	}
	type pair struct {
		File   string  `json:"file"`
		Text   string  `json:"text,omitempty"`
		Other  string  `json:"other,omitempty"`
		Cosine float64 `json:"cosine"`
		Score  float64 `json:"score,omitempty"` // the probability that the text describes the picture
	}
	for i, p := range c.Pictures {
		for j, t := range c.Text {
			if err := enc.Encode(pair{File: p, Text: t, Cosine: vision.Cosine(imgs[i], texts[j]), Score: m.Probability(imgs[i], texts[j])}); err != nil {
				return err
			}
		}
		if len(c.Text) == 0 {
			for j := i + 1; j < len(c.Pictures); j++ {
				if err := enc.Encode(pair{File: p, Other: c.Pictures[j], Cosine: vision.Cosine(imgs[i], imgs[j])}); err != nil {
					return err
				}
			}
		}
	}
	if len(c.Pictures) == 0 {
		for i := range c.Text {
			for j := i + 1; j < len(c.Text); j++ {
				if err := enc.Encode(struct {
					Text   string  `json:"text"`
					Other  string  `json:"other"`
					Cosine float64 `json:"cosine"`
				}{c.Text[i], c.Text[j], vision.Cosine(texts[i], texts[j])}); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// siglipFlags are label's and embed's flags for the converted programs (Apple's Neural Engine).
type siglipFlags struct {
	backendFlag `embed:""`
	CPU         bool `name:"cpu" help:"Run the converted programs on the CPU interpreter instead of the NPU (slow; for checking a conversion)."`
}

// openSiglip loads SigLIP 2 for --model (ai/vision's Label): on AMD's NPU its export of
// tools/amd/iron_siglip2.py, on Apple's Neural Engine the Core AI programs converted on the
// first run (with --cpu the Core ML packages the CPU interpreter runs), or an export's directory
// of either. The Runtime closes it.
func openSiglip(ctx context.Context, model string, f siglipFlags, lw io.Writer, logf func(string, ...any)) (*vision.Labeler, *ai.Runtime, error) {
	useBackend(f.Backend)
	var how made
	how.interpreter(f.CPU)
	name := model
	if isPath(model) {
		how.program("export", expandHome(model))
		name = "label"
	}
	m, rt, _, err := load(ctx, name, vision.Label, how, lw)
	if err != nil {
		return nil, nil, err
	}
	logf("%s: SigLIP 2, %dx%d pictures, %d-dimensional embeddings", filepath.Base(m.Export()), m.ImageSize(), m.ImageSize(), m.Dim())
	return m, rt, nil
}
