package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	hcache "github.com/shibukawa/hwnpu/internal/cache"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	"github.com/shibukawa/hwnpu/internal/vision/siglip"
)

// LabelCmd is "hwnpu-image label": zero-shot labels of pictures with SigLIP 2 (google/siglip2-*,
// Apache-2.0; internal/vision/siglip) on AMD's NPU (exports of tools/amd/iron_siglip2.py) or
// Apple's Neural Engine (the checkpoint downloaded and converted into Core ML packages on the
// first run; --backend coreai runs them on Core AI).
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
	m, closeAll, err := openSiglip(ctx, c.Model, c.siglipFlags, lw, logf)
	if err != nil {
		return err
	}
	defer closeAll()
	sentences := make([]string, len(labels))
	for i, l := range labels {
		sentences[i] = strings.ReplaceAll(c.Template, "{}", l)
	}
	cache := ""
	if !c.NoCache {
		cache = siglipCache()
	}
	console.Running()
	t0 := time.Now()
	texts, err := m.CachedTexts(ctx, sentences, cache)
	if err != nil {
		return err
	}
	if d := time.Since(t0); d > 50*time.Millisecond {
		logf("embedded %d labels in %.2f s (NPU %.2f s)", len(labels), d.Seconds(), m.Times[1].Seconds())
	}
	enc := json.NewEncoder(os.Stdout)
	for _, p := range c.Pictures {
		img, err := readPicture(p)
		if err != nil {
			return err
		}
		var e []float32
		start := time.Now()
		for i := range max(1, c.Repeat) {
			if i == 1 {
				start = time.Now() // the first loads the graph
			}
			if e, err = m.Image(ctx, img); err != nil {
				return fmt.Errorf("%s: %w", p, err)
			}
		}
		runs := max(1, c.Repeat-1)
		logf("%s: %.1f ms (input %.1f, NPU %.1f)%s", filepath.Base(p), ms(time.Since(start))/float64(runs),
			ms(m.Times[0]), ms(m.Times[1]), repeated(c.Repeat))
		ranked := m.Rank(e, labels, texts)
		out := []siglip.Label{}
		for _, l := range ranked {
			if (c.Top > 0 && len(out) >= c.Top) || l.Score < c.Min {
				break
			}
			out = append(out, l)
		}
		if err := enc.Encode(struct {
			File   string         `json:"file"`
			Labels []siglip.Label `json:"labels"`
		}{p, out}); err != nil {
			return err
		}
	}
	return nil
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
		out = append(out, siglip.ParseLabels(string(b))...)
	}
	if len(out) == 0 && (len(c.Labels) > 0 || c.LabelsFile != "") {
		return nil, errors.New("no labels in --labels or --labels-file")
	}
	if len(out) == 0 {
		out = siglip.DefaultLabels()
	}
	return out, nil
}

// EmbedCmd is "hwnpu-image embed": embeddings of pictures and texts in SigLIP 2's one space
// (google/siglip2-*, Apache-2.0; exports of tools/amd/iron_siglip2.py on AMD's NPU, Core ML
// packages converted on the first run on Apple's Neural Engine), for image
// search and similarity: the cosine (the dot product: they are normalized) of a picture's and a
// text's, or of two pictures'.
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
	m, closeAll, err := openSiglip(ctx, c.Model, c.siglipFlags, lw, logf)
	if err != nil {
		return err
	}
	defer closeAll()
	console.Running()
	var imgs [][]float32
	for _, p := range c.Pictures {
		img, err := readPicture(p)
		if err != nil {
			return err
		}
		e, err := m.Image(ctx, img)
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		imgs = append(imgs, e)
	}
	var texts [][]float32
	if len(c.Text) > 0 {
		if texts, err = m.Texts(ctx, c.Text); err != nil {
			return err
		}
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
			if err := enc.Encode(pair{File: p, Text: t, Cosine: siglip.Dot(imgs[i], texts[j]), Score: m.Probability(imgs[i], texts[j])}); err != nil {
				return err
			}
		}
		if len(c.Text) == 0 {
			for j := i + 1; j < len(c.Pictures); j++ {
				if err := enc.Encode(pair{File: p, Other: c.Pictures[j], Cosine: siglip.Dot(imgs[i], imgs[j])}); err != nil {
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
				}{c.Text[i], c.Text[j], siglip.Dot(texts[i], texts[j])}); err != nil {
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

// openSiglip loads SigLIP 2 for --model: on AMD's NPU its export of tools/amd/iron_siglip2.py,
// on Apple's Neural Engine the Core AI programs siglip.Prepare converts on the first run (the
// checkpoint downloaded; with --cpu the Core ML packages the CPU interpreter runs), or an export's directory of either. The closer releases it and the
// device.
func openSiglip(ctx context.Context, model string, f siglipFlags, lw io.Writer, logf func(string, ...any)) (*siglip.Model, func(), error) {
	dir := expandHome(model)
	if !isPath(model) {
		m, err := resolve(model, chat.KindLabel)
		if err != nil {
			return nil, nil, err
		}
		format := ""
		if f.CPU {
			format = llm.BackendCoreML // what the CPU interpreter runs
		} else if chat.HereVendor() != "amd" || f.Backend != "" {
			if format, err = imageFormat(ctx, f.Backend); err != nil && runtime.GOOS == "darwin" {
				return nil, nil, err
			}
		}
		switch format {
		case llm.BackendCoreML, llm.BackendCoreAI:
			if dir, err = siglip.Prepare(chat.WithModel(ctx, m), siglip.Setup{Dir: chat.ModelsDir(), Repo: m.Source(), Name: m.Name, Format: format, Log: lw}); err != nil {
				return nil, nil, err
			}
		case "":
			if dir, err = amdExport(m, "tools/amd/iron_siglip2.py"); err != nil {
				return nil, nil, err
			}
		default:
			return nil, nil, fmt.Errorf("SigLIP 2 runs on AMD's NPU and Apple's Neural Engine so far (this machine's NPU takes %s programs)", format)
		}
	}
	x, err := siglip.ReadExport(dir)
	if err != nil {
		return nil, nil, err
	}
	var dev *hwnpu.Device
	switch {
	case siglip.IsProgramFormat(x.Format) && f.CPU:
	case siglip.IsProgramFormat(x.Format):
		d, _, err := llm.OpenDevice(ctx, exportDevice(x.Format, f.Backend), cacheDirOf(""), hwnpu.WithPlacement(hwnpu.PlacementStrict))
		if err != nil {
			return nil, nil, err
		}
		dev = d
	default:
		d, err := hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD))
		if err != nil {
			return nil, nil, err
		}
		dev = d
		iron.ShareExport(dir, logf)
	}
	closeDev := func() {
		if dev != nil {
			dev.Close()
		}
	}
	m, err := siglip.Open(ctx, dev, dir)
	if err != nil {
		closeDev()
		return nil, nil, err
	}
	m.Logf = logf
	logf("%s: SigLIP 2, %dx%d pictures, %d-dimensional embeddings", filepath.Base(dir), m.Config.Image, m.Config.Image, m.Dim())
	held := func() {}
	if x.Format == siglip.FormatCoreML && dev != nil {
		held = npuHeld(logf)
	}
	return m, func() {
		held()
		m.Close()
		closeDev()
	}, nil
}

// siglipCache is where label embeddings are kept: hwnpu/siglip in the user cache directory.
func siglipCache() string {
	return hcache.Sub("siglip")
}
