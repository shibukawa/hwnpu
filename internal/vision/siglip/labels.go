package siglip

import (
	"bufio"
	"context"
	"crypto/sha256"
	_ "embed"
	"encoding/binary"
	"encoding/hex"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

//go:embed labels.txt
var labelsTxt string

// DefaultLabels are hwnpu-image label's labels when none are given: common things, places and
// kinds of picture (labels.txt).
func DefaultLabels() []string { return ParseLabels(labelsTxt) }

// ParseLabels is the labels of a list: one a line, blank lines and lines starting with # left
// out.
func ParseLabels(s string) []string {
	var out []string
	sc := bufio.NewScanner(strings.NewReader(s))
	for sc.Scan() {
		l := strings.TrimSpace(sc.Text())
		if l == "" || strings.HasPrefix(l, "#") {
			continue
		}
		out = append(out, l)
	}
	return out
}

// Label is a label and the model's probability that it describes the picture.
type Label struct {
	Label string  `json:"label"`
	Score float64 `json:"score"`
}

// Rank is the labels by their probability for the picture of embedding img, the most likely
// first; texts are their embeddings.
func (m *Model) Rank(img []float32, labels []string, texts [][]float32) []Label {
	out := make([]Label, len(labels))
	for i, l := range labels {
		out[i] = Label{l, m.Probability(img, texts[i])}
	}
	slices.SortStableFunc(out, func(a, b Label) int {
		switch {
		case a.Score > b.Score:
			return -1
		case a.Score < b.Score:
			return 1
		}
		return 0
	})
	return out
}

// CachedTexts is Texts, kept in dir (a file for the model's name and the texts): a list of
// labels is embedded once. dir "" does not keep them.
func (m *Model) CachedTexts(ctx context.Context, texts []string, dir string) ([][]float32, error) {
	if dir == "" {
		return m.Texts(ctx, texts)
	}
	h := sha256.New()
	h.Write([]byte(m.Name))
	for _, t := range texts {
		h.Write([]byte{0})
		h.Write([]byte(t))
	}
	path := filepath.Join(dir, m.Name+"-"+hex.EncodeToString(h.Sum(nil))[:16]+".f32")
	P := m.Config.Projection
	if b, err := os.ReadFile(path); err == nil && len(b) == 4*P*len(texts) {
		out := make([][]float32, len(texts))
		for i := range out {
			out[i] = make([]float32, P)
			for j := range out[i] {
				out[i][j] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*(i*P+j):]))
			}
		}
		return out, nil
	}
	out, err := m.Texts(ctx, texts)
	if err != nil {
		return nil, err
	}
	b := make([]byte, 0, 4*P*len(texts))
	for _, e := range out {
		for _, v := range e {
			b = binary.LittleEndian.AppendUint32(b, math.Float32bits(v))
		}
	}
	if os.MkdirAll(dir, 0o755) == nil {
		tmp := path + ".tmp"
		if os.WriteFile(tmp, b, 0o644) == nil {
			os.Rename(tmp, path)
		}
	}
	return out, nil
}
