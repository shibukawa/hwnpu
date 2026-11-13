package sfx

import (
	"context"
	"fmt"
	"html"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/qwen3"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

// PromptRows are the rows the text encoder's programs take: a prompt of more tokens is cut there
// (the pipeline cuts at 512, TextRows; prompts of sound effects are a sentence or two).
const PromptRows = 256

const qwenLayers = 28

// TextFP16 are the layers the text encoder keeps in fp16 in lut8: the first three, where the
// first token's massive activation (1.2e4) is made, and the last two. The output is 4.9 % off the
// reference (relative RMS, each token's cosine 0.989 or more) against 10 % (0.943) with every layer
// in lut8, for 0.3 GB more (1.7 GB; fp16 2.8 GB, 0.7 % off).
var TextFP16 = []string{"layers.0.=fp16", "layers.1.=fp16", "layers.2.=fp16", "layers.26.=fp16", "layers.27.=fp16"}

// TextOptions are the text encoder's weight options in a format: lut8 (and int8, Intel's 8 bits)
// keeps TextFP16's layers in fp16.
func TextOptions(format string) llm.Options {
	o := llm.Options{Format: format}
	if format == "lut8" || format == "int8" {
		o.FormatFor = TextFP16
	}
	return o
}

// layerBytes is the weights of a layer of Qwen3-1.7B (50.3M) in a format.
func layerBytes(format string) float64 {
	bits := map[string]float64{"fp16": 16, "lut8": 8, "int8": 8, "lut6": 6, "lut4": 4, "int4": 4}[format]
	if bits == 0 {
		bits = 16
	}
	return 50.3e6 * bits / 8
}

// TextSplit is the layers of each of the text encoder's programs, each under 0.85 GiB (Core ML's
// limit of a package is 1 GiB).
func TextSplit(format string) []int {
	per := max(1, int(0.85*(1<<30)/layerBytes(format)))
	n := (qwenLayers + per - 1) / per
	var parts []int
	for i := range n {
		parts = append(parts, (qwenLayers*(i+1))/n-(qwenLayers*i)/n)
	}
	return parts
}

// TextConfig reads text_encoder/config.json.
func TextConfig(dir string) (*qwen3.Config, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "text_encoder", "config.json"))
	if err != nil {
		return nil, err
	}
	return qwen3.ParseConfig(raw)
}

// WriteText writes the text encoder's programs (Qwen3-1.7B over PromptRows rows, the last
// without the final norm, which the host applies) as base-N.mlpackage, in TextSplit's parts.
func WriteText(cfg *qwen3.Config, w llm.Weights, format, base string, each func(path string) error) ([]string, error) {
	return writeText(cfg, w, TextOptions(format), base, each)
}

func writeText(cfg *qwen3.Config, w llm.Weights, base0 llm.Options, base string, each func(path string) error) ([]string, error) {
	format := base0.Format
	var paths []string
	start := 0
	for i, n := range TextSplit(format) {
		path := fmt.Sprintf("%s-%d.mlpackage", base, i)
		os.RemoveAll(path)
		opts := base0
		opts.Backend, opts.Seq, opts.Start, opts.Layers, opts.NoHead = llm.BackendCoreML, PromptRows, start, n, true
		if _, err := llm.WritePart(path, cfg, w, opts, nil); err != nil {
			return nil, fmt.Errorf("sfx: text encoder part %d: %w", i, err)
		}
		if each != nil {
			if err := each(path); err != nil {
				return nil, err
			}
		}
		paths = append(paths, path)
		start += n
	}
	return paths, nil
}

var spaces = regexp.MustCompile(`\s+`)

// Prompt is the text the pipeline encodes for a prompt and a duration: the prompter's
// whitespace clean (ftfy's fix_text left out: it changes nothing in plain text), then the
// duration suffix of training.
func Prompt(prompt string, seconds float64) string {
	p := strings.TrimSpace(html.UnescapeString(html.UnescapeString(prompt)))
	p = strings.TrimSpace(spaces.ReplaceAllString(p, " "))
	return fmt.Sprintf("%s duration: %.1fs", p, math.Round(seconds*10)/10)
}

// TextEncoder runs the text encoder's programs.
type TextEncoder struct {
	tok   *tokenizer.Tokenizer
	w     llm.Weights // the embedding table and the final norm
	parts []Program
	hid   int
	eps   float64
}

// NewTextEncoder is the encoder of the tokenizer (tokenizer/tokenizer.json), the weights holding
// model.embed_tokens.weight and model.norm.weight, and the loaded parts.
func NewTextEncoder(tok *tokenizer.Tokenizer, w llm.Weights, cfg *qwen3.Config, parts []Program) *TextEncoder {
	return &TextEncoder{tok: tok, w: w, parts: parts, hid: cfg.HiddenSize, eps: cfg.NormEps}
}

// IDs are the prompt's tokens, at most PromptRows.
func (e *TextEncoder) IDs(text string) []int32 {
	ids := e.tok.Encode(text)
	if len(ids) > PromptRows {
		ids = ids[:PromptRows]
	}
	return ids
}

// Encode is the text encoder's output for the text (Prompt's): TextRows x 2048, the last hidden
// state after the final norm in the prompt's rows and zeros after them, and the tokens.
func (e *TextEncoder) Encode(ctx context.Context, text string) ([]float32, []int32, error) {
	ids := e.IDs(text)
	d := e.hid
	x := make([]float32, PromptRows*d)
	// the rows after the prompt are padding (<|endoftext|>): with causal attention they change
	// nothing before them, and their output is zeroed
	for i := range PromptRows {
		id := int32(151643)
		if i < len(ids) {
			id = ids[i]
		}
		row, err := e.w.Rows("model.embed_tokens.weight", int(id), int(id)+1)
		if err != nil {
			return nil, nil, err
		}
		copy(x[i*d:], row)
	}
	for i, p := range e.parts {
		y, err := p.Run(ctx, map[string][]float32{"x": x})
		if err != nil {
			return nil, nil, fmt.Errorf("sfx: text encoder part %d: %w", i, err)
		}
		x = y
	}
	norm, _, err := e.w.Get("model.norm.weight")
	if err != nil {
		return nil, nil, err
	}
	out := make([]float32, TextRows*d)
	for r := range len(ids) {
		row := x[r*d : (r+1)*d]
		var ss float64
		for _, v := range row {
			ss += float64(v) * float64(v)
		}
		inv := 1 / math.Sqrt(ss/float64(d)+e.eps)
		for j, v := range row {
			out[r*d+j] = float32(float64(v)*inv) * norm[j]
		}
	}
	return out, ids, nil
}
