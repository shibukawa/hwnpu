package sana

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/imagegen/sana/mil"
	"github.com/shibukawa/hwnpu/internal/llm"
)

// ExportVersion is the layout of the programs Convert writes; an export of another is converted
// again. 2: the time embedding on the host (TimeFile); 3: Gemma 2's GELU written with exp and
// real_div (mil.Text).
const ExportVersion = 3

// ConvertOptions select what Convert writes.
type ConvertOptions struct {
	// Width and Height are the picture's size in pixels, multiples of 32.
	Width, Height int
	// TextFormat and DitFormat are the weight formats of Gemma 2 and of the DiT (and the
	// caption), as hwnpu-convert's -format: fp16, int8, lut8, lut6... The VAE's are fp16.
	TextFormat, DitFormat string
	// Group is the DiT's blocks a program holds; TextGroup Gemma 2's layers.
	Group, TextGroup int
	// Only, when not empty, writes these parts alone (text, caption, time, blocks, final, vae).
	Only map[string]bool
}

// Formats are the default weight formats of Gemma 2 and the DiT: int8 for Gemma 2 (its 5.2 GB
// at fp16 would be a dozen programs; int8 is near fp16, design.md 16.85), fp16 for the DiT.
func Formats() (text, dit string) { return "int8", "fp16" }

func (o *ConvertOptions) defaults() {
	text, dit := Formats()
	if o.TextFormat == "" {
		o.TextFormat = text
	}
	if o.DitFormat == "" {
		o.DitFormat = dit
	}
	if o.Group <= 0 {
		o.Group = 4
	}
	if o.TextGroup <= 0 {
		// a program within about 500 MB of weights (78 MB a layer at int8, 156 at fp16)
		o.TextGroup = 6
		if o.TextFormat == "fp16" {
			o.TextGroup = 3
		}
	}
	if o.Width == 0 {
		o.Width = 1024
	}
	if o.Height == 0 {
		o.Height = o.Width
	}
}

// Convert writes a Core ML export of the Sana repository root (transformer/, vae/,
// scheduler/; Gemma 2's text_encoder/ and tokenizer/ in text, which may be root) into out: the
// programs, text/ (the tokenizer and the embedding table, bf16, as the IRON export's), model.json
// and manifest.json.
func Convert(root, text, out string, o ConvertOptions, log io.Writer) error {
	o.defaults()
	if log == nil {
		log = io.Discard
	}
	dc, err := mil.ReadDit(root, o.Width, o.Height)
	if err != nil {
		return err
	}
	if dc.Layers%o.Group != 0 {
		return fmt.Errorf("sana: %d blocks are no whole groups of %d", dc.Layers, o.Group)
	}
	gc, err := mil.ReadGemma(text)
	if err != nil {
		return err
	}
	var vcfg struct {
		Channels []int   `json:"decoder_block_out_channels"`
		Up       string  `json:"upsample_block_type"`
		Latent   int     `json:"latent_channels"`
		Scaling  float64 `json:"scaling_factor"`
		Layers   []int   `json:"decoder_layers_per_block"`
	}
	b, err := os.ReadFile(filepath.Join(root, "vae", "config.json"))
	if err != nil {
		return err
	}
	if err := json.Unmarshal(b, &vcfg); err != nil {
		return fmt.Errorf("sana: vae/config.json: %w", err)
	}
	if fmt.Sprint(vcfg.Channels) != fmt.Sprint(mil.VaeChannels) || vcfg.Up != "interpolate" || vcfg.Latent != dc.Latent ||
		fmt.Sprint(vcfg.Layers) != "[3 3 3 3 3 3]" {
		return fmt.Errorf("sana: the VAE (%v, %s, %v) is not DC-AE f32c32's", vcfg.Channels, vcfg.Up, vcfg.Layers)
	}
	model, err := readModel(root)
	if err != nil {
		return err
	}
	w, err := mil.OpenWeights(root, text)
	if err != nil {
		return err
	}
	defer w.Close()
	if err := os.MkdirAll(filepath.Join(out, "text"), 0o755); err != nil {
		return err
	}
	groups := mil.TextGroups(gc.Layers, o.TextGroup)
	stages := mil.VaeStages(o.Width, o.Height)
	total := len(groups) + 2 + dc.Layers/o.Group + len(stages)
	written := 0
	write := func(kind string, p *mil.Part, format string) (string, error) {
		name := p.Name + ".mlpackage"
		if len(o.Only) > 0 && !o.Only[kind] {
			return name, nil
		}
		written++
		bar := console.NewBar(log, fmt.Sprintf("  [%2d/%d] %s", written, total, name))
		if _, err := llm.WritePart(filepath.Join(out, name), p, w, llm.Options{Backend: llm.BackendCoreML, Format: format}, nil); err != nil {
			bar.Done("failed")
			return "", fmt.Errorf("sana: %s: %w", p.Name, err)
		}
		bar.Done(fmt.Sprintf("%s, %s", console.Duration(bar.Elapsed()), format))
		return name, nil
	}
	m := &ProgramManifest{Backend: llm.BackendCoreML, Version: ExportVersion,
		Formats: map[string]string{"text": o.TextFormat, "dit": o.DitFormat, "vae": "fp16"},
		Dit: DitConfig{Width: o.Width, Height: o.Height, Hidden: dc.Hidden, Layers: dc.Layers, Latent: dc.Latent,
			Guidance: dc.Guidance, GuidanceScale: dc.GuidanceScale, TextRows: mil.TextRows, Group: o.Group,
			VAEScale: mil.VAEScale, Caption: dc.Caption},
		Text: TextConfig{Rows: mil.TextLen, Hidden: gc.Hidden, Vocab: gc.Vocab}}
	for _, gr := range groups {
		name, err := write("text", mil.Text(gc, gr[0], gr[1]), o.TextFormat)
		if err != nil {
			return err
		}
		m.TextParts = append(m.TextParts, name)
	}
	if m.Caption, err = write("caption", mil.Caption(dc), o.DitFormat); err != nil {
		return err
	}
	m.Time = TimeFile
	if len(o.Only) == 0 || o.Only["time"] {
		if err := writeTimeFile(filepath.Join(out, TimeFile), w, dc.Guidance); err != nil {
			return err
		}
	}
	for first := 0; first < dc.Layers; first += o.Group {
		name, err := write("blocks", mil.Blocks(dc, first, o.Group), o.DitFormat)
		if err != nil {
			return err
		}
		m.Blocks = append(m.Blocks, name)
	}
	if m.Final, err = write("final", mil.Final(dc), o.DitFormat); err != nil {
		return err
	}
	for i := range stages {
		name, err := write("vae", mil.Vae(o.Width, o.Height, i, vcfg.Scaling), "fp16")
		if err != nil {
			return err
		}
		m.Vae = append(m.Vae, name)
	}
	if len(o.Only) == 0 || o.Only["text"] {
		if err := writeTextFiles(text, out, w, gc); err != nil {
			return err
		}
	}
	mb, err := json.MarshalIndent(model, "", " ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(out, "model.json"), append(mb, '\n'), 0o644); err != nil {
		return err
	}
	return writeProgramManifest(out, m)
}

// readModel is model.json of root: which Sana it is and its scheduler's settings (as
// tools/amd/iron_sana.py's write_model).
func readModel(root string) (Model, error) {
	var tc struct {
		Guidance bool `json:"guidance_embeds"`
	}
	b, err := os.ReadFile(filepath.Join(root, "transformer", "config.json"))
	if err != nil {
		return Model{}, err
	}
	if err := json.Unmarshal(b, &tc); err != nil {
		return Model{}, err
	}
	var sc struct {
		Class      string  `json:"_class_name"`
		Prediction string  `json:"prediction_type"`
		SigmaData  float64 `json:"sigma_data"`
		Algorithm  string  `json:"algorithm_type"`
		Order      int     `json:"solver_order"`
		FlowSigmas bool    `json:"use_flow_sigmas"`
		FlowShift  float64 `json:"flow_shift"`
		Train      int     `json:"num_train_timesteps"`
	}
	if b, err = os.ReadFile(filepath.Join(root, "scheduler", "scheduler_config.json")); err != nil {
		return Model{}, err
	}
	if err := json.Unmarshal(b, &sc); err != nil {
		return Model{}, err
	}
	m := Model{Name: filepath.Base(root), Sprint: tc.Guidance, Guidance: 4.5}
	if tc.Guidance {
		if sc.Class != "SCMScheduler" || sc.Prediction != "trigflow" {
			return m, fmt.Errorf("sana: scheduler %s (%s) is not Sana-Sprint's", sc.Class, sc.Prediction)
		}
		m.Steps, m.SigmaData, m.MaxT = 2, sc.SigmaData, 1.57080
	} else {
		if sc.Class != "DPMSolverMultistepScheduler" || sc.Prediction != "flow_prediction" || sc.Algorithm != "dpmsolver++" ||
			sc.Order != 2 || !sc.FlowSigmas {
			return m, fmt.Errorf("sana: scheduler %s (%s, %s, order %d) is not Sana's", sc.Class, sc.Prediction, sc.Algorithm, sc.Order)
		}
		m.Steps, m.FlowShift, m.TrainSteps = 20, sc.FlowShift, sc.Train
	}
	return m, nil
}

// writeTextFiles writes text/: the tokenizer and Gemma 2's embedding table (bf16) the host reads
// the prompt's rows from.
func writeTextFiles(text, out string, w *mil.Weights, gc mil.Gemma) error {
	b, err := os.ReadFile(filepath.Join(text, "tokenizer", "tokenizer.json"))
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(out, "text", "tokenizer.json"), b, 0o644); err != nil {
		return err
	}
	f, err := os.Create(filepath.Join(out, "text", "embed.bin"))
	if err != nil {
		return err
	}
	const chunk = 8192 // rows at a time: the whole table is 2.4 GB as float32
	buf := make([]byte, 2*chunk*gc.Hidden)
	for r := 0; r < gc.Vocab; r += chunk {
		e := min(r+chunk, gc.Vocab)
		v, err := w.Rows("gemma.embed_tokens.weight", r, e)
		if err != nil {
			f.Close()
			return err
		}
		for i, x := range v {
			binary.LittleEndian.PutUint16(buf[2*i:], bf16(x))
		}
		if _, err := f.Write(buf[:2*len(v)]); err != nil {
			f.Close()
			return err
		}
	}
	return f.Close()
}
