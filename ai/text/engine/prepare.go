package engine

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/shibukawa/hwnpu/ai/text/internal/bridge"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/systemone"
)

func prepareEmbed(e airole.PrepareEnv) (f airole.Files, err error) {
	m, ps := e.Model, e.Model.Programs
	if e.NPU.AMD() && len(ps["packages"]) == 0 {
		// AMD's NPU: an export of tools/amd/iron_plamo_embed.py, from the sources or made
		// with IRON (the CPU interpreter runs no embedding model)
		dir, err := chat.PrepareEmbedderAMD(e.Ctx, m, e.Log)
		if err != nil {
			return f, err
		}
		f.Set("export", dir)
		f.SetValue("name", e.Base.Name)
		return f, nil
	}
	if pk := ps["packages"]; len(pk) > 0 {
		f.Set("packages", pk...)
		f.Set("config", cmp.Or(first(ps["config"]), filepath.Join(chat.RepoDir(m.Source()), "config.json")))
		f.Set("weights", cmp.Or(first(ps["weights"]), chat.WeightsIn(m)))
		f.SetValue("name", e.Base.Name)
		f.SetValue("backend", e.NPU.Device)
		return f, nil
	}
	seqs, err := ints(e.Model.Settings["seqs"], chat.DefaultEmbedSeqs)
	if err != nil {
		return f, err
	}
	c, err := chat.PrepareEmbedder(e.Ctx, chat.EmbedSetup{Dir: e.Dir, Repo: m.Source(), Name: e.Base.Name, Seqs: seqs, Backend: e.NPU.Device, Log: e.Log})
	if err != nil {
		return f, err
	}
	f.Set("packages", c.Packages...)
	f.Set("config", c.ModelConfig)
	f.Set("weights", c.Weights)
	f.Set("tokenizer", c.Tokenizer)
	f.SetValue("name", cmp.Or(c.Name, e.Base.Name))
	f.SetValue("backend", c.Backend)
	return f, nil
}

func loadEmbed(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
	if dir := f.Path("export"); dir != "" {
		em, err := chat.LoadEmbedderAMD(ctx, dir, f.Values["name"], e.Log)
		if err != nil {
			return nil, err
		}
		return bridge.Wrap["embed"](&embedder{e: em, model: e.Model}), nil
	}
	em, err := chat.LoadEmbedder(ctx, chat.EmbedConfig{Packages: f.Paths["packages"], ModelConfig: f.Path("config"), Weights: f.Path("weights"),
		Tokenizer: f.Path("tokenizer"), Name: f.Values["name"], Backend: f.Values["backend"], CacheDir: e.Cache, Log: e.Log})
	if err != nil {
		return nil, err
	}
	return bridge.Wrap["embed"](&embedder{e: em, model: e.Model}), nil
}

func prepareDecide(e airole.PrepareEnv) (f airole.Files, err error) {
	m, ps := e.Model, e.Model.Programs
	if e.NPU.AMD() && len(ps["packages"]) == 0 && !e.NPU.CPU(m) {
		// AMD's NPU: an export of tools/amd/iron_export_lm.py d1 (its recipe, model-recipes.yaml),
		// from the sources or made with IRON, as a chat model's
		if len(m.AMDExport) == 0 {
			return f, noAMDForm(e)
		}
		dir, err := chat.AMDExportOf(e.Ctx, m, e.Log)
		if err != nil {
			return f, err
		}
		f.Set("export", dir)
		f.SetValue("name", e.Base.Name)
		return f, nil
	}
	if pk := ps["packages"]; len(pk) > 0 {
		config := first(ps["config"])
		if config == "" {
			return f, errors.New("text: a decision model's packages need its config (the checkpoint's config.json)")
		}
		f.Set("packages", pk...)
		f.Set("config", config)
		f.Set("weights", cmp.Or(first(ps["weights"]), filepath.Join(filepath.Dir(config), "model.safetensors")))
	} else {
		seqs, err := ints(m.Settings["seqs"], systemone.DefaultSeqs)
		if err != nil {
			return f, err
		}
		backend := ""
		if !e.NPU.CPU(m) { // the CPU interpreter runs Core ML packages
			if backend, err = llm.NativeFormat(e.Ctx, e.NPU.Device); err != nil {
				return f, err
			}
		}
		c, err := systemone.Prepare(e.Ctx, systemone.Setup{Dir: e.Dir, Repo: m.Source(), Seqs: seqs, Format: m.Settings["format"], Backend: backend, Log: e.Log})
		if err != nil {
			return f, err
		}
		f.Set("packages", c.Packages...)
		f.Set("config", c.ModelConfig)
		f.Set("weights", c.Weights)
		f.Set("tokenizer", c.Tokenizer)
		f.Set("vision", c.Vision...)
		f.Set("audio", c.Audio...)
		f.SetValue("name", e.Base.Name)
	}
	if t := ps["tokenizer"]; len(t) > 0 { // one converted by hand
		f.Set("tokenizer", t...)
	}
	f.Set("calibration", ps["calibration"]...)
	if v := ps["vision"]; len(v) > 0 {
		f.Set("vision", v...)
	}
	if a := ps["audio"]; len(a) > 0 {
		f.Set("audio", a...)
	}
	return f, nil
}

func loadDecide(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
	if dir := f.Path("export"); dir != "" {
		m, err := systemone.LoadAMD(ctx, systemone.AMDConfig{Dir: dir, Name: f.Values["name"], Log: e.Log})
		if err != nil {
			return nil, err
		}
		return bridge.Wrap["decide"](&decider{m: m, model: e.Model}), nil
	}
	m, err := systemone.Load(ctx, systemone.Config{Packages: f.Paths["packages"], ModelConfig: f.Path("config"), Weights: f.Path("weights"),
		Tokenizer: f.Path("tokenizer"), Calibration: f.Path("calibration"), Vision: f.Paths["vision"], Audio: f.Paths["audio"],
		Name: f.Values["name"], CacheDir: e.Cache, CPU: e.NPU.CPU(e.Def), Backend: e.NPU.Device, Log: e.Log})
	if err != nil {
		return nil, err
	}
	return bridge.Wrap["decide"](&decider{m: m, model: e.Model, cpu: e.NPU.CPU(e.Def)}), nil
}

// ints reads a comma separated list of positive numbers; def for "".
func ints(s string, def []int) ([]int, error) {
	if strings.TrimSpace(s) == "" {
		return def, nil
	}
	var out []int
	for _, f := range strings.Split(s, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(f))
		if err != nil || n <= 0 {
			return nil, fmt.Errorf("text: %q is not a positive number", f)
		}
		out = append(out, n)
	}
	return out, nil
}
