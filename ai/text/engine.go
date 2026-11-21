package text

import (
	"cmp"
	"context"
	"strconv"

	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
)

// prepareEngine makes a chat model ready as hwnpu-chat does on its first run, with the same
// programs for the same NPU, so that what one prepared the other uses.
func prepareEngine(e airole.PrepareEnv) (f airole.Files, err error) {
	m := e.Model
	if e.NPU.AMD() {
		// AMD's NPU runs an export of IRON instead of the programs of the other NPUs: that of
		// the model as it was picked (the export of the memory it runs as)
		iron, err := chat.AMDExportOf(e.Ctx, m, e.Log)
		if err != nil {
			return f, err
		}
		f.Set("iron", iron)
		return f, nil
	}
	// hwnpu-chat's defaults with the model's definition for the memory it runs as (models.yaml's
	// context, paged, prefill, format and tier, Model.On), so that the programs are the ones it
	// converts
	tiers, err := chat.Tiers(cmp.Or(m.Run.Tier, chat.DefaultTiers))
	if err != nil {
		return f, err
	}
	c, err := chat.Prepare(e.Ctx, chat.Setup{Dir: e.Dir, Repo: m.Source(), Arch: m.Arch, Name: m.Name,
		Context: m.Run.Context, Paged: m.Run.Paged, Prefill: m.Run.Prefill, Format: m.Run.Format, CodesHash: m.Run.Codes,
		Draft: 8, MTP: chat.MTPAuto, Tiers: tiers, Backend: e.NPU.Device, Log: e.Log})
	if err != nil {
		return f, err
	}
	f.Set("packages", c.Packages...)
	for i, t := range c.Tiers {
		f.Set("tier"+strconv.Itoa(i), t...)
	}
	f.SetValue("tiers", strconv.Itoa(len(c.Tiers)))
	f.Set("mtp", c.MTP...)
	f.Set("vision", c.Vision...)
	f.Set("audio", c.Audio...)
	f.Set("config", c.ModelConfig)
	f.Set("weights", c.Weights)
	f.Set("tokenizer", c.Tokenizer)
	f.SetValue("name", cmp.Or(c.Name, m.Name))
	f.SetValue("backend", c.Backend)
	return f, nil
}

// loadEngine loads what prepareEngine made.
func loadEngine(ctx context.Context, f airole.Files, e airole.LoadEnv) (*chat.Engine, error) {
	c := chat.Config{IRON: f.Path("iron"), Packages: f.Paths["packages"], MTP: f.Paths["mtp"],
		Vision: f.Paths["vision"], Audio: f.Paths["audio"],
		ModelConfig: f.Path("config"), Weights: f.Path("weights"), Tokenizer: f.Path("tokenizer"),
		Name: f.Values["name"], Backend: f.Values["backend"], CacheDir: e.Cache, Log: e.Log}
	n, _ := strconv.Atoi(f.Values["tiers"])
	for i := range n {
		c.Tiers = append(c.Tiers, f.Paths["tier"+strconv.Itoa(i)])
	}
	eng, err := chat.Load(ctx, c)
	if err != nil {
		return nil, err
	}
	if m, ok := chat.Lookup(e.Model); ok {
		eng.NoThink = m.NoThink
	}
	return eng, nil
}
