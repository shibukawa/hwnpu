package text

import (
	"cmp"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm/experts"
	"github.com/shibukawa/hwnpu/internal/mem"
)

// The settings and programs a chat model's definition may give (ai.Definition), besides its
// Format, Context, Paged, Prefill and Tier:
//
//	settings: draft       tokens per call of the program that checks drafts (8; 0 or 1: none)
//	          mtp         auto (drafts also from the checkpoint's multi-token prediction layer) or off
//	          shortest    with paged, attention programs over shorter chunks from this many tokens
//	                      (0: the backend's, -1: none)
//	          codes       a directory of lookup tables chosen against the activations, for lutN weights
//	          format_for  SUBSTRING=FORMAT,...: another weight format for the tensors so named
//	programs: iron        on AMD's NPU, an export of tools/amd/iron_export_lm.py (its directory)
//	          packages    the programs converted by hwnpu-dev convert, with the same -cache and
//	                      different -seq; with config and weights (default: the checkpoint's)
//	          tokenizer   tokenizer.json (default: the checkpoint's, or the export's)
//	          vision      the vision encoder, a package per image size (default: the one converted)
//	          audio       the audio encoder, a package per clip length (default: the one converted)
//
// And the settings of how it runs, which change nothing of what is prepared (ai.LoadSettings
// for the role, "chat" or "translate"):
//
//	idle         AMD's NPU: give the NPU's memory back after this long without a request (5m;
//	             0: never); a mixture of experts on the CPU keeps only its core wired after it
//	prompt_cache the directory prompts read are kept in (default: prompts in the cache; off: none)
//	experts_key  the name of the app's counts of the experts a mixture of experts runs (default:
//	             the role's name; off: none)
//	sparse       off reads the whole cache on every call of a model split at its attention

// prepareEngine makes a chat model ready as hwnpu-chat does on its first run, with the same
// programs for the same NPU, so that what one prepared the other uses.
func prepareEngine(e airole.PrepareEnv) (f airole.Files, err error) {
	m, ps, set := e.Model, e.Model.Programs, e.Model.Settings
	f.Set("tokenizer", ps["tokenizer"]...)
	if dir := first(ps["iron"]); dir != "" {
		if _, err := os.Stat(filepath.Join(dir, "manifest.json")); err != nil {
			return f, fmt.Errorf("%s is not an export of tools/amd/iron_export_lm.py (no manifest.json)", dir)
		}
		f.Set("iron", dir)
		return f, nil
	}
	if pk := ps["packages"]; len(pk) > 0 {
		// converted by hand: used as they are, with the checkpoint's config.json and weights
		f.Set("packages", pk...)
		f.Set("config", cmp.Or(first(ps["config"]), filepath.Join(chat.RepoDir(m.Source()), "config.json")))
		f.Set("weights", cmp.Or(first(ps["weights"]), chat.WeightsIn(m)))
		f.Set("vision", ps["vision"]...)
		f.Set("audio", ps["audio"]...)
		f.SetValue("name", e.Base.Name)
		f.SetValue("backend", e.NPU.Device)
		return f, nil
	}
	if e.NPU.AMD() {
		// AMD's NPU runs an export of IRON instead of the programs of the other NPUs: that of
		// the model it was made from, found by its name (the export of the memory it runs as)
		iron, err := chat.AMDExportOf(e.Ctx, e.Base, e.Log)
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
		return f, fmt.Errorf("tier %q: %w", m.Run.Tier, err)
	}
	draft, err := intSetting(set, "draft", 8)
	if err != nil {
		return f, err
	}
	shortest, err := intSetting(set, "shortest", 0)
	if err != nil {
		return f, err
	}
	c, err := chat.Prepare(e.Ctx, chat.Setup{Dir: e.Dir, Repo: m.Source(), Arch: m.Arch, Name: e.Base.Name,
		Context: m.Run.Context, Paged: m.Run.Paged, Shortest: shortest, Prefill: m.Run.Prefill, Format: m.Run.Format,
		Codes: set["codes"], CodesHash: m.Run.Codes, FormatFor: split(set["format_for"]),
		Draft: draft, MTP: cmp.Or(set["mtp"], chat.MTPAuto), Tiers: tiers, Backend: e.NPU.Device, Log: e.Log})
	if err != nil {
		return f, err
	}
	f.Set("packages", c.Packages...)
	for i, t := range c.Tiers {
		f.Set("tier"+strconv.Itoa(i), t...)
	}
	f.SetValue("tiers", strconv.Itoa(len(c.Tiers)))
	f.Set("mtp", c.MTP...)
	// the encoders converted by hand, else those Prepare converted (a model with vision or audio)
	f.Set("vision", or(ps["vision"], c.Vision)...)
	f.Set("audio", or(ps["audio"], c.Audio)...)
	f.Set("config", c.ModelConfig)
	f.Set("weights", c.Weights)
	if len(ps["tokenizer"]) == 0 {
		f.Set("tokenizer", c.Tokenizer)
	}
	f.SetValue("name", cmp.Or(c.Name, e.Base.Name))
	f.SetValue("backend", c.Backend)
	if c.Rotated {
		f.SetValue("rotated", "true")
	}
	return f, nil
}

// loadEngine loads what prepareEngine made, run as the app's settings for the role say
// (LoadEnv.Settings).
func loadEngine(ctx context.Context, f airole.Files, e airole.LoadEnv, role string) (*chat.Engine, error) {
	set := e.Settings
	idle := 5 * time.Minute
	if v := set["idle"]; v != "" {
		d, err := time.ParseDuration(v)
		if err != nil {
			return nil, fmt.Errorf("text: idle %q: %w", v, err)
		}
		idle = d
	}
	c := chat.Config{IRON: f.Path("iron"), Packages: f.Paths["packages"], MTP: f.Paths["mtp"],
		Vision: f.Paths["vision"], Audio: f.Paths["audio"], Rotated: f.Values["rotated"] == "true",
		ModelConfig: f.Path("config"), Weights: f.Path("weights"), Tokenizer: f.Path("tokenizer"),
		Name: f.Values["name"], Backend: f.Values["backend"], CacheDir: e.Cache, Log: e.Log,
		PromptDir: chat.PromptStore(set["prompt_cache"], e.Cache), Idle: idle, Dense: set["sparse"] == "off"}
	n, _ := strconv.Atoi(f.Values["tiers"])
	for i := range n {
		c.Tiers = append(c.Tiers, f.Paths["tier"+strconv.Itoa(i)])
	}
	// a mixture of experts' experts that do not all fit: its core first, then the app's counts
	// (a model without experts on the host ignores it)
	c.Experts = expertsPlacement(e.Def, idle, cmp.Or(set["experts_key"], role))
	reserve, err := experts.ParseReserve(set["reserve_moe_memory"], mem.Physical())
	if err != nil {
		return nil, fmt.Errorf("text: reserve_moe_memory: %w", err)
	}
	c.Experts.Reserve = reserve
	eng, err := chat.Load(ctx, c)
	if err != nil {
		return nil, err
	}
	eng.NoThink = e.Def.NoThink
	return eng, nil
}

// expertsPlacement is how a mixture of experts' experts that do not all fit are placed: model
// m's core (models.yaml's experts_core) first, then by the app's counts of key ("off": none).
func expertsPlacement(m chat.Model, idle time.Duration, key string) experts.Placement {
	if idle <= 0 {
		idle = -1 // never
	}
	p := experts.Placement{Core: m.ExpertsCore, Idle: idle, Ceiling: filepath.Join(chat.ModelDir(m), "experts-usage", "ceiling")}
	if key != "off" {
		p.Usage = chat.ExpertsUsage(chat.ModelDir(m), key)
	}
	return p
}

// intSetting is the setting name of set as a number, def when it is not given.
func intSetting(set map[string]string, name string, def int) (int, error) {
	v := strings.TrimSpace(set[name])
	if v == "" {
		return def, nil
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return 0, fmt.Errorf("text: the setting %s %q is not a number", name, v)
	}
	return n, nil
}

// split splits a comma separated list; nil for "".
func split(s string) []string {
	if strings.TrimSpace(s) == "" {
		return nil
	}
	return strings.Split(s, ",")
}

// or is a, else b when a is empty.
func or(a, b []string) []string {
	if len(a) > 0 {
		return a
	}
	return b
}

// first is the first of ps, "" for none.
func first(ps []string) string {
	if len(ps) == 0 {
		return ""
	}
	return ps[0]
}
