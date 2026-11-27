// Package engine runs ai/text's models in this process: the chat, translation, embedding and
// decision models of internal/chat and internal/systemone. ai/engine imports it; ai/text alone is
// their client through hwnpud (docs/design/hwnpud.md, "Clients without the engine").
package engine

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu/ai/text"
	"github.com/shibukawa/hwnpu/ai/text/internal/bridge"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm/experts"
	"github.com/shibukawa/hwnpu/internal/mem"
	"github.com/shibukawa/hwnpu/internal/systemone"
)


func init() {
	for name, r := range bridge.Roles {
		switch name {
		case "chat", "translate":
			r.Prepare = prepareEngine
		case "embed":
			r.Prepare = prepareEmbed
		case "decide":
			r.Prepare = prepareDecide
		}
	}
	bridge.Roles["chat"].Load = func(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
		eng, err := loadEngine(ctx, f, e, "chat")
		if err != nil {
			return nil, err
		}
		return bridge.Wrap["chat"](&chatter{e: eng, model: e.Model, def: e.Def}), nil
	}
	bridge.Roles["translate"].Load = func(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
		eng, err := loadEngine(ctx, f, e, "translate")
		if err != nil {
			return nil, err
		}
		return bridge.Wrap["translate"](&translator{e: eng, model: e.Model}), nil
	}
	bridge.Roles["embed"].Load = loadEmbed
	bridge.Roles["decide"].Load = loadDecide
}

// chatter is a chat model of the engine (text.Chatter's). It runs one request at a time; others
// wait.
type chatter struct {
	mu    sync.Mutex
	e     *chat.Engine
	model string
	def   chat.Model
}

func (c *chatter) Model() string      { return c.model }
func (c *chatter) Name() string       { return c.e.Name }
func (c *chatter) Features() []string { return slices.Clone(c.def.Features) }
func (c *chatter) Thinks() bool       { return !c.e.NoThink }
func (c *chatter) Translates() bool   { return c.e.Template == chat.Plamo3 }
func (c *chatter) Context() int       { return c.e.Context() }
func (c *chatter) Images() bool       { return c.e.Images() }
func (c *chatter) Audio() bool        { return c.e.Audio() }
func (c *chatter) OnNPU() bool        { return c.e.OnNPU() }
func (c *chatter) Device() string     { return c.e.DeviceName() }

func (c *chatter) Reply(ctx context.Context, msgs []text.Message, o text.ChatOptions, fn func(text.Delta) bool) (text.Answer, error) {
	in, err := chatMessages(msgs)
	if err != nil {
		return text.Answer{}, err
	}
	if o.Format != nil {
		in = instruct(o.Format, in)
	}
	tools, err := chatTools(o.Tools)
	if err != nil {
		return text.Answer{}, err
	}
	think := c.Thinks()
	if o.Think != nil {
		think = *o.Think
	}
	opts := chat.Options{NoThink: !think, KeepThinking: o.KeepThinking}
	if o.Translation != nil {
		opts.Translation = translation(o.Translation)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	a, err := generate(ctx, c.e, in, tools, opts, request(o.Sampling, text.DefaultSampling, o.MaxTokens, o.Seed), thinkBudget(o.ThinkBudget), o.Stop, fn)
	if err == nil && o.Format != nil {
		var t string
		if t, err = o.Format.Check(a.Text); err == nil {
			a.Text = t
		}
	}
	return a, err
}

func (c *chatter) Count(ctx context.Context, msgs []text.Message, o text.ChatOptions) (int, error) {
	in, err := chatMessages(msgs)
	if err != nil {
		return 0, err
	}
	tools, err := chatTools(o.Tools)
	if err != nil {
		return 0, err
	}
	think := c.Thinks()
	if o.Think != nil {
		think = *o.Think
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.e.Count(ctx, in, tools, chat.Options{NoThink: !think, KeepThinking: o.KeepThinking})
}

func (c *chatter) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.e.Close()
}

// instruct puts the request for JSON of f (text.Format.Instruction) in the system message of msgs.
func instruct(f *text.Format, msgs []chat.Message) []chat.Message {
	say := f.Instruction()
	msgs = slices.Clone(msgs)
	if len(msgs) > 0 && msgs[0].Role == "system" {
		c := msgs[0].Content
		msgs[0].Content = chat.Content{Text: strings.TrimRight(c.Text, "\n") + "\n\n" + say, Media: c.Media}
		return msgs
	}
	return append([]chat.Message{{Role: "system", Content: chat.Content{Text: say}}}, msgs...)
}

// translation is the translation's settings as PLaMo 3's template takes them.
func translation(o *text.TranslateOptions) *chat.Translation {
	tr := &chat.Translation{TargetLang: o.To, SourceLang: o.From, Form: o.Form, Context: o.Context}
	for _, g := range o.Glossary {
		tr.Glossary = append(tr.Glossary, chat.Term{Source: g.Source, Target: g.Target})
	}
	return tr
}

// translator is a translation model of the engine (text.Translator's).
type translator struct {
	mu    sync.Mutex
	e     *chat.Engine
	model string
}

func (t *translator) Model() string { return t.model }
func (t *translator) OnNPU() bool   { return t.e.OnNPU() }

func (t *translator) Reply(ctx context.Context, s string, o text.TranslateOptions, fn func(text.Delta) bool) (text.Translation, error) {
	var msgs []chat.Message
	for _, p := range o.History {
		msgs = append(msgs, chat.Message{Role: "user", Content: chat.Content{Text: p.Source}},
			chat.Message{Role: "assistant", Content: chat.Content{Text: p.Target}})
	}
	msgs = append(msgs, chat.Message{Role: "user", Content: chat.Content{Text: s}})
	t.mu.Lock()
	defer t.mu.Unlock()
	a, err := generate(ctx, t.e, msgs, nil, chat.Options{Translation: translation(&o)}, request(o.Sampling, text.Sampling{}, o.MaxTokens, o.Seed), 0, nil, fn)
	return text.Translation{Text: a.Text, Stats: a.Stats}, err
}

func (t *translator) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.e.Close()
}

// embedder is an embedding model of the engine (text.Embedder's).
type embedder struct {
	e     *chat.Embedder
	model string
}

func (e *embedder) Model() string  { return e.model }
func (e *embedder) Name() string   { return e.e.Name }
func (e *embedder) Dim() int       { return e.e.Dim() }
func (e *embedder) MaxTokens() int { return e.e.MaxTokens() }
func (e *embedder) Close() error   { return e.e.Close() }

func (e *embedder) Embed(ctx context.Context, s string, o text.EmbedOptions) (text.Embedding, error) {
	r, err := e.e.Embed(ctx, s, o.Query, o.Truncate)
	return text.Embedding{Vector: r.Vector, Tokens: r.Tokens, Truncated: r.Truncated, Took: r.Time}, err
}

// decider is a decision model of the engine (text.Decider's).
type decider struct {
	m     *systemone.Model
	model string
	cpu   bool
}

func (d *decider) Model() string  { return d.model }
func (d *decider) Name() string   { return d.m.Name }
func (d *decider) MaxTokens() int { return d.m.MaxTokens() }
func (d *decider) OnCPU() bool    { return d.cpu }
func (d *decider) Close() error   { return d.m.Close() }

func (d *decider) Decide(ctx context.Context, request []byte) (text.Decision, error) {
	t0 := time.Now()
	req, err := systemone.ParseRequest(request)
	if err == nil {
		var res *systemone.Response
		if res, err = d.m.Decide(ctx, req); err == nil {
			b, err := json.Marshal(res)
			return text.Decision{JSON: b, Questions: len(req.Questions), InputTokens: res.InputTokens, Took: time.Since(t0)}, err
		}
	}
	var bad *systemone.BadRequest
	var unsup *systemone.Unsupported
	switch {
	case errors.As(err, &bad):
		return text.Decision{}, kindError{err, text.ErrBadRequest}
	case errors.As(err, &unsup):
		return text.Decision{}, kindError{err, text.ErrUnsupported}
	}
	return text.Decision{}, err
}

// kindError is err, which also is kind (errors.Is), with err's message.
type kindError struct{ err, kind error }

func (e kindError) Error() string   { return e.err.Error() }
func (e kindError) Unwrap() []error { return []error{e.kind, e.err} }



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


// noAMDForm is why a decision model without an export for AMD's NPU (its recipe, model-recipes.yaml)
// does not run there: hwnpu has no form of it, said at once instead of looking for a device that
// runs the converted programs ("no NPU device found" on a machine that has one). Programs given
// (packages) and the CPU interpreter still run.
func noAMDForm(e airole.PrepareEnv) error {
	if !e.NPU.AMD() || len(e.Model.Programs["packages"]) > 0 || e.NPU.CPU(e.Model) {
		return nil
	}
	return fmt.Errorf("%s has no form for AMD's NPU: it has no recipe of iron_export_lm.py in model-recipes.yaml (device: cpu in a definition of your own runs it on the CPU interpreter, slowly)", e.Model.Name)
}
