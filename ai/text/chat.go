package text

import (
	"cmp"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/flow"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
)

// Chat is the use of a chat model for an app: a model, a family or a use of the definitions
// ("chat", "lfm3", "Qwen3.5-2B").
func Chat(name string) ai.Use[*Chatter] {
	return ai.Define[*Chatter](chatRole, name)
}

var chatRole = &airole.Role{
	Name:    "chat",
	Kinds:   []chat.Kind{chat.KindChat},
	Prepare: prepareEngine,
	// 1: the attention programs of a model split at its attention are written again with the
	// model's scale of the values' weights (paged.Shape.OutScale: Qwen3's), which Prepare does;
	// 2: Qwen3.5's too
	Revision: 2,
	Load: func(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
		eng, err := loadEngine(ctx, f, e, "chat")
		if err != nil {
			return nil, err
		}
		return &Chatter{e: eng, model: e.Model, def: e.Def}, nil
	},
}

// Message is a turn of a conversation.
type Message struct {
	// Role is system, user, assistant or tool (the result of a tool call).
	Role string
	Text string
	// Media are images and audio clips sent with Text, before it (a model with the vision or
	// the audio feature reads them).
	Media []Media
	// Parts are the message's texts, images and clips in their order, for a message that mixes
	// them (the parts of the OpenAI API's content): when there are parts, Text and Media are
	// not read.
	Parts []Part
	// Reasoning is an assistant's thinking before its answer ("" for none), which the next
	// prompt keeps for the last turn or with ChatOptions.KeepThinking.
	Reasoning string
	// ToolCalls are the calls of an assistant's turn.
	ToolCalls []ToolCall
	// ToolCallID and Name are a tool message's: the call it answers and the tool's name.
	ToolCallID, Name string
}

// Media is an image or an audio clip of a message.
type Media struct {
	// Audio makes it a clip, a WAV file; else it is an image file (JPEG, PNG, ...).
	Audio bool
	Data  []byte
	// URL is the media as a data URL (data:image/png;base64,...), or a clip as base64 WAV, in
	// place of Data.
	URL string
}

// Part is a text, an image or a clip of a message of parts.
type Part struct {
	Text  string
	Media *Media
}

// Tool is a function the model may call. Parameters is its arguments' JSON Schema.
type Tool struct {
	Name, Description string
	Parameters        json.RawMessage
	raw               []byte
}

// ParseTool reads a tool as the OpenAI API sends it ({"type": "function", "function": {...}} or
// the function object alone), kept as it was written: the prompt shows it so.
func ParseTool(raw []byte) (Tool, error) {
	var t chat.Tool
	if err := json.Unmarshal(raw, &t); err != nil {
		return Tool{}, err
	}
	var f struct {
		Function *struct {
			Description string          `json:"description"`
			Parameters  json.RawMessage `json:"parameters"`
		} `json:"function"`
		Description string          `json:"description"`
		Parameters  json.RawMessage `json:"parameters"`
	}
	json.Unmarshal(raw, &f)
	out := Tool{Name: t.Name(), Description: f.Description, Parameters: f.Parameters, raw: slices.Clone(raw)}
	if f.Function != nil {
		out.Description, out.Parameters = f.Function.Description, f.Function.Parameters
	}
	return out, nil
}

// MarshalJSON writes the tool as the OpenAI API does.
func (t Tool) MarshalJSON() ([]byte, error) {
	if t.raw != nil {
		return t.raw, nil
	}
	fn := map[string]any{"name": t.Name}
	if t.Description != "" {
		fn["description"] = t.Description
	}
	if len(t.Parameters) > 0 {
		fn["parameters"] = t.Parameters
	}
	return json.Marshal(map[string]any{"type": "function", "function": fn})
}

// ToolCall is a call of a tool by the model. Arguments is a JSON object.
type ToolCall struct {
	ID, Name, Arguments string
}

// Sampling is how the next token is drawn: the zero value picks the most likely one.
type Sampling struct {
	// Temperature 0 picks the most likely token.
	Temperature float64
	// TopK and TopP keep the K most likely tokens, then the fewest of them whose probabilities
	// add up to TopP; 0 turns either off.
	TopK int
	TopP float64
	// Repetition divides the logits of the tokens already in the answer (Hugging Face's
	// repetition_penalty); 0 or 1 turns it off.
	Repetition float64
}

// DefaultSampling is a chat's sampling when ChatOptions does not give one: hwnpu-chat's.
var DefaultSampling = Sampling{Temperature: 0.1, TopK: 50, Repetition: 1.05}

// Unlimited as ChatOptions.MaxTokens (or TranslateOptions') lets the answer go on until the
// model ends it or its cache is full.
const Unlimited = -1

// ChatOptions are how a chat model answers. The zero value is hwnpu-chat's defaults.
type ChatOptions struct {
	// MaxTokens is the most tokens of the answer, its thinking included (0: 1024; Unlimited:
	// until the cache is full).
	MaxTokens int
	// Think lets the model think before it answers (nil: unless the model's definition turns it
	// off, Chatter.Thinks), up to ThinkBudget tokens (0: 1024; below 0: no limit), after which it
	// is made to answer. KeepThinking keeps the thinking of the assistant's earlier turns in the
	// prompt.
	Think        *bool
	ThinkBudget  int
	KeepThinking bool
	// Sampling is how the tokens are drawn (nil: DefaultSampling), with Seed (0: a new one for
	// each answer, which Stats.Seed tells).
	Sampling *Sampling
	Seed     uint64
	// Stop are texts that end the answer where the model writes them; the answer leaves them out.
	Stop []string
	// Tools are the functions the model may call (a model with the tools feature): its calls are
	// in Answer.ToolCalls.
	Tools []Tool
	// Format asks for the answer as JSON (nil: text).
	Format *Format
	// Translation is what a translation model (PLaMo 3 Translate) is asked besides the texts (its
	// History and MaxTokens are not read: the conversation is the history); other models ignore
	// it. Translator does the same with a history of its own.
	Translation *TranslateOptions
}

// Answer is a chat model's answer.
type Answer struct {
	// Text is the answer (without the thinking and the tool calls), Reasoning the thinking
	// before it.
	Text, Reasoning string
	ToolCalls       []ToolCall
	// Finish is stop (the model ended the answer, or a stop text, or the caller), length (the
	// most tokens, or the cache full) or tool_calls.
	Finish string
	Stats  Stats
	// Prompt is the conversation as the model read it, in its chat format, and Raw the text it
	// generated with the format's tags: what to look at when an answer is wrong.
	Prompt, Raw string
}

// Stats are what answering took.
type Stats struct {
	// Prompt is the tokens of the prompt, Reused those of them the cache held from before (of
	// which Restored came from the prompts kept on disk), Generated the tokens of the answer.
	Prompt, Reused, Restored, Generated int
	// Prefill is the time reading the prompt took, Decode the time generating.
	Prefill, Decode time.Duration
	// Drafted and Accepted count the tokens guessed ahead (speculative decoding) and those the
	// model confirmed; Calls the runs of the model while generating.
	Drafted, Accepted, Calls int
	// Cache is the length of the cache the answer ended on.
	Cache int
	// Seed is the seed the tokens were drawn with.
	Seed uint64
	// ThinkCut says that the thinking reached its budget and was ended; ThinkLooped, that it was
	// ended going round in circles; Looped, that the answer stopped repeating itself.
	ThinkCut, ThinkLooped, Looped bool
}

// TokensPerSecond is the speed of generating.
func (s Stats) TokensPerSecond() float64 {
	return float64(s.Generated) / max(s.Decode.Seconds(), 1e-9)
}

// Delta is what an answer adds as it is generated: to its thinking and to its text.
type Delta struct {
	Reasoning, Text string
}

// Chatter answers conversations with a chat model. It runs one request at a time; others wait.
type Chatter struct {
	mu    sync.Mutex
	e     *chat.Engine
	model string
	def   chat.Model
}

// Model is the name of the model in the definitions.
func (c *Chatter) Model() string { return c.model }

// Name is the model's name as it calls itself (the API's, the logs'): its definition's, or the
// export's.
func (c *Chatter) Name() string { return c.e.Name }

// Features are the model's features in the definitions (tools, vision, audio, translation, ...).
func (c *Chatter) Features() []string { return slices.Clone(c.def.Features) }

// Thinks reports whether the model thinks before it answers when ChatOptions does not say.
func (c *Chatter) Thinks() bool { return !c.e.NoThink }

// Translates reports whether the model is a translation model (ChatOptions.Translation).
func (c *Chatter) Translates() bool { return c.e.Template == chat.Plamo3 }

// Context is the most tokens a prompt and its answer can have together.
func (c *Chatter) Context() int { return c.e.Context() }

// Images and Audio report whether the model reads images and audio clips.
func (c *Chatter) Images() bool { return c.e.Images() }
func (c *Chatter) Audio() bool  { return c.e.Audio() }

// OnNPU reports whether the programs were verified to run on the NPU.
func (c *Chatter) OnNPU() bool { return c.e.OnNPU() }

// Device is the NPU the model runs on, as the logs call it.
func (c *Chatter) Device() string { return c.e.DeviceName() }

// Complete answers a conversation: its text, trimmed.
func (c *Chatter) Complete(ctx context.Context, msgs []Message, o ChatOptions) (string, error) {
	a, err := c.Reply(ctx, msgs, o, nil)
	return strings.TrimSpace(a.Text), err
}

// CompleteStream answers a conversation as a stream: the answer so far after each token, each
// chunk replacing the one before it, and the whole answer as the final chunk.
func (c *Chatter) CompleteStream(ctx context.Context, msgs []Message, o ChatOptions) flow.Stream[flow.Text] {
	return func(yield func(flow.Chunk[flow.Text], error) bool) {
		sofar, stopped := "", false
		a, err := c.Reply(ctx, msgs, o, func(d Delta) bool {
			if d.Text == "" {
				return true
			}
			sofar += d.Text
			stopped = !yield(flow.Chunk[flow.Text]{Data: flow.Text{Text: sofar}}, nil)
			return !stopped
		})
		if stopped {
			return
		}
		if err != nil {
			yield(flow.Chunk[flow.Text]{}, err)
			return
		}
		yield(flow.Chunk[flow.Text]{Data: flow.Text{Text: strings.TrimSpace(a.Text)}, Final: true}, nil)
	}
}

// Reply answers a conversation, telling fn (nil for none) what the answer adds after each token,
// its thinking and its text; fn returns false to stop the answer there. When ctx is cancelled the
// answer so far is returned with ctx's error. An answer not of o.Format is returned with a
// *FormatError.
func (c *Chatter) Reply(ctx context.Context, msgs []Message, o ChatOptions, fn func(Delta) bool) (Answer, error) {
	if len(msgs) == 0 {
		return Answer{}, errors.New("text: no messages to answer")
	}
	in, err := chatMessages(msgs)
	if err != nil {
		return Answer{}, err
	}
	if o.Format != nil {
		in = o.Format.instruct(in)
	}
	tools, err := chatTools(o.Tools)
	if err != nil {
		return Answer{}, err
	}
	think := c.Thinks()
	if o.Think != nil {
		think = *o.Think
	}
	opts := chat.Options{NoThink: !think, KeepThinking: o.KeepThinking}
	if o.Translation != nil {
		opts.Translation = o.Translation.chat()
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	a, err := generate(ctx, c.e, in, tools, opts, request(o.Sampling, DefaultSampling, o.MaxTokens, o.Seed), thinkBudget(o.ThinkBudget), o.Stop, fn)
	if err == nil && o.Format != nil {
		var text string
		if text, err = o.Format.check(a.Text); err == nil {
			a.Text = text
		}
	}
	return a, err
}

// Count is the tokens of the prompt the conversation is, with o's tools and thinking.
func (c *Chatter) Count(ctx context.Context, msgs []Message, o ChatOptions) (int, error) {
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

// Close unloads the model. An app leaves it to the Runtime (its Close, or the use's Release).
func (c *Chatter) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.e.Close()
}

// ErrMedia is returned for an image or a clip the model cannot read (not one, or a model
// without the feature).
var ErrMedia = chat.ErrImage

// ErrContext is returned for a prompt that does not fit in the model's context.
var ErrContext = chat.ErrContext

// request is the generation of the options given: sampling s (nil: def), maxTokens (0: 1024,
// Unlimited: until the cache is full) and seed.
func request(s *Sampling, def Sampling, maxTokens int, seed uint64) chat.Request {
	sm := cmp.Or(s, &def)
	r := chat.Request{Sampling: chat.Sampling{Temperature: sm.Temperature, TopK: sm.TopK, TopP: sm.TopP, Repetition: sm.Repetition}, Seed: seed}
	switch {
	case maxTokens == 0:
		r.MaxTokens = 1024
	case maxTokens > 0:
		r.MaxTokens = maxTokens
	}
	return r
}

// thinkBudget is ChatOptions.ThinkBudget as the engine takes it (0: none).
func thinkBudget(b int) int {
	switch {
	case b == 0:
		return 1024
	case b < 0:
		return 0
	}
	return b
}

// generate answers msgs on e as hwnpu-chat and its server do: it renders the prompt, reads it,
// generates, tells fn what each token adds (holding back what may become a stop text) and takes
// the answer apart.
func generate(ctx context.Context, e *chat.Engine, msgs []chat.Message, tools []chat.Tool, opts chat.Options, req chat.Request, budget int, stops []string, fn func(Delta) bool) (Answer, error) {
	var a Answer
	prompt, thinking, err := e.Template.Render(msgs, tools, opts)
	if err != nil {
		return a, err
	}
	a.Prompt = prompt
	var media []chat.Media
	for _, m := range msgs {
		media = append(media, m.Content.Media...)
	}
	ids, err := e.Tokens(ctx, prompt, media)
	if err != nil {
		return a, err
	}
	req.Prompt, req.Think = ids, e.ThinkLimit(budget, thinking)
	hold := 0 // the text held back while it may become a stop text
	for _, st := range stops {
		hold = max(hold, len(st)-1)
	}
	cut := func(c string) (string, bool) {
		for _, st := range stops {
			if i := strings.Index(c, st); st != "" && i >= 0 {
				return c[:i], true
			}
		}
		return c, false
	}
	var sent chat.Stream
	var gen []int32
	called := false // fn said to stop
	st, err := e.Generate(ctx, req, func(sofar []int32) (bool, error) {
		gen = sofar
		p := e.Template.Parse(e.Tok.Decode(sofar, false), thinking, tools, false)
		c, stopped := cut(p.Content)
		if fn == nil {
			return stopped, nil
		}
		if !stopped && !p.Calling {
			c = validPrefix(c[:max(0, len(c)-hold)])
		}
		if dr, dc := sent.Next(p.Reasoning, c); dr != "" || dc != "" {
			if !fn(Delta{Reasoning: dr, Text: dc}) {
				called = true
				return true, nil
			}
		}
		return stopped, nil
	})
	a.Stats = stats(st)
	a.Raw = e.Tok.Decode(gen, false)
	p := e.Template.Parse(a.Raw, thinking, tools, true)
	a.Text, _ = cut(p.Content)
	a.Reasoning = p.Reasoning
	for _, tc := range p.ToolCalls {
		a.ToolCalls = append(a.ToolCalls, ToolCall{ID: tc.ID, Name: tc.Function.Name, Arguments: tc.Function.Arguments})
	}
	a.Finish = st.Finish
	if len(a.ToolCalls) > 0 {
		a.Finish = "tool_calls"
	}
	if err != nil {
		return a, err
	}
	if fn != nil && !called {
		// what the last parse adds (a tail held back while it might have begun a call)
		if dr, dc := sent.Next(a.Reasoning, a.Text); dr != "" || dc != "" {
			fn(Delta{Reasoning: dr, Text: dc})
		}
	}
	return a, nil
}

func stats(s chat.Stats) Stats {
	return Stats{Prompt: s.Prompt, Reused: s.Reused, Restored: s.Restored, Generated: s.Generated, Prefill: s.Prefill, Decode: s.Decode,
		Drafted: s.Drafted, Accepted: s.Accepted, Calls: s.Calls, Cache: s.Cache, Seed: s.Seed, ThinkCut: s.ThinkCut, ThinkLooped: s.ThinkLooped, Looped: s.Looped}
}

// validPrefix is s without an incomplete UTF-8 character at its end.
func validPrefix(s string) string {
	for i := 0; i < utf8.UTFMax && len(s) > 0 && !utf8.ValidString(s); i++ {
		s = s[:len(s)-1]
	}
	return s
}

// chatMessages are msgs as the engine takes them: each medium a MediaMarker in the text.
func chatMessages(msgs []Message) ([]chat.Message, error) {
	out := make([]chat.Message, len(msgs))
	for i, m := range msgs {
		cm := chat.Message{Role: m.Role, ToolCallID: m.ToolCallID, Name: m.Name}
		if m.Reasoning != "" {
			r := m.Reasoning
			cm.Reasoning = &r
		}
		for _, tc := range m.ToolCalls {
			var c chat.ToolCall
			c.ID, c.Type, c.Function.Name, c.Function.Arguments = tc.ID, "function", tc.Name, tc.Arguments
			cm.ToolCalls = append(cm.ToolCalls, c)
		}
		if len(m.Parts) > 0 {
			var parts []string
			var media []chat.Media
			for _, p := range m.Parts {
				if p.Media == nil {
					parts = append(parts, p.Text)
					continue
				}
				md, err := chatMedia(*p.Media)
				if err != nil {
					return nil, err
				}
				parts, media = append(parts, chat.MediaMarker), append(media, md)
			}
			cm.Content = chat.PartsContent(parts, media)
		} else {
			cm.Content = chat.Content{Text: strings.Repeat(chat.MediaMarker, len(m.Media)) + m.Text}
			for _, md := range m.Media {
				c, err := chatMedia(md)
				if err != nil {
					return nil, err
				}
				cm.Content.Media = append(cm.Content.Media, c)
			}
		}
		out[i] = cm
	}
	return out, nil
}

// chatMedia is m as the engine reads it: an image's data URL, or a clip's base64 WAV.
func chatMedia(m Media) (chat.Media, error) {
	if m.URL != "" {
		return chat.Media{Audio: m.Audio, Data: m.URL}, nil
	}
	if m.Audio {
		return chat.Media{Audio: true, Data: base64.StdEncoding.EncodeToString(m.Data)}, nil
	}
	mime := http.DetectContentType(m.Data)
	if !strings.HasPrefix(mime, "image/") {
		return chat.Media{}, fmt.Errorf("%w: %s, not an image", ErrMedia, mime)
	}
	return chat.Media{Data: "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(m.Data)}, nil
}

// chatTools are tools as the engine takes them.
func chatTools(tools []Tool) ([]chat.Tool, error) {
	var out []chat.Tool
	for _, t := range tools {
		raw, err := t.MarshalJSON()
		if err != nil {
			return nil, err
		}
		var ct chat.Tool
		if err := json.Unmarshal(raw, &ct); err != nil {
			return nil, fmt.Errorf("text: the tool %s: %w", t.Name, err)
		}
		out = append(out, ct)
	}
	return out, nil
}

func init() { airole.Register(chatRole) }
