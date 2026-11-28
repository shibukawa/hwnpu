package text

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/flow"
	"github.com/shibukawa/hwnpu/ai/text/internal/bridge"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/airpc"
	"github.com/shibukawa/hwnpu/internal/catalog"
	"github.com/shibukawa/hwnpu/internal/chatwire"
)

// Chat is the use of a chat model for an app: a model, a family or a use of the definitions
// ("chat", "lfm3", "Qwen3.5-2B").
func Chat(name string) ai.Use[*Chatter] {
	return ai.Define[*Chatter](chatRole, name)
}

// chatRole's Prepare and Load are the engine's (ai/text/engine).
var chatRole = &airole.Role{
	LoadSettings: loadSettings,
	Name:         "chat",
	Remote:       func(h *airpc.Handle) io.Closer { return &Chatter{p: airpc.RemoteProxy[chatterImpl](h)} },
	Kinds:        []catalog.Kind{catalog.KindChat},
	// 1: the attention programs of a model split at its attention are written again with the
	// model's scale of the values' weights (paged.Shape.OutScale: Qwen3's), which Prepare does;
	// 2: Qwen3.5's too; 3: the tokenizer as its binary file (chat.PrepareTokenizer): a store
	// prepared before named the checkpoint's tokenizer.json, which a checkpoint fetched since is
	// without
	Revision: 3,
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
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil || obj == nil {
		return Tool{}, fmt.Errorf("chat: a tool is not an object")
	}
	type function struct {
		Name        string          `json:"name"`
		Description string          `json:"description"`
		Parameters  json.RawMessage `json:"parameters"`
	}
	var f struct {
		function
		Function *function `json:"function"`
	}
	json.Unmarshal(raw, &f)
	out := Tool{Name: f.Name, Description: f.Description, Parameters: f.Parameters, raw: slices.Clone(raw)}
	if f.Function != nil {
		out.Name, out.Description, out.Parameters = f.Function.Name, f.Function.Description, f.Function.Parameters
	}
	var name struct {
		Name     json.RawMessage `json:"name"`
		Function *struct {
			Name json.RawMessage `json:"name"`
		} `json:"function"`
	}
	json.Unmarshal(raw, &name)
	n := name.Name
	if name.Function != nil {
		n = name.Function.Name
	}
	if len(n) == 0 || n[0] != '"' {
		return Tool{}, fmt.Errorf("chat: a tool has no name")
	}
	return out, nil
}

// loadSettings are the settings the engine's text models read (ai.LoadSettings).
var loadSettings = []string{"idle", "prompt_cache", "experts_key", "sparse", "reserve_moe_memory"}

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

// GobEncode sends the tool to hwnpud as it was written (ParseTool's), not as MarshalJSON would
// write it again.
func (t Tool) GobEncode() ([]byte, error) { return t.MarshalJSON() }

// GobDecode reads what GobEncode sent.
func (t *Tool) GobDecode(raw []byte) error {
	out, err := ParseTool(raw)
	if err != nil {
		return err
	}
	*t = out
	return nil
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
	p airpc.Proxy[chatterImpl]
}

// chatterImpl is a chat model of the engine in this process: Chatter's methods but those made
// of them.
type chatterImpl interface {
	Model() string
	Name() string
	Features() []string
	Thinks() bool
	Translates() bool
	Context() int
	Images() bool
	Audio() bool
	OnNPU() bool
	Device() string
	Reply(ctx context.Context, msgs []Message, o ChatOptions, fn func(Delta) bool) (Answer, error)
	Count(ctx context.Context, msgs []Message, o ChatOptions) (int, error)
	Close() error
}

// Model is the name of the model in the definitions.
func (c *Chatter) Model() string { return airpc.ProxyValue(c.p, "Model", chatterImpl.Model) }

// Name is the model's name as it calls itself (the API's, the logs'): its definition's, or the
// export's.
func (c *Chatter) Name() string { return airpc.ProxyValue(c.p, "Name", chatterImpl.Name) }

// Features are the model's features in the definitions (tools, vision, audio, translation, ...).
func (c *Chatter) Features() []string { return airpc.ProxyValue(c.p, "Features", chatterImpl.Features) }

// Thinks reports whether the model thinks before it answers when ChatOptions does not say.
func (c *Chatter) Thinks() bool { return airpc.ProxyValue(c.p, "Thinks", chatterImpl.Thinks) }

// Translates reports whether the model is a translation model (ChatOptions.Translation).
func (c *Chatter) Translates() bool {
	return airpc.ProxyValue(c.p, "Translates", chatterImpl.Translates)
}

// Context is the most tokens a prompt and its answer can have together.
func (c *Chatter) Context() int { return airpc.ProxyValue(c.p, "Context", chatterImpl.Context) }

// Images and Audio report whether the model reads images and audio clips.
func (c *Chatter) Images() bool { return airpc.ProxyValue(c.p, "Images", chatterImpl.Images) }

func (c *Chatter) Audio() bool { return airpc.ProxyValue(c.p, "Audio", chatterImpl.Audio) }

// OnNPU reports whether the programs were verified to run on the NPU.
func (c *Chatter) OnNPU() bool { return airpc.ProxyValue(c.p, "OnNPU", chatterImpl.OnNPU) }

// Device is the NPU the model runs on, as the logs call it.
func (c *Chatter) Device() string { return airpc.ProxyValue(c.p, "Device", chatterImpl.Device) }

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
	return airpc.ProxyCall1(c.p, ctx, "Reply", func(l chatterImpl) (Answer, error) {
		return l.Reply(ctx, msgs, o, fn)
	}, msgs, o, fn)
}

// Count is the tokens of the prompt the conversation is, with o's tools and thinking.
func (c *Chatter) Count(ctx context.Context, msgs []Message, o ChatOptions) (int, error) {
	return airpc.ProxyCall1(c.p, ctx, "Count", func(l chatterImpl) (int, error) {
		return l.Count(ctx, msgs, o)
	}, msgs, o)
}

// Close unloads the model. An app leaves it to the Runtime (its Close, or the use's Release).
func (c *Chatter) Close() error { return c.p.Close() }

// ErrMedia is returned for an image or a clip the model cannot read (not one, or a model
// without the feature).
var ErrMedia = chatwire.ErrImage

// ErrContext is returned for a prompt that does not fit in the model's context.
var ErrContext = chatwire.ErrContext

func init() {
	airole.Register(chatRole)
	bridge.Roles[chatRole.Name] = chatRole
	bridge.Wrap[chatRole.Name] = func(l any) io.Closer { return &Chatter{p: airpc.LocalProxy(l.(chatterImpl))} }
	// what the callers of a model in hwnpud check for (errors.Is, errors.As)
	airpc.RegisterError("text.ErrMedia", ErrMedia)
	airpc.RegisterError("text.ErrContext", ErrContext)
	airpc.RegisterErrorType[*FormatError]("text.FormatError")
}
