package text

import (
	"cmp"
	"context"
	"errors"
	"io"
	"strings"
	"sync"

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
	Load: func(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
		eng, err := loadEngine(ctx, f, e)
		if err != nil {
			return nil, err
		}
		return &Chatter{e: eng, model: e.Model}, nil
	},
}

// Message is a turn of a conversation.
type Message struct {
	// Role is system, user or assistant.
	Role string
	Text string
}

// ChatOptions are how a chat model answers. The zero value is hwnpu-chat's defaults.
type ChatOptions struct {
	// MaxTokens is the most tokens of the answer, its thinking included (0: 1024).
	MaxTokens int
	// Think lets the model think before it answers, up to ThinkBudget tokens (0: 1024). Without
	// it the model answers at once.
	Think       bool
	ThinkBudget int
	// Temperature is the sampling temperature (nil: 0.1; 0 picks the most likely token).
	Temperature *float64
}

// Chatter answers conversations with a chat model. It runs one request at a time; others wait.
type Chatter struct {
	mu    sync.Mutex
	e     *chat.Engine
	model string
}

// Model is the name of the model in the definitions.
func (c *Chatter) Model() string { return c.model }

// OnNPU reports whether the programs were verified to run on the NPU.
func (c *Chatter) OnNPU() bool { return c.e.OnNPU() }

// Complete answers a conversation.
func (c *Chatter) Complete(ctx context.Context, msgs []Message, o ChatOptions) (string, error) {
	text, _, err := flow.Last(c.CompleteStream(ctx, msgs, o))
	return text.Text, err
}

// CompleteStream answers a conversation as a stream: the answer so far after each token, each
// chunk replacing the one before it, and the whole answer as the final chunk.
func (c *Chatter) CompleteStream(ctx context.Context, msgs []Message, o ChatOptions) flow.Stream[flow.Text] {
	return func(yield func(flow.Chunk[flow.Text], error) bool) {
		if len(msgs) == 0 {
			yield(flow.Chunk[flow.Text]{}, errors.New("text: no messages to answer"))
			return
		}
		in := make([]chat.Message, len(msgs))
		for i, m := range msgs {
			in[i] = chat.Message{Role: m.Role, Content: chat.Content{Text: m.Text}}
		}
		s := chat.Sampling{Temperature: 0.1, TopK: 50, Repetition: 1.05}
		if o.Temperature != nil {
			s.Temperature = *o.Temperature
		}
		budget := 0
		if o.Think {
			budget = cmp.Or(o.ThinkBudget, 1024)
		}
		c.mu.Lock()
		defer c.mu.Unlock()
		stopped, last := false, ""
		done, err := c.e.Complete(ctx, in, nil, chat.Options{NoThink: !o.Think}, s, cmp.Or(o.MaxTokens, 1024), budget, func(a chat.Answer) bool {
			if a.Content == last {
				return false
			}
			last = a.Content
			stopped = !yield(flow.Chunk[flow.Text]{Data: flow.Text{Text: a.Content}}, nil)
			return stopped
		})
		if stopped {
			return
		}
		if err != nil {
			yield(flow.Chunk[flow.Text]{}, err)
			return
		}
		yield(flow.Chunk[flow.Text]{Data: flow.Text{Text: strings.TrimSpace(done.Content)}, Final: true}, nil)
	}
}

// Close unloads the model. An app leaves it to the Runtime (its Close, or the use's Release).
func (c *Chatter) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.e.Close()
}

func init() { airole.Register(chatRole) }
