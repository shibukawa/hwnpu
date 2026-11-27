package engine

import (
	"cmp"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"unicode/utf8"

	"github.com/shibukawa/hwnpu/ai/text"
	"github.com/shibukawa/hwnpu/internal/chat"
)

// request is the generation of the options given: sampling s (nil: def), maxTokens (0: 1024,
// Unlimited: until the cache is full) and seed.
func request(s *text.Sampling, def text.Sampling, maxTokens int, seed uint64) chat.Request {
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

// thinkBudget is text.ChatOptions.ThinkBudget as the engine takes it (0: none).
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
func generate(ctx context.Context, e *chat.Engine, msgs []chat.Message, tools []chat.Tool, opts chat.Options, req chat.Request, budget int, stops []string, fn func(text.Delta) bool) (text.Answer, error) {
	var a text.Answer
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
			if !fn(text.Delta{Reasoning: dr, Text: dc}) {
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
		a.ToolCalls = append(a.ToolCalls, text.ToolCall{ID: tc.ID, Name: tc.Function.Name, Arguments: tc.Function.Arguments})
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
			fn(text.Delta{Reasoning: dr, Text: dc})
		}
	}
	return a, nil
}

func stats(s chat.Stats) text.Stats {
	return text.Stats{Prompt: s.Prompt, Reused: s.Reused, Restored: s.Restored, Generated: s.Generated, Prefill: s.Prefill, Decode: s.Decode,
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
func chatMessages(msgs []text.Message) ([]chat.Message, error) {
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
func chatMedia(m text.Media) (chat.Media, error) {
	if m.URL != "" {
		return chat.Media{Audio: m.Audio, Data: m.URL}, nil
	}
	if m.Audio {
		return chat.Media{Audio: true, Data: base64.StdEncoding.EncodeToString(m.Data)}, nil
	}
	mime := http.DetectContentType(m.Data)
	if !strings.HasPrefix(mime, "image/") {
		return chat.Media{}, fmt.Errorf("%w: %s, not an image", text.ErrMedia, mime)
	}
	return chat.Media{Data: "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(m.Data)}, nil
}

// chatTools are tools as the engine takes them.
func chatTools(tools []text.Tool) ([]chat.Tool, error) {
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

