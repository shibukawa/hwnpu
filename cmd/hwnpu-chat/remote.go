package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/daemon"
)

// runThrough is ChatCmd through hwnpud (HWNPU_DAEMON=on or auto): hwnpud makes the model ready
// and runs it as its definition says, this process only sends the conversation and writes the
// answers. The flags of how a model is converted and run are hwnpud's to decide.
func (c *ChatCmd) runThrough(ctx context.Context, stop context.CancelFunc, cl *daemon.Client) error {
	if given := c.notThroughDaemon(); len(given) > 0 {
		return fmt.Errorf("%s through hwnpud (%s): hwnpud runs the model as its definition says (models.yaml); %s=off runs it here", strings.Join(given, ", "), cl.Socket, daemon.ModeEnv)
	}
	log := logw()
	info, err := cl.Load(ctx, daemon.LoadRequest{Model: c.Model, Memory: c.Memory, Terminal: console.Live(log), Allowlist: os.Getenv(chat.PolicyEnv)}, log)
	if err != nil {
		return err
	}
	where := "all on the NPU"
	if !info.OnNPU {
		where = "NOT all on the NPU"
	}
	fmt.Fprintf(log, "%s: through hwnpud (%s), context %d tokens, %s\n", info.Name, cl.Socket, info.Context, where)
	stop() // Ctrl-C stops an answer, not the chat
	think := !info.NoThink
	if c.Think != nil {
		think = *c.Think
	}
	return c.converse(remote{c: cl, info: info}, think)
}

// notThroughDaemon are the flags given that hwnpud does not take: those of where the models are
// and how they are converted and run.
func (c *ChatCmd) notThroughDaemon() []string {
	var out []string
	add := func(given bool, flag string) {
		if given {
			out = append(out, flag)
		}
	}
	lf, cf, ef := c.loadFlags, c.convertFlags, c.encoderFlags
	add(lf.Backend != "", "--backend")
	add(lf.Cache != "", "--cache")
	add(lf.PromptCache != "", "--prompt-cache")
	add(lf.Idle != 5*time.Minute, "--idle")
	add(lf.ExpertsKey != "", "--experts-key")
	add(lf.Models != "", "--models")
	add(cf.Context != 0, "--context")
	add(cf.Paged != 0, "--paged")
	add(cf.Shortest != 0, "--shortest")
	add(!cf.Sparse, "--no-sparse")
	add(cf.Package != "", "--package")
	add(cf.Config != "", "--config")
	add(cf.Weights != "", "--weights")
	add(cf.Tokenizer != "", "--tokenizer")
	add(cf.Format != "", "--format")
	add(cf.Codes != "", "--codes")
	add(cf.FormatFor != "", "--format-for")
	add(cf.Tier != nil, "--tier")
	add(cf.Draft != 8, "--draft")
	add(cf.MTP != chat.MTPAuto, "--mtp")
	add(cf.Prefill != "", "--prefill")
	add(ef.Vision != "", "--vision")
	add(ef.AudioEncoder != "", "--audio-encoder")
	add(isPath(c.Model), "--model with an export's directory")
	return out
}

// remote is a chat model hwnpud holds, answered through its chat completions.
type remote struct {
	c    *daemon.Client
	info daemon.ModelInfo
}

func (r remote) features() features {
	return features{images: r.info.Images, audio: r.info.Audio, translator: r.info.Translator}
}

// wireMessages are the messages as the API takes them: a content with images or clips as a list
// of parts in their order (the media markers are this process's, not hwnpud's).
func wireMessages(msgs []chat.Message) []map[string]any {
	out := make([]map[string]any, 0, len(msgs))
	for _, m := range msgs {
		w := map[string]any{"role": m.Role}
		if len(m.Content.Media) == 0 {
			w["content"] = m.Content.Text
		} else {
			var parts []map[string]any
			text := func(s string) {
				if s != "" {
					parts = append(parts, map[string]any{"type": "text", "text": s})
				}
			}
			media := func(md chat.Media) {
				if md.Audio {
					parts = append(parts, map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": md.Data, "format": "wav"}})
				} else {
					parts = append(parts, map[string]any{"type": "image_url", "image_url": map[string]any{"url": md.Data}})
				}
			}
			pieces := strings.Split(m.Content.Text, chat.MediaMarker)
			for i, p := range pieces {
				text(p)
				if i < len(pieces)-1 && i < len(m.Content.Media) {
					media(m.Content.Media[i])
				}
			}
			for i := len(pieces) - 1; i < len(m.Content.Media); i++ {
				media(m.Content.Media[i])
			}
			w["content"] = parts
		}
		if m.Reasoning != nil {
			w["reasoning_content"] = *m.Reasoning
		}
		out = append(out, w)
	}
	return out
}

// streamChunk is a server-sent event of a streamed chat completion.
type streamChunk struct {
	Seed    uint64 `json:"seed"`
	Choices []struct {
		Delta struct {
			Content   string `json:"content"`
			Reasoning string `json:"reasoning_content"`
		} `json:"delta"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		Details          struct {
			Cached int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
		HWNPU chat.UsageStats `json:"hwnpu"`
	} `json:"usage"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

func (r remote) reply(ctx context.Context, msgs []chat.Message, o turn, out *printer) (chat.Message, turnStats, error) {
	kwargs := map[string]any{"enable_thinking": o.think}
	if o.tr != nil {
		kwargs["target_lang"], kwargs["source_lang"], kwargs["form"] = o.tr.TargetLang, o.tr.SourceLang, o.tr.Form
	}
	body, err := json.Marshal(map[string]any{
		"model": r.info.Name, "messages": wireMessages(msgs), "stream": true,
		"stream_options": map[string]any{"include_usage": true},
		"max_tokens":     o.maxNew, "temperature": o.s.Temperature, "top_k": o.s.TopK, "top_p": o.s.TopP,
		"repetition_penalty": o.s.Repetition, "seed": o.seed, "chat_template_kwargs": kwargs,
	})
	if err != nil {
		return chat.Message{}, turnStats{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, r.c.URL("/v1/chat/completions"), bytes.NewReader(body))
	if err != nil {
		return chat.Message{}, turnStats{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(daemon.ThinkBudgetHeader, strconv.Itoa(o.thinkBudget))
	if v := os.Getenv(chat.PolicyEnv); v != "" {
		req.Header.Set(daemon.AllowlistHeader, v)
	}
	resp, err := r.c.HTTP.Do(req)
	if err != nil {
		return chat.Message{}, turnStats{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return chat.Message{}, turnStats{}, daemon.ResponseError(resp)
	}
	var reasoning, content strings.Builder
	var st turnStats
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 64<<10), 16<<20)
	for sc.Scan() {
		data, ok := strings.CutPrefix(sc.Text(), "data: ")
		if !ok {
			continue
		}
		if data == "[DONE]" {
			break
		}
		var ch streamChunk
		if err := json.Unmarshal([]byte(data), &ch); err != nil {
			return chat.Message{}, turnStats{}, fmt.Errorf("hwnpud: %w", err)
		}
		if ch.Error != nil {
			return chat.Message{}, turnStats{}, errors.New(ch.Error.Message)
		}
		if ch.Seed != 0 {
			st.seed = ch.Seed
		}
		for _, c := range ch.Choices {
			reasoning.WriteString(c.Delta.Reasoning)
			content.WriteString(c.Delta.Content)
			out.write(c.Delta.Reasoning, c.Delta.Content)
		}
		if u := ch.Usage; u != nil {
			st.prompt, st.reused, st.generated = u.PromptTokens, u.Details.Cached, u.CompletionTokens
			h := u.HWNPU
			st.prefill, st.decode = time.Duration(h.PrefillMS)*time.Millisecond, time.Duration(h.DecodeMS)*time.Millisecond
			st.thinkCut, st.thinkLooped, st.looped, st.drafted, st.accepted = h.ThinkCut, h.ThinkLooped, h.Looped, h.Drafted, h.Accepted
		}
	}
	err = sc.Err()
	if ctx.Err() != nil {
		err = context.Canceled // Ctrl-C: the answer so far is kept, as here
	}
	rs := reasoning.String()
	return chat.Message{Role: "assistant", Content: chat.Content{Text: content.String()}, Reasoning: &rs}, st, err
}
