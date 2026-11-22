package agent

import (
	"context"
	"errors"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/ai/text"
)

// script is a model that answers from a script, one answer a round, and keeps what it read.
type script struct {
	mu      sync.Mutex
	answers []text.Answer
	read    [][]text.Message
	tools   []int // the tools offered at each round
	wait    chan struct{}
}

func (s *script) Reply(ctx context.Context, msgs []text.Message, o text.ChatOptions, fn func(text.Delta) bool) (text.Answer, error) {
	s.mu.Lock()
	s.read = append(s.read, slices.Clone(msgs))
	s.tools = append(s.tools, len(o.Tools))
	if len(s.answers) == 0 {
		s.mu.Unlock()
		return text.Answer{Text: "…"}, nil
	}
	a := s.answers[0]
	s.answers = s.answers[1:]
	wait := s.wait
	s.mu.Unlock()
	for _, piece := range strings.SplitAfter(a.Text, "。") {
		if piece != "" && fn != nil && !fn(text.Delta{Text: piece}) {
			break
		}
	}
	if wait != nil {
		select {
		case <-wait:
		case <-ctx.Done():
			return a, ctx.Err()
		}
	}
	return a, nil
}

func clock() Tool {
	return Tool{Name: "get_time", Run: func(ctx context.Context, a Args) (string, error) { return "午後3時", nil }}
}

func TestTurn(t *testing.T) {
	m := &script{answers: []text.Answer{
		{ToolCalls: []text.ToolCall{{ID: "1", Name: "get_time", Arguments: `{}`}}},
		{Text: "午後3時です。"},
	}}
	var calls []string
	a := New(m, Config{System: "sys", Tools: []Tool{clock()}})
	var said strings.Builder
	ans, err := a.Turn(context.Background(), "今何時？", TurnOptions{
		Text: func(d string) { said.WriteString(d) },
		Tool: func(name, args string, slow bool) { calls = append(calls, name) },
	})
	if err != nil || ans != "午後3時です。" || said.String() != ans {
		t.Fatalf("Turn = %q, %v; said %q", ans, err, said.String())
	}
	if !slices.Equal(calls, []string{"get_time"}) {
		t.Errorf("calls %v", calls)
	}
	// the second round read the call and its result
	last := m.read[1]
	if r := last[len(last)-1]; r.Role != "tool" || r.Text != "午後3時" || r.ToolCallID != "1" {
		t.Errorf("the result told: %+v", r)
	}
	h := a.History()
	if len(h) != 4 || h[0].Text != "今何時？" || h[3].Text != "午後3時です。" {
		t.Errorf("history %+v", h)
	}
}

func TestLastRoundWithoutTools(t *testing.T) {
	call := text.Answer{ToolCalls: []text.ToolCall{{Name: "get_time"}}}
	m := &script{answers: []text.Answer{call, call, {Text: "答え。"}}}
	a := New(m, Config{Tools: []Tool{clock()}, Rounds: 3})
	if ans, err := a.Turn(context.Background(), "q", TurnOptions{}); err != nil || ans != "答え。" {
		t.Fatalf("Turn = %q, %v", ans, err)
	}
	if !slices.Equal(m.tools, []int{1, 1, 0}) {
		t.Errorf("tools offered by round: %v", m.tools)
	}
}

func TestToolError(t *testing.T) {
	m := &script{answers: []text.Answer{{ToolCalls: []text.ToolCall{{Name: "boom", Arguments: `{"x": 1}`}}}, {Text: "だめでした。"}}}
	boom := Tool{Name: "boom", Run: func(ctx context.Context, a Args) (string, error) { return "", errors.New("つながらない") }}
	a := New(m, Config{Tools: []Tool{boom}})
	a.Turn(context.Background(), "q", TurnOptions{})
	last := m.read[1]
	if got := last[len(last)-1].Text; got != "失敗しました: つながらない" {
		t.Errorf("the model was told %q", got)
	}
}

func TestGate(t *testing.T) {
	m := &script{answers: []text.Answer{{Text: "一つ目。二つ目。"}}}
	a := New(m, Config{})
	gate := make(chan struct{})
	var mu sync.Mutex
	var said []string
	done := make(chan string)
	go func() {
		ans, _ := a.Turn(context.Background(), "q", TurnOptions{Gate: gate, Text: func(d string) {
			mu.Lock()
			said = append(said, d)
			mu.Unlock()
		}})
		done <- ans
	}()
	select {
	case <-done:
		t.Fatal("the turn ended before its gate")
	case <-time.After(50 * time.Millisecond):
	}
	mu.Lock()
	if len(said) != 0 {
		t.Errorf("said %q before the gate", said)
	}
	mu.Unlock()
	close(gate)
	if ans := <-done; ans != "一つ目。二つ目。" {
		t.Errorf("answer %q", ans)
	}
	if !slices.Equal(said, []string{"一つ目。", "二つ目。"}) {
		t.Errorf("said %q", said)
	}
}

func TestCancelBeforeGate(t *testing.T) {
	m := &script{answers: []text.Answer{{Text: "答え。"}}}
	a := New(m, Config{})
	ctx, cancel := context.WithCancel(context.Background())
	var said []string
	go func() { time.Sleep(30 * time.Millisecond); cancel() }()
	_, err := a.Turn(ctx, "q", TurnOptions{Gate: make(chan struct{}), Text: func(d string) { said = append(said, d) }})
	if !errors.Is(err, context.Canceled) || len(said) != 0 || len(a.History()) != 0 {
		t.Errorf("err %v, said %q, history %v: want nothing", err, said, a.History())
	}
}

func TestKeep(t *testing.T) {
	a := New(&script{}, Config{MaxMessages: 4})
	turn := func(q string, tools bool) []text.Message {
		m := []text.Message{{Role: "user", Text: q}}
		if tools {
			m = append(m, text.Message{Role: "assistant", ToolCalls: []text.ToolCall{{Name: "get_time"}}}, text.Message{Role: "tool", Text: "3時"})
		}
		return append(m, text.Message{Role: "assistant", Text: "答え"})
	}
	a.keep(turn("1", false))
	a.keep(turn("2", true))
	if len(a.history) != 4 || a.history[0].Text != "2" {
		t.Errorf("history %v: want the second turn alone", a.history)
	}
	a.keep(turn("3", false))
	if len(a.history) != 2 || a.history[0].Text != "3" {
		t.Errorf("history %v: want the third turn alone", a.history)
	}
}

func TestResetAfter(t *testing.T) {
	m := &script{answers: []text.Answer{{Text: "a"}, {Text: "b"}}}
	a := New(m, Config{ResetAfter: time.Millisecond})
	a.Turn(context.Background(), "1", TurnOptions{})
	time.Sleep(5 * time.Millisecond)
	a.Turn(context.Background(), "2", TurnOptions{})
	if got := len(m.read[1]); got != 2 { // the system prompt and the request
		t.Errorf("the second turn read %d messages: want the first forgotten", got)
	}
}

func TestArgs(t *testing.T) {
	a := Args{"n": 180.0, "s": " 90 ", "t": " x "}
	if a.Int("n") != 180 || a.Int("s") != 90 || a.Int("none") != 0 || a.String("t") != "x" {
		t.Errorf("args %v", a)
	}
}
