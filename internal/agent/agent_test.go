package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/internal/chat"
)

// script is a model that answers from a list, and remembers what it was asked.
type script struct {
	answers []chat.Answer
	seen    [][]chat.Message
	tools   []int // tools offered in each request
	context int
}

func (s *script) Complete(ctx context.Context, msgs []chat.Message, tools []chat.Tool, o chat.Options, _ chat.Sampling, _ int, emit func(chat.Answer) bool) (chat.Completion, error) {
	s.seen = append(s.seen, slices.Clone(msgs))
	s.tools = append(s.tools, len(tools))
	if len(s.answers) == 0 {
		return chat.Completion{}, errors.New("the script has ended")
	}
	a := s.answers[0]
	s.answers = s.answers[1:]
	if emit != nil {
		emit(a)
	}
	return chat.Completion{Answer: a, Rendered: "prompt", Raw: "generated", Stats: chat.Stats{Finish: "stop", Generated: 1}}, nil
}

// Count charges 4 tokens a message and a token per 4 characters.
func (s *script) Count(ctx context.Context, msgs []chat.Message, tools []chat.Tool, o chat.Options) (int, error) {
	n := 0
	for _, m := range msgs {
		n += 4 + len(m.Content.Text)/4
	}
	return n, nil
}

func (s *script) Context() int {
	if s.context == 0 {
		return 1 << 20
	}
	return s.context
}

func call(name string, args map[string]any) chat.ToolCall {
	var tc chat.ToolCall
	tc.ID, tc.Type = "call_"+name, "function"
	tc.Function.Name = name
	b, _ := json.Marshal(args)
	tc.Function.Arguments = string(b)
	return tc
}

func calls(cs ...chat.ToolCall) chat.Answer { return chat.Answer{Reasoning: "hmm", ToolCalls: cs} }

func said(s string) chat.Answer { return chat.Answer{Reasoning: "hmm", Content: s} }

func skipShell(t *testing.T) {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("the test uses sh")
	}
}

const mainGo = "package main\n\nimport \"fmt\"\n\nfunc add(a, b int) int {\n\treturn a - b\n}\n\nfunc main() {\n\tfmt.Println(add(2, 3))\n}\n"

func project(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte(mainGo), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// A task runs read, edit and bash in turn and ends with the answer; the tool results go back
// with the call's id, the thinking of earlier rounds is dropped, and the file is changed.
func TestRun(t *testing.T) {
	skipShell(t)
	dir := project(t)
	m := &script{answers: []chat.Answer{
		calls(call("read", map[string]any{"path": "main.go"})),
		calls(call("edit", map[string]any{"path": "main.go", "old": "return a - b", "new": "return a + b"})),
		calls(call("bash", map[string]any{"command": "grep -n 'a + b' main.go"})),
		said("Fixed add."),
	}}
	s, err := New(m, Config{Dir: dir, Think: true})
	if err != nil {
		t.Fatal(err)
	}
	var log []string
	ev := Events{
		Call:   func(name, what string) { log = append(log, name+" "+what) },
		Result: func(name, result string) { log = append(log, "= "+strings.SplitN(result, "\n", 2)[0]) },
	}
	got, err := s.Run(context.Background(), "fix add", ev)
	if err != nil || got != "Fixed add." {
		t.Fatalf("Run = %q, %v", got, err)
	}
	want := []string{"read main.go", "= package main", "edit main.go", "= Edited main.go at line 6.", "bash grep -n 'a + b' main.go", "= 6:\treturn a + b"}
	if !slices.Equal(log, want) {
		t.Errorf("events %q\nwant   %q", log, want)
	}
	if b, _ := os.ReadFile(filepath.Join(dir, "main.go")); !strings.Contains(string(b), "return a + b") {
		t.Errorf("main.go not edited:\n%s", b)
	}
	last := m.seen[3]
	if len(last) != 8 {
		t.Fatalf("%d messages in the last request, want 8", len(last))
	}
	for i, role := range []string{"system", "user", "assistant", "tool", "assistant", "tool", "assistant", "tool"} {
		if last[i].Role != role {
			t.Errorf("message %d is %s, want %s", i, last[i].Role, role)
		}
	}
	if last[3].ToolCallID != "call_read" || last[3].Name != "read" || !strings.Contains(last[3].Content.Text, "return a - b") {
		t.Errorf("tool message %+v", last[3])
	}
	if last[2].Reasoning != nil || last[2].Content.Text != "" || len(last[2].ToolCalls) != 1 {
		t.Errorf("assistant message %+v", last[2])
	}
	if msgs := s.Messages(); len(msgs) != 9 || msgs[8].Content.Text != "Fixed add." {
		t.Errorf("%d messages after the task", len(msgs))
	}
	s.Reset()
	if len(s.Messages()) != 1 {
		t.Error("Reset kept messages")
	}
}

// -keep-thinking keeps the reasoning of the rounds of a task.
func TestKeepThinking(t *testing.T) {
	m := &script{answers: []chat.Answer{calls(call("read", map[string]any{"path": "none"})), said("done")}}
	s, err := New(m, Config{Dir: t.TempDir(), KeepThinking: true})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Run(context.Background(), "task", Events{}); err != nil {
		t.Fatal(err)
	}
	if r := m.seen[1][2].Reasoning; r == nil || *r != "hmm" {
		t.Errorf("reasoning %v", r)
	}
}

func sess(t *testing.T, cfg Config) *Session {
	t.Helper()
	s, err := New(&script{}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

// run makes one call in a fresh session.
func run(t *testing.T, cfg Config, tc chat.ToolCall) string {
	t.Helper()
	return sess(t, cfg).call(context.Background(), tc, Events{})
}

func TestEdit(t *testing.T) {
	dir := project(t)
	s := sess(t, Config{Dir: dir})
	edit := func(old, new string) string {
		return s.call(context.Background(), call("edit", map[string]any{"path": "main.go", "old": old, "new": new}), Events{})
	}
	// a file is read before it is edited
	if got := edit("return a - b", "x"); got != "Read main.go before editing it." {
		t.Errorf("unread: %q", got)
	}
	if got := s.call(context.Background(), call("read", map[string]any{"path": "main.go"}), Events{}); !strings.HasPrefix(got, "package main") {
		t.Fatalf("read: %q", got)
	}
	if got := edit("return a * b", "x"); !strings.Contains(got, "was not found") {
		t.Errorf("missing: %q", got)
	}
	if got := edit("a", "x"); !strings.Contains(got, "occurs") || !strings.Contains(got, "times") {
		t.Errorf("ambiguous: %q", got)
	}
	// blank lines around old that the file does not have
	if got := edit("\n\nreturn a - b\n\n", "\nreturn a + b\n"); got != "Edited main.go at line 6." {
		t.Errorf("trimmed: %q", got)
	}
	if b, _ := os.ReadFile(filepath.Join(dir, "main.go")); string(b) != strings.Replace(mainGo, "a - b", "a + b", 1) {
		t.Errorf("file:\n%s", b)
	}
	if got := edit("", "x"); !strings.Contains(got, "old is empty") {
		t.Errorf("empty: %q", got)
	}
	if got := s.call(context.Background(), call("edit", map[string]any{"path": "../x.go", "old": "a", "new": "b"}), Events{}); !strings.Contains(got, "outside") {
		t.Errorf("outside: %q", got)
	}
	if got := s.call(context.Background(), call("edit", map[string]any{"path": "none.go", "old": "a", "new": "b"}), Events{}); got != "none.go does not exist. The files here are: main.go. Use write to create a file." {
		t.Errorf("missing file: %q", got)
	}
	s.Reset()
	if got := edit("a + b", "a - b"); got != "Read main.go before editing it." {
		t.Errorf("after Reset: %q", got)
	}
}

// A small model writes /main.go for main.go: a path outside the directory names the file
// there when it exists, or a new file in a directory that exists.
func TestSlashPaths(t *testing.T) {
	dir := project(t)
	s := sess(t, Config{Dir: dir})
	do := func(name string, args map[string]any) string {
		return s.call(context.Background(), call(name, args), Events{})
	}
	if got := do("read", map[string]any{"path": "/main.go"}); !strings.HasPrefix(got, "package main") {
		t.Errorf("read: %q", got)
	}
	if got := do("edit", map[string]any{"path": "main.go", "old": "a - b", "new": "a + b"}); got != "Edited main.go at line 6." {
		t.Errorf("edit after reading /main.go: %q", got)
	}
	if got := do("write", map[string]any{"path": "/new.txt", "content": "x"}); got != "Wrote /new.txt (1 bytes)." {
		t.Errorf("write: %q", got)
	}
	if _, err := os.Stat(filepath.Join(dir, "new.txt")); err != nil {
		t.Error("new.txt not in the directory")
	}
	if got := do("write", map[string]any{"path": "/no/such/dir/x.txt", "content": "x"}); !strings.Contains(got, "outside the working directory; use a path relative") {
		t.Errorf("write elsewhere: %q", got)
	}
	if got := do("read", map[string]any{"path": "/no-such-file-zzz"}); got != "/no-such-file-zzz does not exist. The files here are: main.go, new.txt." {
		t.Errorf("read elsewhere: %q", got)
	}
}

func TestWrite(t *testing.T) {
	dir := t.TempDir()
	s := sess(t, Config{Dir: dir})
	write := func(path, content string) string {
		return s.call(context.Background(), call("write", map[string]any{"path": path, "content": content}), Events{})
	}
	if got := write("a/b.txt", "hi\n"); got != "Wrote a/b.txt (3 bytes)." {
		t.Errorf("write: %q", got)
	}
	if b, _ := os.ReadFile(filepath.Join(dir, "a", "b.txt")); string(b) != "hi\n" {
		t.Errorf("content %q", b)
	}
	// a file it wrote is known; one it has not read is not overwritten
	if got := write("a/b.txt", "yo"); got != "Replaced a/b.txt (2 bytes)." {
		t.Errorf("rewrite: %q", got)
	}
	if got := run(t, Config{Dir: dir}, call("write", map[string]any{"path": "a/b.txt", "content": "zz"})); got != "a/b.txt exists and you have not read it. Use edit to change it, or read it first." {
		t.Errorf("unread rewrite: %q", got)
	}
	if b, _ := os.ReadFile(filepath.Join(dir, "a", "b.txt")); string(b) != "yo" {
		t.Errorf("content after refusal %q", b)
	}
	if got := write(filepath.Join(dir, "..", "x"), ""); !strings.Contains(got, "outside") {
		t.Errorf("outside: %q", got)
	}
}

func TestRead(t *testing.T) {
	dir := t.TempDir()
	var lines []string
	for i := 1; i <= 10; i++ {
		lines = append(lines, fmt.Sprintf("line %d", i))
	}
	if err := os.WriteFile(filepath.Join(dir, "f.txt"), []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "bin"), []byte("a\x00b"), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg := Config{Dir: dir, ReadLines: 4}
	read := func(args map[string]any) string { return run(t, cfg, call("read", args)) }
	if got, want := read(map[string]any{"path": "f.txt"}), "line 1\nline 2\nline 3\nline 4\n[Lines 1-4 of 10. Use offset=5 to read on.]"; got != want {
		t.Errorf("read:\n%q\nwant %q", got, want)
	}
	// offset as text, limit over the cap
	if got, want := read(map[string]any{"path": "f.txt", "offset": "8", "limit": 100}), "line 8\nline 9\nline 10\n[Lines 8-10 of 10.]"; got != want {
		t.Errorf("offset:\n%q\nwant %q", got, want)
	}
	if got, want := read(map[string]any{"path": "f.txt", "offset": 2, "limit": 2}), "line 2\nline 3\n[Lines 2-3 of 10. Use offset=4 to read on.]"; got != want {
		t.Errorf("limit:\n%q\nwant %q", got, want)
	}
	if got := read(map[string]any{"path": "f.txt", "offset": 11}); !strings.Contains(got, "past the end") {
		t.Errorf("past: %q", got)
	}
	if got := read(map[string]any{"path": "bin"}); got != "bin is a binary file." {
		t.Errorf("binary: %q", got)
	}
	if got := read(map[string]any{"path": "."}); !strings.Contains(got, "directory") {
		t.Errorf("dir: %q", got)
	}
	if got := read(map[string]any{"path": "none"}); got != "none does not exist. The files here are: bin, f.txt." {
		t.Errorf("missing: %q", got)
	}
	// a byte limit cuts between lines
	cfg.ReadLines, cfg.ReadBytes = 200, 15
	if got, want := read(map[string]any{"path": "f.txt"}), "line 1\nline 2\n[Lines 1-2 of 10. Use offset=3 to read on.]"; got != want {
		t.Errorf("bytes:\n%q\nwant %q", got, want)
	}
}

func TestBash(t *testing.T) {
	skipShell(t)
	dir := t.TempDir()
	cfg := Config{Dir: dir, TailLines: 2}
	bash := func(args map[string]any) string { return run(t, cfg, call("bash", args)) }
	if got := bash(map[string]any{"command": "printf 'a\\nb\\nc\\nd\\n'"}); got != "[... 2 lines omitted]\nc\nd" {
		t.Errorf("tail: %q", got)
	}
	if got := bash(map[string]any{"command": "echo oops >&2; exit 3"}); got != "oops\n[exit status 3]" {
		t.Errorf("exit: %q", got)
	}
	if got := bash(map[string]any{"command": "true"}); got != "(no output)" {
		t.Errorf("silent: %q", got)
	}
	t0 := time.Now()
	if got := bash(map[string]any{"command": "echo start; sleep 20", "timeout": 1}); got != "start\n[killed after 1 seconds]" {
		t.Errorf("timeout: %q", got)
	}
	if d := time.Since(t0); d > 5*time.Second {
		t.Errorf("the timeout took %v", d)
	}
	if got := bash(map[string]any{}); got != "command is required." {
		t.Errorf("no command: %q", got)
	}
	// the directory's path is "." in outputs, and cd elsewhere is refused with the rest to run
	if got := bash(map[string]any{"command": "pwd"}); got != "." {
		t.Errorf("pwd: %q", got)
	}
	for _, cmd := range []string{"cd /tmp && go run main.go", `cd "/tmp/x" ; go run main.go`, "cd ~ && go run main.go  # comment"} {
		if got := bash(map[string]any{"command": cmd}); !strings.HasPrefix(got, "Commands run in the current directory") || !strings.Contains(got, "Run: go run main.go") {
			t.Errorf("%s: %q", cmd, got)
		}
	}
	if err := os.Mkdir(filepath.Join(dir, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	if got := bash(map[string]any{"command": "cd sub && pwd"}); got != "./sub" {
		t.Errorf("cd inside: %q", got)
	}
	if got := bash(map[string]any{"command": "cd " + dir + " && echo here"}); got != "here" {
		t.Errorf("cd to the directory itself: %q", got)
	}
}

func TestTail(t *testing.T) {
	for _, tc := range []struct {
		in           string
		lines, bytes int
		want         string
		wantOmitted  int
	}{
		{"", 10, 100, "", 0},
		{"a\nb\nc\n", 10, 100, "a\nb\nc", 0},
		{"a\nb\nc\n", 2, 100, "b\nc", 1},
		{"aaaa\nbbbb\ncccc", 10, 9, "bbbb\ncccc", 1},
		{"aaaa\nbbbbbbbbbbbb", 10, 5, "…bbbbb", 1},
	} {
		got, omitted := tail(tc.in, tc.lines, tc.bytes)
		if got != tc.want || omitted != tc.wantOmitted {
			t.Errorf("tail(%q, %d, %d) = %q, %d; want %q, %d", tc.in, tc.lines, tc.bytes, got, omitted, tc.want, tc.wantOmitted)
		}
	}
}

// Old tool results are folded to their first line when the conversation outgrows the context,
// the latest Keep stay whole, and a conversation that does not fit even so is an error.
func TestBudget(t *testing.T) {
	dir := t.TempDir()
	long := strings.Repeat("0123456789\n", 20) // 220 bytes: 59 tokens a result
	if err := os.WriteFile(filepath.Join(dir, "f.txt"), []byte(long), 0o644); err != nil {
		t.Fatal(err)
	}
	// the same file three times, each call a little different so that none counts as a repeat
	read := func(limit int) chat.Answer {
		return calls(call("read", map[string]any{"path": "f.txt", "limit": limit}))
	}
	// system (about 200 tokens) + 3 whole results (177) is over 320; folding 2 brings it under
	m := &script{context: 420, answers: []chat.Answer{read(100), read(101), read(102), said("ok")}}
	s, err := New(m, Config{Dir: dir, Reserve: 100, Keep: 1})
	if err != nil {
		t.Fatal(err)
	}
	if n, _ := s.Count(context.Background()); n > 220 {
		t.Fatalf("the system prompt takes %d tokens", n)
	}
	if _, err := s.Run(context.Background(), "read it", Events{}); err != nil {
		t.Fatal(err)
	}
	last := m.seen[3]
	var tools []string
	for _, msg := range last {
		if msg.Role == "tool" {
			tools = append(tools, msg.Content.Text)
		}
	}
	if len(tools) != 3 {
		t.Fatalf("%d tool messages", len(tools))
	}
	folded := "0123456789\n[... 19 more lines omitted]"
	if tools[0] != folded || tools[1] != folded {
		t.Errorf("old results not folded: %q, %q", tools[0], tools[1])
	}
	if !strings.HasPrefix(tools[2], long[:50]) {
		t.Errorf("the latest result was folded: %q", tools[2])
	}
	if n, _ := s.Count(context.Background()); n > 320 {
		t.Errorf("%d tokens after folding", n)
	}

	m = &script{context: 150, answers: []chat.Answer{read(100), read(101), said("ok")}}
	s, err = New(m, Config{Dir: dir, Reserve: 10, Keep: 1})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Run(context.Background(), "read it", Events{}); !errors.Is(err, chat.ErrContext) {
		t.Errorf("err = %v, want ErrContext", err)
	}
}

func TestUnknownTool(t *testing.T) {
	m := &script{answers: []chat.Answer{calls(call("grep", map[string]any{"pattern": "x"})), said("ok")}}
	s, err := New(m, Config{Dir: t.TempDir(), Tools: []string{"read", "bash"}})
	if err != nil {
		t.Fatal(err)
	}
	if got, err := s.Run(context.Background(), "task", Events{}); err != nil || got != "ok" {
		t.Fatal(got, err)
	}
	if got := m.seen[1][3].Content.Text; got != `There is no tool "grep". The tools are read, bash.` {
		t.Errorf("tool message %q", got)
	}
	if _, err := New(m, Config{Tools: []string{"grep"}}); err == nil {
		t.Error("New took a tool that does not exist")
	}
}

func TestBadArguments(t *testing.T) {
	tc := call("read", nil)
	tc.Function.Arguments = "{not json"
	m := &script{answers: []chat.Answer{{ToolCalls: []chat.ToolCall{tc}}, said("ok")}}
	s, err := New(m, Config{Dir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Run(context.Background(), "task", Events{}); err != nil {
		t.Fatal(err)
	}
	if got := m.seen[1][3].Content.Text; !strings.HasPrefix(got, "The arguments are not a JSON object") {
		t.Errorf("tool message %q", got)
	}
}

func TestSteps(t *testing.T) {
	// three different calls, then the rounds run out: a request without tools asks for the answer
	reads := func(path string) chat.Answer { return calls(call("read", map[string]any{"path": path})) }
	m := &script{answers: []chat.Answer{reads("a"), reads("b"), said("I looked at a and b."), said("never")}}
	s, err := New(m, Config{Dir: t.TempDir(), Steps: 2})
	if err != nil {
		t.Fatal(err)
	}
	if got, err := s.Run(context.Background(), "task", Events{}); err != nil || got != "I looked at a and b." {
		t.Errorf("Run = %q, %v", got, err)
	}
	if len(m.seen) != 3 || m.tools[2] != 0 {
		t.Errorf("%d requests, tools %v", len(m.seen), m.tools)
	}
	m = &script{answers: []chat.Answer{reads("a"), reads("b"), reads("c")}}
	s, _ = New(m, Config{Dir: t.TempDir(), Steps: 2})
	if _, err := s.Run(context.Background(), "task", Events{}); !errors.Is(err, ErrSteps) {
		t.Errorf("err = %v, want ErrSteps", err)
	}
}

// A model that writes its final answer and adds a call anyway, round after round, ends the
// task at the second answer, without its calls.
func TestRepeatedAnswer(t *testing.T) {
	skipShell(t)
	answer := "Fixed the bug in main.go: the add function returned a - b instead of a + b, so add(2, 3) gave -1; it now returns a + b and prints 5."
	first := chat.Answer{Content: answer, ToolCalls: []chat.ToolCall{call("bash", map[string]any{"command": "echo done"})}}
	second := chat.Answer{Content: strings.Replace(answer, "Fixed", "I fixed", 1), ToolCalls: []chat.ToolCall{call("bash", map[string]any{"command": "echo complete"})}}
	m := &script{answers: []chat.Answer{first, second, said("never")}}
	s, err := New(m, Config{Dir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	got, err := s.Run(context.Background(), "task", Events{})
	if err != nil || got != second.Content {
		t.Fatalf("Run = %q, %v", got, err)
	}
	if len(m.seen) != 2 {
		t.Errorf("%d requests, want 2", len(m.seen))
	}
	msgs := s.Messages()
	if last := msgs[len(msgs)-1]; last.Role != "assistant" || len(last.ToolCalls) != 0 {
		t.Errorf("last message %+v", last)
	}
	// a short remark before a call is not a final answer
	m = &script{answers: []chat.Answer{
		{Content: "Let me look.", ToolCalls: []chat.ToolCall{call("read", map[string]any{"path": "a"})}},
		{Content: "Let me look.", ToolCalls: []chat.ToolCall{call("read", map[string]any{"path": "b"})}},
		said("ok"),
	}}
	s, _ = New(m, Config{Dir: t.TempDir()})
	if got, err := s.Run(context.Background(), "task", Events{}); err != nil || got != "ok" {
		t.Errorf("Run = %q, %v", got, err)
	}
}

// The same call twice in a row is answered with a hint instead of being run; a third time
// ends the task.
func TestRepeatedCall(t *testing.T) {
	dir := project(t)
	read := calls(call("read", map[string]any{"path": "main.go"}))
	m := &script{answers: []chat.Answer{read, read, said("ok")}}
	s, err := New(m, Config{Dir: dir})
	if err != nil {
		t.Fatal(err)
	}
	if got, err := s.Run(context.Background(), "task", Events{}); err != nil || got != "ok" {
		t.Fatalf("Run = %q, %v", got, err)
	}
	if got := m.seen[2][5].Content.Text; got != nudge {
		t.Errorf("second result %q", got)
	}
	if got := m.seen[2][3].Content.Text; !strings.HasPrefix(got, "package main") {
		t.Errorf("first result %q", got)
	}
	// the third time, a request without tools asks for the answer
	m = &script{answers: []chat.Answer{read, read, {Content: "stuck", ToolCalls: read.ToolCalls}, said("I read main.go."), said("never")}}
	s, _ = New(m, Config{Dir: dir})
	if got, err := s.Run(context.Background(), "task", Events{}); err != nil || got != "I read main.go." {
		t.Errorf("Run = %q, %v", got, err)
	}
	if len(m.seen) != 4 || m.tools[3] != 0 || m.tools[2] != 4 {
		t.Errorf("%d requests, tools %v", len(m.seen), m.tools)
	}
	if last := m.seen[3][len(m.seen[3])-1]; last.Role != "user" || last.Content.Text != closing {
		t.Errorf("closing request ends with %+v", last)
	}
	// and when that gives nothing, ErrLoop
	m = &script{answers: []chat.Answer{read, read, read, said("")}}
	s, _ = New(m, Config{Dir: dir})
	if got, err := s.Run(context.Background(), "task", Events{}); !errors.Is(err, ErrLoop) || got != "" {
		t.Errorf("Run = %q, %v", got, err)
	}
}

func TestSimilar(t *testing.T) {
	long := "Fixed the bug in main.go: the add function returned a minus b instead of a plus b, so calling it with two and three gave minus one; it now returns the sum and prints five as expected."
	for _, tc := range []struct {
		a, b string
		want bool
	}{
		{"", "", false},
		{"short answer", "short answer", false},
		{long, long, true},
		{long + " Let me know if you need anything else!", long, true},
		{strings.Replace(long, "Fixed", "I fixed", 1), long, true},
		{long, "The tests pass now. I changed the loop in parse.go to stop at the first blank line, added a case for an empty input, and removed the unused helper that the linter complained about yesterday.", false},
	} {
		if got := similar(tc.a, tc.b); got != tc.want {
			t.Errorf("similar(%q, %q) = %v", tc.a, tc.b, got)
		}
	}
}

func TestCut(t *testing.T) {
	m := &cut{script{answers: []chat.Answer{said("half an")}}}
	s, err := New(m, Config{Dir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	if got, err := s.Run(context.Background(), "task", Events{}); !errors.Is(err, ErrCut) || got != "half an" {
		t.Errorf("Run = %q, %v", got, err)
	}
}

// cut is a script whose answers end by the token limit.
type cut struct{ script }

func (c *cut) Complete(ctx context.Context, msgs []chat.Message, tools []chat.Tool, o chat.Options, s chat.Sampling, n int, emit func(chat.Answer) bool) (chat.Completion, error) {
	r, err := c.script.Complete(ctx, msgs, tools, o, s, n, emit)
	r.Finish = "length"
	return r, err
}

// Allow refuses a tool that writes: it does not run, and the model is told.
func TestAllow(t *testing.T) {
	skipShell(t)
	dir := t.TempDir()
	m := &script{answers: []chat.Answer{
		calls(call("bash", map[string]any{"command": "touch made"}), call("read", map[string]any{"path": "none"})),
		said("ok"),
	}}
	var asked []string
	s, err := New(m, Config{Dir: dir, Allow: func(tool, what string) bool {
		asked = append(asked, tool+" "+what)
		return false
	}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Run(context.Background(), "task", Events{}); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(asked, []string{"bash touch made"}) {
		t.Errorf("asked %q", asked)
	}
	if _, err := os.Stat(filepath.Join(dir, "made")); err == nil {
		t.Error("the command ran")
	}
	if got := m.seen[1][3].Content.Text; got != "Not allowed by the user." {
		t.Errorf("tool message %q", got)
	}
	if got := m.seen[1][4].Content.Text; !strings.HasPrefix(got, "none does not exist.") {
		t.Errorf("read ran anyway: %q", got)
	}
	// a command on the allow list runs without asking
	m = &script{answers: []chat.Answer{calls(call("bash", map[string]any{"command": "touch made"})), said("ok")}}
	asked = nil
	s, _ = New(m, Config{Dir: dir, AllowList: []string{"touch *", "ls*"}, Allow: func(tool, what string) bool {
		asked = append(asked, tool+" "+what)
		return false
	}})
	if _, err := s.Run(context.Background(), "task", Events{}); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "made")); err != nil || len(asked) != 0 {
		t.Errorf("allowed command: %v, asked %q", err, asked)
	}
}

func TestAllowed(t *testing.T) {
	patterns := []string{"go *", "ls*", "git status", "cat *"}
	for cmd, want := range map[string]bool{
		"go run main.go":                true,
		"go test ./... && ls -la":       true,
		"ls":                            true,
		"lsof":                          true,
		"git status":                    true,
		"git push":                      false,
		"ls; rm -rf .":                  false,
		"cat main.go | grep add":        false,
		"go run main.go || echo failed": false,
		"  go build  ":                  true,
		"":                              true,
	} {
		if got := Allowed(cmd, patterns); got != want {
			t.Errorf("Allowed(%q) = %v, want %v", cmd, got, want)
		}
	}
	if Allowed("ls", nil) {
		t.Error("no patterns allow nothing")
	}
	if !glob("a*b*c", "aXXbYYc") || glob("a*b*c", "aXXcYYb") || !glob("*", "anything") {
		t.Error("glob")
	}
}

// Thinking past the budget is cut, and the answer is asked without thinking.
func TestThinkBudget(t *testing.T) {
	m := &thinker{}
	s, err := New(m, Config{Dir: t.TempDir(), Think: true, ThinkBudget: 10})
	if err != nil {
		t.Fatal(err)
	}
	var notes []string
	got, err := s.Run(context.Background(), "task", Events{Note: func(text string) { notes = append(notes, text) }})
	if err != nil || got != "answer" {
		t.Fatalf("Run = %q, %v", got, err)
	}
	if !slices.Equal(m.noThink, []bool{false, true}) || len(notes) != 1 || !strings.HasPrefix(notes[0], "thinking cut after 11 tokens") {
		t.Errorf("requests %v, notes %q", m.noThink, notes)
	}
	// without a limit, the model finishes thinking on its own
	m = &thinker{}
	s, _ = New(m, Config{Dir: t.TempDir(), Think: true, ThinkBudget: -1})
	if got, err := s.Run(context.Background(), "task", Events{}); err != nil || got != "answer" || !slices.Equal(m.noThink, []bool{false}) {
		t.Errorf("Run = %q, %v, requests %v", got, err, m.noThink)
	}
}

// thinker thinks for 50 tokens, unless stopped, and then answers.
type thinker struct {
	script
	noThink []bool
}

func (m *thinker) Complete(ctx context.Context, msgs []chat.Message, tools []chat.Tool, o chat.Options, _ chat.Sampling, _ int, emit func(chat.Answer) bool) (chat.Completion, error) {
	m.noThink = append(m.noThink, o.NoThink)
	n := 0
	if !o.NoThink {
		for ; n < 50; n++ {
			if emit(chat.Answer{Reasoning: strings.Repeat("hm ", n+1), Thinking: true}) {
				return chat.Completion{Answer: chat.Answer{Reasoning: strings.Repeat("hm ", n+1), Thinking: true}, Stats: chat.Stats{Finish: "stop", Generated: n + 1}}, nil
			}
		}
	}
	emit(chat.Answer{Reasoning: strings.Repeat("hm ", n), Content: "answer"})
	return chat.Completion{Answer: chat.Answer{Reasoning: strings.Repeat("hm ", n), Content: "answer"}, Stats: chat.Stats{Finish: "stop", Generated: n + 1}}, nil
}

// A cancelled task leaves no trace in the conversation.
func TestCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	m := &stopper{script: script{answers: []chat.Answer{calls(call("read", map[string]any{"path": "x"})), said("ok")}}, cancel: cancel}
	s, err := New(m, Config{Dir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Run(ctx, "task", Events{}); !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v", err)
	}
	if len(s.Messages()) != 1 {
		t.Errorf("%d messages after the cancelled task", len(s.Messages()))
	}
}

// stopper cancels the context after its first answer.
type stopper struct {
	script
	cancel context.CancelFunc
}

func (s *stopper) Complete(ctx context.Context, msgs []chat.Message, tools []chat.Tool, o chat.Options, smp chat.Sampling, n int, emit func(chat.Answer) bool) (chat.Completion, error) {
	c, err := s.script.Complete(ctx, msgs, tools, o, smp, n, emit)
	s.cancel()
	return c, err
}

func TestDump(t *testing.T) {
	dump := t.TempDir()
	m := &script{answers: []chat.Answer{said("ok")}}
	s, err := New(m, Config{Dir: t.TempDir(), Dump: dump})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Run(context.Background(), "task", Events{}); err != nil {
		t.Fatal(err)
	}
	files, _ := filepath.Glob(filepath.Join(dump, "agent-*-001.json"))
	if len(files) != 1 {
		t.Fatalf("dumped %v", files)
	}
	var d struct {
		Messages  []chat.Message
		Prompt    string
		Generated string
	}
	b, _ := os.ReadFile(files[0])
	if err := json.Unmarshal(b, &d); err != nil || len(d.Messages) != 2 || d.Prompt != "prompt" || d.Generated != "generated" {
		t.Errorf("dump %s: %v", b, err)
	}
}
