package main

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/text"
)

// A line calls a command when it starts with / and a name in lowercase letters; a path does not.
func TestCommandOf(t *testing.T) {
	for _, c := range []struct {
		line, name, arg string
		ok              bool
	}{
		{"/model", "model", "", true},
		{"/model  gemma4:e4b ", "model", "gemma4:e4b", true},
		{"/system You are terse.\nAnswer in a line.", "system", "You are terse.\nAnswer in a line.", true},
		{"/?", "?", "", true},
		{"/tmp is full", "tmp", "is full", true}, // a name: do sends it as a message, not being near a command
		{"/usr/bin/env is a path", "", "", false},
		{"/Users/me/x.go has a bug", "", "", false},
		{"/", "", "", false},
		{"/ model", "", "", false},
		{"/モデル", "", "", false},
		{"hello /model", "", "", false},
	} {
		name, arg, ok := commandOf(c.line)
		if name != c.name || arg != c.arg || ok != c.ok {
			t.Errorf("%q: %q %q %v, want %q %q %v", c.line, name, arg, ok, c.name, c.arg, c.ok)
		}
	}
}

// do runs the command of any of its names, says so for a slip of the keys and for a command
// that is not for the model, and leaves every other line as a message.
func TestCommandsDo(t *testing.T) {
	var ran []string
	not := ""
	cs := commands{
		{names: []string{"model", "models"}, args: "[NAME]", run: func(arg string) bool { ran = append(ran, "model "+arg); return false }},
		{names: []string{"reset", "clear", "new"}, run: func(string) bool { ran = append(ran, "reset"); return false }},
		{names: []string{"to"}, unless: func() string { return not }, run: func(arg string) bool { ran = append(ran, "to "+arg); return false }},
		quitCommand(),
	}
	for _, c := range []struct {
		line          string
		handled, quit bool
		ran           string
	}{
		{"/model", true, false, "model "},
		{"/models qwen3.5", true, false, "model qwen3.5"},
		{"/clear", true, false, "reset"},
		{"/new", true, false, "reset"},
		{"/to en", true, false, "to en"},
		{"/exit", true, true, ""},
		{"/quit", true, true, ""},
		{"/bye", true, true, ""},
		{"/help", true, false, ""},
		{"/?", true, false, ""},
		{"/modle", true, false, ""},  // a slip of /model: not sent to the model
		{"/rest", true, false, ""},   // of /reset
		{"/hlep", true, false, ""},   // of /help
		{"/exti", true, false, ""},   // of /exit
		{"/hello", false, false, ""}, // two from /help: a message
		{"/static", false, false, ""},
		{"/tmp is full", false, false, ""},
		{"/dev/null", false, false, ""},
		{"/sys", false, false, ""},
		{"/etc", false, false, ""},
		{"hello", false, false, ""},
	} {
		ran = nil
		handled, quit := cs.do(c.line)
		if handled != c.handled || quit != c.quit || strings.Join(ran, ",") != c.ran {
			t.Errorf("%q: handled %v, quit %v, ran %q; want %v, %v, %q", c.line, handled, quit, ran, c.handled, c.quit, c.ran)
		}
	}
	// a command that is not for the model at hand is said so, and not run
	not, ran = "not a translation model", nil
	if handled, _ := cs.do("/to en"); !handled || ran != nil {
		t.Errorf("/to for a model that does not translate: handled %v, ran %q", handled, ran)
	}
}

// fakeChat is a chat model that answers each conversation with the number of its messages.
type fakeChat struct {
	images, audio bool
	fail          error
	got           [][]text.Message
}

func (f *fakeChat) Reply(_ context.Context, msgs []text.Message, _ text.ChatOptions, fn func(text.Delta) bool) (text.Answer, error) {
	f.got = append(f.got, slices.Clone(msgs))
	if f.fail != nil {
		return text.Answer{}, f.fail
	}
	return text.Answer{Text: "answer"}, nil
}
func (f *fakeChat) Count(context.Context, []text.Message, text.ChatOptions) (int, error) {
	return 0, nil
}
func (f *fakeChat) Context() int     { return 4096 }
func (f *fakeChat) Images() bool     { return f.images }
func (f *fakeChat) Audio() bool      { return f.audio }
func (f *fakeChat) Translates() bool { return false }

func roles(msgs []text.Message) string {
	var r []string
	for _, m := range msgs {
		r = append(r, m.Role+":"+m.Text)
	}
	return strings.Join(r, " ")
}

// /retry takes back what the last message added and sends it again, also when its answer
// failed; /reset and /system forget it; after /model the same message goes to the new model.
func TestRetry(t *testing.T) {
	f := &fakeChat{}
	tk := &talker{c: f, system: "be terse"}
	tk.reset()
	tk.retry() // nothing sent yet
	if len(f.got) != 0 {
		t.Fatalf("retry before a message sent %v", f.got)
	}
	tk.say(text.Message{Role: "user", Text: "one"})
	tk.say(text.Message{Role: "user", Text: "two"})
	tk.retry()
	if got, want := roles(f.got[2]), "system:be terse user:one assistant:answer user:two"; got != want {
		t.Errorf("retry sent %q, want %q", got, want)
	}
	if got, want := roles(tk.msgs), "system:be terse user:one assistant:answer user:two assistant:answer"; got != want {
		t.Errorf("after retry: %q, want %q", got, want)
	}
	// a message whose answer failed is not in the conversation, and is the one sent again
	f.fail = errors.New("the NPU is busy")
	tk.say(text.Message{Role: "user", Text: "three"})
	f.fail = nil
	tk.retry()
	if got, want := roles(tk.msgs), "system:be terse user:one assistant:answer user:two assistant:answer user:three assistant:answer"; got != want {
		t.Errorf("after a failed answer and retry: %q, want %q", got, want)
	}
	// another model, a new conversation (restart): the last message is kept to ask again
	g := &fakeChat{}
	tk.c = g
	tk.restart()
	tk.retry()
	if got, want := roles(g.got[0]), "system:be terse user:three"; got != want {
		t.Errorf("retry after restart sent %q, want %q", got, want)
	}
	tk.reset()
	tk.retry()
	if len(g.got) != 1 {
		t.Errorf("retry after /reset sent %q", roles(g.got[len(g.got)-1]))
	}
}

// A conversation with an image goes on only with a model that reads images.
func TestReads(t *testing.T) {
	tk := &talker{c: &fakeChat{}}
	tk.msgs = []text.Message{{Role: "user", Text: "hello"}}
	if !tk.reads() {
		t.Error("a conversation of texts is not read")
	}
	tk.msgs = append(tk.msgs, text.Message{Role: "user", Text: "what is it?", Media: []text.Media{{Data: []byte("png")}}})
	if tk.reads() {
		t.Error("an image is read by a model without vision")
	}
	tk.c = &fakeChat{images: true}
	if !tk.reads() {
		t.Error("an image is not read by a model with vision")
	}
	tk.msgs = append(tk.msgs, text.Message{Role: "user", Media: []text.Media{{Audio: true, Data: []byte("wav")}}})
	if tk.reads() {
		t.Error("a clip is read by a model without audio")
	}
}

// What a model that is not here takes: its download and what it keeps on the disk, as the
// lines before a download and as /model's column.
func TestFetchSize(t *testing.T) {
	for _, c := range []struct {
		it          ai.Item
		line, state string
	}{
		{ai.Item{Model: "m", Ready: true}, "", "yes"},
		{ai.Item{Model: "m", Download: 7_500_000_000, Disk: 8_100_000_000}, "m is not here yet: about 7.5 GB to download, 8.1 GB kept on the disk", "no: 7.5 GB download, 8.1 GB disk"},
		{ai.Item{Model: "m", Download: 830_000_000}, "m is not here yet: about 830.0 MB to download", "no: 830.0 MB download"},
		{ai.Item{Model: "m", Disk: 2_300_000_000}, "m is not here yet: its checkpoint is here; 2.3 GB kept on the disk once it is made ready", "no: 2.3 GB disk (checkpoint here)"},
		{ai.Item{Model: "m"}, "m is not here yet", "no"},
	} {
		lines := toFetch(&ai.Plan{Items: []ai.Item{c.it, c.it}}) // a model of two uses is said once
		if got := strings.Join(lines, "|"); got != c.line {
			t.Errorf("%+v: %q, want %q", c.it, got, c.line)
		}
		if got := (modelChoice{item: c.it}).state(); got != c.state {
			t.Errorf("%+v: state %q, want %q", c.it, got, c.state)
		}
	}
	if got := (modelChoice{err: errors.New("ai: the model policy refuses it")}).state(); got != "no: the model policy refuses it" {
		t.Errorf("a model that cannot be planned: %q", got)
	}
}

// A model /model names runs with the command line's flags, without the files made by hand for
// the model the command line named; that model takes them again.
func TestFlagsFor(t *testing.T) {
	c := &ChatCmd{Model: "LFM3-1.7B"}
	c.Context, c.Package, c.Tokenizer, c.Vision = 4096, "a.mlpackage", "tokenizer.json", "v.mlpackage"
	cf, ef := c.flagsFor("gemma-4-E2B-it")
	if cf.Context != 4096 || cf.Package != "" || cf.Tokenizer != "" || ef.Vision != "" {
		t.Errorf("another model: %+v %+v", cf, ef)
	}
	if c.Package != "a.mlpackage" {
		t.Error("the command's flags were changed")
	}
	cf, ef = c.flagsFor("lfm3-1.7b")
	if cf.Package != "a.mlpackage" || cf.Tokenizer != "tokenizer.json" || ef.Vision != "v.mlpackage" {
		t.Errorf("the command line's model: %+v %+v", cf, ef)
	}
}
