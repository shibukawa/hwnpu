package main

import (
	"bytes"
	"context"
	"errors"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"regexp"
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
// lines before a download and as /model's columns.
func TestFetchSize(t *testing.T) {
	for _, c := range []struct {
		it             ai.Item
		line, dl, disk string
	}{
		{ai.Item{Model: "m", Ready: true, Disk: 2_280_000_000}, "", "ready", "2.3G"},
		{ai.Item{Model: "m", Download: 7_500_000_000, Disk: 8_100_000_000}, "m is not here yet: about 7.5 GB to download, 8.1 GB kept on the disk", "7.5G", "8.1G"},
		{ai.Item{Model: "m", Download: 22_600_000_000, Disk: 27_110_000_000}, "m is not here yet: about 22.6 GB to download, 27.1 GB kept on the disk", "22.6G", "27.1G"},
		{ai.Item{Model: "m", Download: 830_000_000}, "m is not here yet: about 830.0 MB to download", "830M", "-"},
		{ai.Item{Model: "m", Disk: 2_300_000_000}, "m is not here yet: its checkpoint is here; 2.3 GB kept on the disk once it is made ready", "0", "2.3G"},
		{ai.Item{Model: "m"}, "m is not here yet", "0", "-"},
	} {
		lines := toFetch(&ai.Plan{Items: []ai.Item{c.it, c.it}}) // a model of two uses is said once
		if got := strings.Join(lines, "|"); got != c.line {
			t.Errorf("%+v: %q, want %q", c.it, got, c.line)
		}
		if dl, disk := (modelChoice{item: c.it}).dl(), shortSize(c.it.Disk); dl != c.dl || disk != c.disk {
			t.Errorf("%+v: columns %q %q, want %q %q", c.it, dl, disk, c.dl, c.disk)
		}
	}
	for n, want := range map[int64]string{0: "-", 140_000: "<1M", 31_000_000: "31M", 999_400_000: "999M", 999_600_000: "1.0G", 1_740_000_000: "1.7G"} {
		if got := shortSize(n); got != want {
			t.Errorf("shortSize(%d) = %q, want %q", n, got, want)
		}
	}
}

// /model's list shows every column whole on a terminal of 80 columns, the features included:
// the longest name, sizes and features of models.yaml fit in the 78 the list leaves a row. On a
// terminal of colors the features are tags, each in its color: with a background and a space on
// each side where there is room for those, else as words.
func TestModelTable(t *testing.T) {
	tk := &talker{run: &chatRun{cm: chatModel{base: "LFM3-Omni-2B"}}}
	cs := []modelChoice{
		{info: ai.ModelInfo{Name: "Qwen2.5-Coder-7B-Instruct", Features: []string{"tools"}}, item: ai.Item{MemoryGB: 4.4, Download: 15_200_000_000}},
		{info: ai.ModelInfo{Name: "Qwen3.6-35B-A3B", Features: []string{"tools", "thinking"}}, item: ai.Item{MemoryGB: 26.3, Download: 22_600_000_000, Disk: 27_110_000_000}},
		{info: ai.ModelInfo{Name: "LFM3-Omni-2B", Features: []string{"tools", "thinking", "vision", "audio"}}, item: ai.Item{MemoryGB: 3.0, Ready: true, Disk: 3_200_000_000}},
		{info: ai.ModelInfo{Name: "plamo-3-translate-0.8b", Features: []string{"translation"}, MemoryGB: 1.0}, err: errors.New("ai: the model policy refuses it")},
	}
	want := []string{
		" MODEL                      MEM    DL     DISK   FEATURES",
		" Qwen2.5-Coder-7B-Instruct  4.4G   15.2G  -      tools",
		" Qwen3.6-35B-A3B            26.3G  22.6G  27.1G  tools,thinking",
		"*LFM3-Omni-2B               3.0G   ready  3.2G   tools,thinking,vision,audio",
		" plamo-3-translate-0.8b     1.0G   -      -      not to be had: the model policy refuses it",
	}
	sgr := regexp.MustCompile("\x1b\\[[0-9;]*m")
	check := func(what string, head string, rows []string, want []string) {
		t.Helper()
		if got := append([]string{head}, rows...); !slices.Equal(got, want) {
			t.Errorf("%s:\n%s\nwant:\n%s", what, strings.Join(got, "\n"), strings.Join(want, "\n"))
		}
	}

	// without colors: as models.yaml writes the features, and nothing painted
	head, rows, paint := tk.table(cs, 80, false)
	check("the table", head, rows, want)
	for _, r := range rows[:3] {
		if len(r) > 78 {
			t.Errorf("a row of %d columns, over the 78 of a terminal of 80: %q", len(r), r)
		}
		if paint(r) != r {
			t.Errorf("painted without colors: %q", paint(r))
		}
	}

	// colors on 80 columns: the same rows, the features' words in their colors
	head, rows, paint = tk.table(cs, 80, true)
	check("the table of 80 columns", head, rows, want)
	const left = "*LFM3-Omni-2B               3.0G   ready  3.2G   "
	if got, want := paint(rows[2]), left+"\x1b[36mtools\x1b[39m \x1b[35mthinking\x1b[39m \x1b[32mvision\x1b[39m \x1b[33maudio\x1b[39m"; got != want {
		t.Errorf("the words:\n%q\nwant\n%q", got, want)
	}
	if got := paint(" Qwen3.6-35B-A3B            26.3G  22.6G  27.1G  tools,thin…"); !strings.HasSuffix(got, "\x1b[36mtools\x1b[39m \x1b[35mthin…\x1b[39m") {
		t.Errorf("a row the terminal cut: %q", got)
	}
	if got := paint(rows[3]); !strings.HasSuffix(got, "\x1b[31mnot to be had: the model policy refuses it\x1b[39m") {
		t.Errorf("a model not to be had: %q", got)
	}

	// colors and room: tags, a space on each side and one between them
	head, rows, paint = tk.table(cs, 100, true)
	check("the table of 100 columns", head, rows, []string{
		" MODEL                      MEM    DL     DISK    FEATURES",
		" Qwen2.5-Coder-7B-Instruct  4.4G   15.2G  -       tools ",
		" Qwen3.6-35B-A3B            26.3G  22.6G  27.1G   tools   thinking ",
		"*LFM3-Omni-2B               3.0G   ready  3.2G    tools   thinking   vision   audio ",
		" plamo-3-translate-0.8b     1.0G   -      -      not to be had: the model policy refuses it",
	})
	if got, want := paint(rows[2]), left+"\x1b[30;46m tools \x1b[39;49m \x1b[30;45m thinking \x1b[39;49m \x1b[30;42m vision \x1b[39;49m \x1b[30;43m audio \x1b[39;49m"; got != want {
		t.Errorf("the tags:\n%q\nwant\n%q", got, want)
	}
	for _, r := range rows {
		// the paint adds no column, and ends no mark of the list's (no reset)
		if got := paint(r); sgr.ReplaceAllString(got, "") != r && !strings.Contains(r, ",") || strings.Contains(got, "\x1b[0m") {
			t.Errorf("painted %q of %q", got, r)
		}
	}
	// the tags do not fit beside the list's mark below 87 columns
	if _, rows, _ = tk.table(cs, 86, true); rows[2] != want[3] {
		t.Errorf("86 columns: %q", rows[2])
	}
	if _, rows, _ = tk.table(cs, 87, true); !strings.HasSuffix(rows[2], " tools   thinking   vision   audio ") {
		t.Errorf("87 columns: %q", rows[2])
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

// /image and /audio send the file with the text after it, to a model that reads them; a model
// that does not, a file that is not there and a file that is no image send nothing.
func TestMediaCommands(t *testing.T) {
	dir := t.TempDir()
	var pic bytes.Buffer
	if err := png.Encode(&pic, image.NewRGBA(image.Rect(0, 0, 2, 2))); err != nil {
		t.Fatal(err)
	}
	photo, clip, note := filepath.Join(dir, "photo.png"), filepath.Join(dir, "clip.wav"), filepath.Join(dir, "note.txt")
	for path, data := range map[string][]byte{photo: pic.Bytes(), clip: []byte("RIFF....WAVEfmt "), note: []byte("only text")} {
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	cat := &ai.Catalog{Models: []ai.ModelInfo{{Name: "LFM3-VL-2B", Kind: "chat", Features: []string{"vision"}}}}
	f := &fakeChat{images: true, audio: true}
	tk := &talker{c: f, cat: cat}
	tk.reset()
	cs := tk.commands()
	last := func() text.Message { msgs := f.got[len(f.got)-1]; return msgs[len(msgs)-1] }

	if handled, _ := cs.do("/image " + photo + " what is in it?"); !handled || len(f.got) != 1 {
		t.Fatalf("/image: handled %v, %d messages sent", handled, len(f.got))
	}
	if m := last(); m.Text != "what is in it?" || len(m.Media) != 1 || m.Media[0].Audio || !bytes.Equal(m.Media[0].Data, pic.Bytes()) {
		t.Errorf("/image sent %q with %d media", m.Text, len(m.Media))
	}
	cs.do("/audio " + clip)
	if m := last(); len(f.got) != 2 || m.Text != "" || len(m.Media) != 1 || !m.Media[0].Audio {
		t.Errorf("/audio sent %q with %d media (%d messages)", m.Text, len(m.Media), len(f.got))
	}
	// /retry sends the clip again
	tk.retry()
	if m := last(); len(f.got) != 3 || len(m.Media) != 1 || !m.Media[0].Audio {
		t.Errorf("/retry after /audio: %d messages, %d media", len(f.got), len(m.Media))
	}
	for _, line := range []string{"/image", "/image " + filepath.Join(dir, "none.png") + " hello", "/image " + note + " hello"} {
		if handled, _ := cs.do(line); !handled || len(f.got) != 3 {
			t.Errorf("%q: handled %v, %d messages sent", line, handled, len(f.got))
		}
	}
	// a model that reads neither: said so, with the models that do, and nothing sent
	g := &fakeChat{}
	tk.c = g
	for _, line := range []string{"/image " + photo + " hello", "/audio " + clip} {
		if handled, _ := cs.do(line); !handled || len(g.got) != 0 {
			t.Errorf("%q to a model that reads no media: handled %v, %d messages sent", line, handled, len(g.got))
		}
	}
}
