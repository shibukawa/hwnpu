package main

import (
	"cmp"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"strings"

	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
	"github.com/shibukawa/hwnpu/internal/mem"
)

// ChatCmd is "hwnpu-chat [chat]": the answer to the prompt given on standard output, or a
// conversation in the terminal without one.
type ChatCmd struct {
	Prompt string   `arg:"" optional:"" help:"What to ask; without it, a conversation in the terminal."`
	Model  string   `short:"m" default:"LFM3-1.7B" placeholder:"NAME|EXPORT" help:"The chat model: a name of hwnpu-chat list, a use or a family picked for the machine's memory (chat, translate, plamo3translate) or with a tag (plamo3translate:2b, chat:16g), a Hugging Face repository OWNER/NAME, or an export's directory (AMD: of tools/amd/iron_export_lm.py)."`
	System string   `placeholder:"TEXT" help:"The system message."`
	N      int      `short:"n" placeholder:"TOKENS" help:"Most tokens of an answer (0: until the cache is full)."`
	Seed   uint64   `placeholder:"N" help:"Random seed of the sampling, the same for every answer (0: a new one for each answer, shown after it)."`
	Image  []string `type:"existingfile" placeholder:"FILE" help:"An image to send with the prompt (a model that reads images: LFM3-VL-2B, LFM3-Omni-2B)."`
	Audio  []string `type:"existingfile" placeholder:"FILE.wav" help:"A clip to send with the prompt (a model that reads audio: LFM3-Audio-2B, LFM3-Omni-2B)."`
	To     string   `placeholder:"LANG" help:"The language a translation model (PLaMo 3 Translate) translates into, a lowercase code: en, ja, zh-hans, ko, ja(easy)... (default: English for a text with Japanese characters, else Japanese)."`
	From   string   `placeholder:"LANG" help:"The language of the texts a translation model translates (default: the model finds it)."`
	Form   string   `enum:"polite,plain," default:"" help:"polite or plain: the style of a translation into Japanese."`
	loadFlags
	convertFlags
	samplingFlags
	encoderFlags
}

// Run loads the model and answers the prompt, or talks.
func (c *ChatCmd) Run(ctx context.Context, stop context.CancelFunc) error {
	if c.Prompt == "" && (len(c.Image) > 0 || len(c.Audio) > 0) {
		return errors.New("--image and --audio go with a prompt (in the terminal: /image FILE, /audio FILE.wav)")
	}
	if err := c.prepare(); err != nil {
		return err
	}
	e, err := loadChat(ctx, c.Model, &c.loadFlags, &c.convertFlags, &c.encoderFlags)
	if err != nil {
		return err
	}
	defer e.Close()
	stop() // Ctrl-C stops an answer, not the chat
	t := &talker{e: e.Engine, think: c.think(e.Engine), thinkBudget: c.ThinkBudget, maxNew: c.N, s: c.sampling(), seed: c.Seed,
		tr: &chat.Translation{TargetLang: c.To, SourceLang: c.From, Form: c.Form}}
	t.reset(c.System)
	if c.Prompt == "" {
		return t.talk(c.System)
	}
	content := chat.Content{Text: c.Prompt}
	for _, f := range c.Image {
		m, err := mediaFile(f, false)
		if err != nil {
			return err
		}
		content.Media = append(content.Media, m)
	}
	for _, f := range c.Audio {
		m, err := mediaFile(f, true)
		if err != nil {
			return err
		}
		content.Media = append(content.Media, m)
	}
	if len(content.Media) > 0 {
		content.Text = strings.Repeat(chat.MediaMarker, len(content.Media)) + content.Text
	}
	return t.answer(content)
}

// talker holds a conversation with a chat model.
type talker struct {
	e           *chat.Engine
	msgs        []chat.Message
	think       bool
	thinkBudget int
	maxNew      int
	s           chat.Sampling
	seed        uint64
	tr          *chat.Translation
}

// reset starts the conversation again, with the system message.
func (t *talker) reset(system string) {
	t.msgs = t.msgs[:0]
	if system != "" {
		t.msgs = append(t.msgs, chat.Message{Role: "system", Content: chat.Content{Text: system}})
	}
}

// answer sends content and writes the answer: the thinking to standard error in dim, the answer
// to standard output, a line of statistics to standard error in gray. On an error the message is
// taken back.
func (t *talker) answer(content chat.Content) error {
	e := t.e
	t.msgs = append(t.msgs, chat.Message{Role: "user", Content: content})
	undo := func(err error) error {
		t.msgs = t.msgs[:len(t.msgs)-1]
		return err
	}
	prompt, thinking, err := e.Template.Render(t.msgs, nil, chat.Options{NoThink: !t.think, Translation: t.tr})
	if err != nil {
		return undo(err)
	}
	var media []chat.Media
	for _, m := range t.msgs {
		media = append(media, m.Content.Media...)
	}
	ids, err := e.Tokens(context.Background(), prompt, media)
	if err != nil {
		return undo(err)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	var sent chat.Stream
	var gen []int32
	thought, answered := false, false
	st, err := e.Generate(ctx, chat.Request{Prompt: ids, MaxTokens: t.maxNew, Sampling: t.s, Seed: t.seed, Think: e.ThinkLimit(t.thinkBudget, thinking)}, func(g []int32) (bool, error) {
		gen = g
		a := e.Template.Parse(e.Tok.Decode(g, false), thinking, nil, false)
		dr, dc := sent.Next(a.Reasoning, a.Content)
		if dr != "" {
			fmt.Fprint(os.Stderr, dimmed(dr))
			thought = true
		}
		if dc != "" {
			if !answered && thought {
				fmt.Fprint(os.Stderr, "\n\n")
			}
			fmt.Print(dc)
			answered = true
		}
		return false, nil
	})
	stop()
	if answered || thought {
		fmt.Println()
	}
	if err != nil && !errors.Is(err, context.Canceled) {
		return undo(err)
	}
	a := e.Template.Parse(e.Tok.Decode(gen, false), thinking, nil, true)
	r := a.Reasoning
	t.msgs = append(t.msgs, chat.Message{Role: "assistant", Content: chat.Content{Text: a.Content}, Reasoning: &r})
	cut := ""
	if st.ThinkCut {
		cut = fmt.Sprintf("; thinking cut at %d tokens", t.thinkBudget)
	}
	if st.ThinkLooped {
		cut = "; thinking cut going round in circles"
	}
	if st.Looped {
		cut += "; stopped repeating itself"
	}
	if st.Drafted > 0 {
		cut += fmt.Sprintf("; %d of %d drafts accepted", st.Accepted, st.Drafted)
	}
	if t.s.Temperature > 0 {
		cut += fmt.Sprintf("; seed %d", st.Seed)
	}
	if errors.Is(err, context.Canceled) {
		cut += "; stopped"
	}
	console.Notef("[prompt %d tokens, %d new in %.1fs; %d tokens at %.1f tokens/s%s]\n",
		st.Prompt, st.Prompt-st.Reused, st.Prefill.Seconds(), st.Generated, float64(st.Generated)/max(st.Decode.Seconds(), 1e-9), cut)
	if os.Getenv("HWNPU_PROFILE") != "" {
		decode.WriteProfile(os.Stderr)
		if b, ok := mem.NPUWired(); ok {
			fmt.Fprintf(os.Stderr, "profile: the Neural Engine holds %.2f GiB\n", float64(b)/(1<<30))
		}
	}
	return nil
}

// talk is the conversation in the terminal.
func (t *talker) talk(system string) error {
	e := t.e
	media := ""
	if e.Images() {
		media += "/image FILE [text] sends an image, "
	}
	if e.Audio() {
		media += "/audio FILE.wav [text] sends a clip, "
	}
	translator := e.Template == chat.Plamo3
	if translator {
		to := cmp.Or(t.tr.TargetLang, "English for Japanese text, else Japanese")
		media += fmt.Sprintf("each text is translated into %s, the earlier ones kept as its history; /to LANG changes the language and clears the history (/to alone: by the text), ", to)
	}
	console.Notef("%s/reset clears the conversation, /think on|off, /exit (or Ctrl-D) quits; Ctrl-C stops an answer, ↑ ↓ go through what was typed\n", media)
	ed := console.NewEditor("chat")
	for {
		text, ok := read(ed)
		if !ok {
			return nil
		}
		switch {
		case text == "":
			continue
		case text == "/exit" || text == "/quit":
			return nil
		case text == "/reset":
			t.reset(system)
			console.Notef("[conversation cleared]\n")
			continue
		case translator && (text == "/to" || strings.HasPrefix(text, "/to ")):
			// the history's translations are labelled with the target language: start again
			t.tr.TargetLang = strings.TrimSpace(strings.TrimPrefix(text, "/to"))
			t.reset(system)
			console.Notef("[translating into %s; history cleared]\n", cmp.Or(t.tr.TargetLang, "the language by the text"))
			continue
		case strings.HasPrefix(text, "/think"):
			t.think = strings.TrimSpace(strings.TrimPrefix(text, "/think")) != "off"
			console.Notef("[thinking %v]\n", t.think)
			continue
		}
		content := chat.Content{Text: text}
		// /image FILE [text], /audio FILE.wav [text]: the image or the clip, then the text
		for _, cmd := range []string{"/image ", "/audio "} {
			rest, ok := strings.CutPrefix(text, cmd)
			if !ok {
				continue
			}
			file, after, _ := strings.Cut(strings.TrimSpace(rest), " ")
			m, err := mediaFile(expandHome(file), cmd == "/audio ")
			if err != nil {
				printError(err)
				content.Text = ""
				break
			}
			content = chat.Content{Text: chat.MediaMarker + strings.TrimSpace(after), Media: []chat.Media{m}}
		}
		if content.Text == "" {
			continue
		}
		if err := t.answer(content); err != nil {
			printError(err)
		}
	}
}

// mediaFile reads an image file as a data URL, or a WAV file as base64.
func mediaFile(path string, isAudio bool) (chat.Media, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return chat.Media{}, err
	}
	if isAudio {
		return chat.Media{Audio: true, Data: base64.StdEncoding.EncodeToString(raw)}, nil
	}
	mime := http.DetectContentType(raw)
	if !strings.HasPrefix(mime, "image/") {
		return chat.Media{}, fmt.Errorf("%s is %s, not an image", path, mime)
	}
	return chat.Media{Data: "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(raw)}, nil
}

// read prompts for and reads the next input with the line editor (a history on the arrows; a line
// ending in \ continues on the next). It returns false at the end of the input.
func read(ed *console.Editor) (string, bool) {
	fmt.Println()
	text, err := ed.Read("> ")
	if err != nil {
		return "", false
	}
	return strings.TrimSpace(text), true
}

// printError writes an error of a turn, after which the conversation goes on, in red on a
// terminal.
func printError(err error) {
	if console.Colors(os.Stderr) {
		fmt.Fprintf(os.Stderr, "\x1b[31merror:\x1b[0m %v\n", err)
		return
	}
	fmt.Fprintln(os.Stderr, "error:", err)
}

// dimmed is s in dim on a terminal (the thinking).
func dimmed(s string) string {
	if !console.Colors(os.Stderr) {
		return s
	}
	return "\x1b[2m" + s + "\x1b[0m"
}
