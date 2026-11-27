package main

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"strings"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/text"
	"github.com/shibukawa/hwnpu/internal/chatcmd"
	"github.com/shibukawa/hwnpu/internal/console"
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
	st, cat, err := c.loadFlags.prepare(ctx)
	if err != nil {
		return err
	}
	cm, err := chatModelOf(st, cat, c.Model, &c.convertFlags, &c.encoderFlags)
	if err != nil {
		return err
	}
	st.Definitions = cm.defs
	var use ai.Requirement = cm.chat
	if cm.translate != nil {
		use = *cm.translate // a translation model translates each text, with the chat's turns as its history
	}
	app := ai.NewApp("hwnpu-chat", use)
	app.Store = st
	rt, err := open(ctx, app, options(&c.loadFlags, &c.convertFlags, "chat")...)
	if err != nil {
		return err
	}
	defer rt.Close()
	t := &talker{maxNew: c.N, s: c.sampling(), seed: c.Seed, thinkBudget: c.ThinkBudget,
		tr: text.TranslateOptions{To: c.To, From: c.From, Form: c.Form}}
	if cm.translate != nil {
		if t.tl, err = cm.translate.Get(ctx, rt); err != nil {
			return err
		}
	} else {
		cchat, err := cm.chat.Get(ctx, rt)
		if err != nil {
			return err
		}
		t.c = cchat
		t.think = c.think(cchat)
	}
	prepared(rt)
	cm.keep(ctx)
	stop() // Ctrl-C stops an answer, not the chat
	return c.converse(t)
}

func (c *ChatCmd) converse(t *talker) error {
	t.reset(c.System)
	if c.Prompt == "" {
		return t.talk(c.System)
	}
	console.Running() // the times of a conversation in the terminal are not told
	msg := text.Message{Role: "user", Text: c.Prompt}
	for _, f := range c.Image {
		m, err := mediaFile(f, false)
		if err != nil {
			return err
		}
		msg.Media = append(msg.Media, m)
	}
	for _, f := range c.Audio {
		m, err := mediaFile(f, true)
		if err != nil {
			return err
		}
		msg.Media = append(msg.Media, m)
	}
	return t.answer(msg)
}

// talker holds a conversation with a chat model, or a translation model's history.
type chatter interface {
	Reply(context.Context, []text.Message, text.ChatOptions, func(text.Delta) bool) (text.Answer, error)
	Images() bool
	Audio() bool
	Translates() bool
}

type talker struct {
	c           chatter
	tl          *text.Translator // a translation model, in place of c
	msgs        []text.Message
	pairs       []text.Pair // the translations so far (tl)
	think       bool
	thinkBudget int
	maxNew      int
	s           text.Sampling
	seed        uint64
	tr          text.TranslateOptions
}

// reset starts the conversation again, with the system message.
func (t *talker) reset(system string) {
	t.msgs, t.pairs = t.msgs[:0], nil
	if system != "" {
		t.msgs = append(t.msgs, text.Message{Role: "system", Text: system})
	}
}

// answer sends a message and writes the answer: the thinking to standard error in dim, the
// answer to standard output, a line of statistics to standard error in gray. On an error the
// message is taken back.
func (t *talker) answer(msg text.Message) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	thought, answered := false, false
	show := func(d text.Delta) bool {
		if d.Reasoning != "" {
			fmt.Fprint(os.Stderr, dimmed(d.Reasoning))
			thought = true
		}
		if d.Text != "" {
			if !answered && thought {
				fmt.Fprint(os.Stderr, "\n\n")
			}
			fmt.Print(d.Text)
			answered = true
		}
		return true
	}
	maxNew := t.maxNew
	if maxNew <= 0 {
		maxNew = text.Unlimited
	}
	var st text.Stats
	var err error
	if t.tl != nil {
		o := t.tr
		o.History, o.MaxTokens, o.Sampling, o.Seed = t.pairs, maxNew, &t.s, t.seed
		var tr text.Translation
		tr, err = t.tl.Reply(ctx, msg.Text, o, show)
		st = tr.Stats
		if err == nil || errors.Is(err, context.Canceled) {
			t.pairs = append(t.pairs, text.Pair{Source: msg.Text, Target: tr.Text})
		}
	} else {
		var a text.Answer
		msgs := append(t.msgs, msg)
		o := text.ChatOptions{MaxTokens: maxNew, Think: &t.think, ThinkBudget: samplingFlags{ThinkBudget: t.thinkBudget}.budget(),
			Sampling: &t.s, Seed: t.seed}
		if t.c.Translates() {
			o.Translation = &t.tr
		}
		a, err = t.c.Reply(ctx, msgs, o, show)
		st = a.Stats
		if err == nil || errors.Is(err, context.Canceled) {
			t.msgs = append(msgs, text.Message{Role: "assistant", Text: a.Text, Reasoning: a.Reasoning})
		}
	}
	stop()
	if answered || thought {
		fmt.Println()
	}
	if err != nil && !errors.Is(err, context.Canceled) {
		return err
	}
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
	if t.s.Temperature > 0 && st.Seed != 0 {
		cut += fmt.Sprintf("; seed %d", st.Seed)
	}
	if errors.Is(err, context.Canceled) {
		cut += "; stopped"
	}
	console.Notef("[prompt %d tokens, %d new in %.1fs; %d tokens at %.1f tokens/s%s]\n",
		st.Prompt, st.Prompt-st.Reused, st.Prefill.Seconds(), st.Generated, st.TokensPerSecond(), cut)
	if console.Profile {
		// the steps' profile is of the process that ran them: hwnpud's (chatcmd.Profile)
		if err := chatcmd.Profile(context.Background(), os.Stderr); err != nil {
			fmt.Fprintln(os.Stderr, "profile:", err)
		}
		if b, ok := mem.NPUWired(); ok {
			fmt.Fprintf(os.Stderr, "profile: the Neural Engine holds %.2f GiB\n", float64(b)/(1<<30))
		}
		if pm, ok := mem.ProcessMemory(); ok {
			fmt.Fprintf(os.Stderr, "profile: process footprint %.2f GiB (peak %.2f GiB, external %.2f GiB, neural_nofp %.2f GiB)\n",
				float64(pm.PhysFootprint)/(1<<30), float64(pm.PeakFootprint)/(1<<30),
				float64(pm.External)/(1<<30), float64(pm.NeuralNoFootprint)/(1<<30))
		}
	}
	return nil
}

// talk is the conversation in the terminal.
func (t *talker) talk(system string) error {
	media := ""
	if t.c != nil && t.c.Images() {
		media += "/image FILE [text] sends an image, "
	}
	if t.c != nil && t.c.Audio() {
		media += "/audio FILE.wav [text] sends a clip, "
	}
	translator := t.tl != nil || (t.c != nil && t.c.Translates())
	if translator {
		to := cmp.Or(t.tr.To, "English for Japanese text, else Japanese")
		media += fmt.Sprintf("each text is translated into %s, the earlier ones kept as its history; /to LANG changes the language and clears the history (/to alone: by the text), ", to)
	}
	console.Notef("%s/reset clears the conversation, /think on|off, /exit (or Ctrl-D) quits; Ctrl-C stops an answer, ↑ ↓ go through what was typed\n", media)
	ed := console.NewEditor("chat")
	for {
		line, ok := read(ed)
		if !ok {
			return nil
		}
		switch {
		case line == "":
			continue
		case line == "/exit" || line == "/quit":
			return nil
		case line == "/reset":
			t.reset(system)
			console.Notef("[conversation cleared]\n")
			continue
		case translator && (line == "/to" || strings.HasPrefix(line, "/to ")):
			// the history's translations are labelled with the target language: start again
			t.tr.To = strings.TrimSpace(strings.TrimPrefix(line, "/to"))
			t.reset(system)
			console.Notef("[translating into %s; history cleared]\n", cmp.Or(t.tr.To, "the language by the text"))
			continue
		case strings.HasPrefix(line, "/think"):
			t.think = strings.TrimSpace(strings.TrimPrefix(line, "/think")) != "off"
			console.Notef("[thinking %v]\n", t.think)
			continue
		}
		msg := text.Message{Role: "user", Text: line}
		// /image FILE [text], /audio FILE.wav [text]: the image or the clip, then the text
		for _, cmd := range []string{"/image ", "/audio "} {
			rest, ok := strings.CutPrefix(line, cmd)
			if !ok {
				continue
			}
			file, after, _ := strings.Cut(strings.TrimSpace(rest), " ")
			m, err := mediaFile(expandHome(file), cmd == "/audio ")
			if err != nil {
				printError(err)
				msg.Text = ""
				break
			}
			msg = text.Message{Role: "user", Text: strings.TrimSpace(after), Media: []text.Media{m}}
		}
		if msg.Text == "" && len(msg.Media) == 0 {
			continue
		}
		if err := t.answer(msg); err != nil {
			printError(err)
		}
	}
}

// mediaFile reads an image or a WAV file.
func mediaFile(path string, isAudio bool) (text.Media, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return text.Media{}, err
	}
	if isAudio {
		return text.Media{Audio: true, Data: raw}, nil
	}
	if mime := http.DetectContentType(raw); !strings.HasPrefix(mime, "image/") {
		return text.Media{}, fmt.Errorf("%s is %s, not an image", path, mime)
	}
	return text.Media{Data: raw}, nil
}

// read prompts for and reads the next input with the line editor (a history on the arrows; a line
// ending in \ continues on the next). It returns false at the end of the input.
func read(ed *console.Editor) (string, bool) {
	fmt.Println()
	line, err := ed.Read("> ")
	if err != nil {
		return "", false
	}
	return strings.TrimSpace(line), true
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
