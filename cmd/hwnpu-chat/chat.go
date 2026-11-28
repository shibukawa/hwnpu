package main

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"slices"
	"strings"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/text"
	"github.com/shibukawa/hwnpu/internal/catalog"
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
	run, err := c.start(ctx, cm, said, nil)
	if err != nil {
		return err
	}
	t := &talker{cmd: c, st: st, cat: cat, maxNew: c.N, s: c.sampling(), seed: c.Seed, thinkGiven: c.Think, thinkBudget: c.ThinkBudget,
		system: c.System, tr: text.TranslateOptions{To: c.To, From: c.From, Form: c.Form}}
	t.use(run)
	defer func() { t.run.close() }() // the model at the end, which /model may have changed
	prepared(run.rt)
	cm.keep(ctx)
	stop() // Ctrl-C stops an answer, not the chat
	return c.converse(t)
}

// chatRun is a chat model loaded: the model --model or /model named, its Runtime and what
// answers.
type chatRun struct {
	cm chatModel
	rt *ai.Runtime
	c  *text.Chatter    // nil for a translation model
	tl *text.Translator // a translation model, in place of c
}

// close unloads the model (nothing for none).
func (r *chatRun) close() {
	if r != nil && r.rt != nil {
		r.rt.Close()
	}
}

// start makes the model cm ready and loads it. before is called when it is not here, before
// it is downloaded and converted (its error ends start: openAfter); ready, when not nil, once
// it is ready, before it is loaded.
func (c *ChatCmd) start(ctx context.Context, cm chatModel, before func(*ai.Plan) error, ready func()) (*chatRun, error) {
	rt, err := openAfter(ctx, cm.app(), before, ready, options(&c.loadFlags, &c.convertFlags, "chat")...)
	if err != nil {
		return nil, err
	}
	r := &chatRun{cm: cm, rt: rt}
	if cm.translate != nil {
		r.tl, err = cm.translate.Get(ctx, rt)
	} else {
		r.c, err = cm.chat.Get(ctx, rt)
	}
	if err != nil {
		rt.Close()
		return nil, err
	}
	return r, nil
}

// flagsFor are the flags the model /model names runs with: the command line's, without the
// files made by hand for the model it named (--package, --config, --weights, --tokenizer,
// --codes, --format-for and the encoders), which are that model's alone. Named as on the
// command line, the model takes them all again.
func (c *ChatCmd) flagsFor(name string) (*convertFlags, *encoderFlags) {
	if strings.EqualFold(strings.TrimSpace(name), strings.TrimSpace(c.Model)) {
		return &c.convertFlags, &c.encoderFlags
	}
	cf := c.convertFlags
	cf.Package, cf.Config, cf.Weights, cf.Tokenizer, cf.Codes, cf.FormatFor = "", "", "", "", "", ""
	return &cf, &encoderFlags{}
}

func (c *ChatCmd) converse(t *talker) error {
	t.reset()
	if c.Prompt == "" {
		return t.talk()
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
	Count(context.Context, []text.Message, text.ChatOptions) (int, error)
	Context() int
	Images() bool
	Audio() bool
	Translates() bool
}

type talker struct {
	// cmd is the command, st and cat where its models come from: what /model loads another
	// with. run is the model loaded (nil when /model lost it), c and tl what answers of it.
	cmd *ChatCmd
	st  ai.Store
	cat *ai.Catalog
	run *chatRun
	ed  *console.Editor // the terminal's (nil with a prompt on the command line)

	c      chatter
	tl     *text.Translator // a translation model, in place of c
	system string           // the system message ("" for none)
	msgs   []text.Message
	pairs  []text.Pair // the translations so far (tl)
	// last is the message sent last and lastAt how long the conversation (msgs, or pairs) was
	// before it, for /retry; sent is false when there is none.
	last   text.Message
	lastAt int
	sent   bool

	think       bool
	thinkGiven  *bool // --think or /think, which the next model keeps (nil: each model's own)
	thinkBudget int
	maxNew      int
	s           text.Sampling
	seed        uint64
	tr          text.TranslateOptions
}

// use makes r the model that answers (nil: none).
func (t *talker) use(r *chatRun) {
	t.run, t.c, t.tl = r, nil, nil
	if r == nil {
		return
	}
	t.tl = r.tl
	if r.c != nil {
		t.c = r.c
		t.think = r.c.Thinks() // unless its definition turns the thinking off
	}
	if t.thinkGiven != nil {
		t.think = *t.thinkGiven
	}
}

// name is the name of the model that answers: its name in the definitions, or its export's.
func (t *talker) name() string {
	if t.run == nil {
		return "no model"
	}
	return cmp.Or(t.run.cm.export, t.run.cm.base)
}

// at reports whether the model of the definitions named name is the one that answers.
func (t *talker) at(name string) bool {
	return t.run != nil && t.run.cm.export == "" && strings.EqualFold(name, t.run.cm.base)
}

// translator reports whether the model translates each text.
func (t *talker) translator() bool {
	return t.tl != nil || (t.c != nil && t.c.Translates())
}

// reset starts the conversation again, with the system message.
func (t *talker) reset() {
	t.msgs, t.pairs, t.sent = t.msgs[:0], nil, false
	if t.system != "" {
		t.msgs = append(t.msgs, text.Message{Role: "system", Text: t.system})
	}
}

// options are the options of an answer of a chat model.
func (t *talker) options() text.ChatOptions {
	maxNew := t.maxNew
	if maxNew <= 0 {
		maxNew = text.Unlimited
	}
	o := text.ChatOptions{MaxTokens: maxNew, Think: &t.think, ThinkBudget: samplingFlags{ThinkBudget: t.thinkBudget}.budget(),
		Sampling: &t.s, Seed: t.seed}
	if t.c != nil && t.c.Translates() {
		o.Translation = &t.tr
	}
	return o
}

// answer sends a message and writes the answer: the thinking to standard error in dim, the
// answer to standard output, a line of statistics to standard error in gray. On an error the
// message is taken back.
func (t *talker) answer(msg text.Message) error {
	if t.c == nil && t.tl == nil {
		return errors.New("no model is loaded (/model NAME loads one)")
	}
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
	var st text.Stats
	var err error
	t.last, t.sent = msg, true
	if t.tl != nil {
		t.lastAt = len(t.pairs)
		o := t.tr
		o.History, o.MaxTokens, o.Sampling, o.Seed = t.pairs, t.options().MaxTokens, &t.s, t.seed
		var tr text.Translation
		tr, err = t.tl.Reply(ctx, msg.Text, o, show)
		st = tr.Stats
		if err == nil || errors.Is(err, context.Canceled) {
			t.pairs = append(t.pairs, text.Pair{Source: msg.Text, Target: tr.Text})
		}
	} else {
		t.lastAt = len(t.msgs)
		var a text.Answer
		msgs := append(t.msgs, msg)
		a, err = t.c.Reply(ctx, msgs, t.options(), show)
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

// retry answers the message sent last again: what it added to the conversation is taken back
// first. After /model it is the same question asked of another model.
func (t *talker) retry() {
	if !t.sent {
		console.Notef("[nothing to answer again yet]\n")
		return
	}
	if t.tl != nil {
		t.pairs = t.pairs[:min(t.lastAt, len(t.pairs))]
	} else {
		t.msgs = t.msgs[:min(t.lastAt, len(t.msgs))]
	}
	if err := t.answer(t.last); err != nil {
		printError(err)
	}
}

// hints says what the model at hand takes besides texts, for the notes of the start and of
// /model: the images and clips it reads, the language it translates into.
func (t *talker) hints() string {
	h := ""
	if t.c != nil && t.c.Images() {
		h += "/image FILE [text] sends an image, "
	}
	if t.c != nil && t.c.Audio() {
		h += "/audio FILE.wav [text] sends a clip, "
	}
	if t.translator() {
		to := cmp.Or(t.tr.To, "English for Japanese text, else Japanese")
		h += fmt.Sprintf("each text is translated into %s, the earlier ones kept as its history; /to LANG changes the language and clears the history (/to alone: by the text), ", to)
	}
	return h
}

// commands are the commands of the conversation in the terminal (/help lists them).
func (t *talker) commands() commands {
	reads := func(feature, what string, has func() bool) func() string {
		return func() string {
			if has() {
				return ""
			}
			why := fmt.Sprintf("%s does not read %s", t.name(), what)
			var can []string
			for _, m := range t.cat.OfKind(string(catalog.KindChat)) {
				if slices.Contains(m.Features, feature) {
					can = append(can, m.Name)
				}
			}
			if len(can) > 0 {
				why += " (/model picks one that does: " + strings.Join(can, ", ") + ")"
			}
			return why
		}
	}
	media := func(isAudio bool) func(string) bool {
		return func(arg string) bool {
			file, after, _ := strings.Cut(arg, " ")
			if file == "" {
				printError(errors.New("a file to send: /image FILE [text], /audio FILE.wav [text]"))
				return false
			}
			m, err := mediaFile(expandHome(file), isAudio)
			if err != nil {
				printError(err)
				return false
			}
			t.say(text.Message{Role: "user", Text: strings.TrimSpace(after), Media: []text.Media{m}})
			return false
		}
	}
	return commands{
		{names: []string{"model", "models"}, args: "[NAME]", about: "choose the model from the list of them, or load NAME (a name, a family or a use, as --model); the conversation goes on with it",
			run: func(arg string) bool { t.model(arg); return false }},
		{names: []string{"reset", "clear", "new"}, about: "clear the conversation", run: func(string) bool {
			t.reset()
			console.Notef("[conversation cleared]\n")
			return false
		}},
		{names: []string{"retry", "regen"}, about: "answer the last message again (after /model: by the other model)", run: func(string) bool { t.retry(); return false }},
		{names: []string{"system"}, args: "[TEXT|none]", about: "show the system message, or set it and clear the conversation (none: without one)",
			unless: func() string {
				if t.tl != nil {
					return t.name() + " is a translation model, which takes no system message"
				}
				return ""
			},
			run: func(arg string) bool {
				switch arg {
				case "":
					if t.system == "" {
						console.Notef("[no system message]\n")
					} else {
						console.Notef("[system message: %s]\n", t.system)
					}
					return false
				case "none":
					arg = ""
				}
				t.system = arg
				t.reset()
				if arg == "" {
					console.Notef("[no system message; conversation cleared]\n")
				} else {
					console.Notef("[system message set; conversation cleared]\n")
				}
				return false
			}},
		{names: []string{"think"}, args: "on|off", about: "let the model think before it answers, or not", run: func(arg string) bool {
			t.think = arg != "off"
			t.thinkGiven = &t.think
			console.Notef("[thinking %s]\n", onOff(t.think))
			return false
		}},
		{names: []string{"to"}, args: "[LANG]", about: "the language to translate into, clearing the history (alone: by the text)",
			unless: func() string {
				if !t.translator() {
					return t.name() + " is not a translation model (/model translate picks one)"
				}
				return ""
			},
			run: func(arg string) bool {
				// the history's translations are labelled with the target language: start again
				t.tr.To = arg
				t.reset()
				console.Notef("[translating into %s; history cleared]\n", cmp.Or(t.tr.To, "the language by the text"))
				return false
			}},
		{names: []string{"image"}, args: "FILE [text]", about: "send an image, with a text",
			unless: reads("vision", "images", func() bool { return t.c != nil && t.c.Images() }), run: media(false)},
		{names: []string{"audio"}, args: "FILE.wav [text]", about: "send a clip, with a text",
			unless: reads("audio", "audio", func() bool { return t.c != nil && t.c.Audio() }), run: media(true)},
		{names: []string{"status", "info"}, about: "the model, the length of the conversation and the settings", run: func(string) bool { t.status(); return false }},
		quitCommand(),
	}
}

// say sends a message and writes an error as one of the turn: the conversation goes on.
func (t *talker) say(msg text.Message) {
	if err := t.answer(msg); err != nil {
		printError(err)
	}
}

// status writes the model at hand, how long the conversation is and how the model answers.
func (t *talker) status() {
	if t.run == nil {
		console.Notef("[no model is loaded: /model NAME loads one]\n")
		return
	}
	line := t.name()
	if t.tl != nil {
		line += fmt.Sprintf(": a translation model, %d texts in its history", len(t.pairs))
	} else {
		turns := 0
		for _, m := range t.msgs {
			if m.Role == "user" {
				turns++
			}
		}
		line += fmt.Sprintf(": context %d tokens; %d messages sent", t.c.Context(), turns)
		if len(t.msgs) > 0 {
			if n, err := t.c.Count(context.Background(), t.msgs, t.options()); err == nil {
				line += fmt.Sprintf(", %d tokens", n)
			}
		}
		line += "; thinking " + onOff(t.think)
		if t.system != "" {
			line += "; system message: " + oneLine(t.system)
		}
	}
	if t.translator() {
		line += "; translating into " + cmp.Or(t.tr.To, "the language by the text")
	}
	console.Notef("[%s]\n", line)
}

// onOff is a setting as the commands write it.
func onOff(on bool) string {
	if on {
		return "on"
	}
	return "off"
}

// talk is the conversation in the terminal.
func (t *talker) talk() error {
	console.Notef("%s: %s/help lists the commands (/model picks another model, /reset clears the conversation, /exit or Ctrl-D quits); Ctrl-C stops an answer, ↑ ↓ go through what was typed\n", t.name(), t.hints())
	t.ed = console.NewEditor("chat")
	cs := t.commands()
	for {
		line, ok := read(t.ed)
		if !ok {
			return nil
		}
		if line == "" {
			continue
		}
		if handled, quit := cs.do(line); handled {
			if quit {
				return nil
			}
			continue
		}
		t.say(text.Message{Role: "user", Text: line})
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
