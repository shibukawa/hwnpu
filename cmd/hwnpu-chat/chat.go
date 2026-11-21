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
	"time"

	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/daemon"
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
	cl, err := daemon.Use(ctx, "off", logw())
	if err != nil {
		return err
	}
	if cl != nil {
		return c.runThrough(ctx, stop, cl)
	}
	if err := c.prepare(); err != nil {
		return err
	}
	c.loadFlags.ExpertsKey = cmp.Or(c.loadFlags.ExpertsKey, "chat")
	e, err := loadChat(ctx, c.Model, &c.loadFlags, &c.convertFlags, &c.encoderFlags)
	if err != nil {
		return err
	}
	defer e.Close()
	stop() // Ctrl-C stops an answer, not the chat
	return c.converse(local{e.Engine}, c.think(e.Engine))
}

// converse answers the prompt given, or talks, with the model r.
func (c *ChatCmd) converse(r replier, think bool) error {
	t := &talker{r: r, think: think, thinkBudget: c.ThinkBudget, maxNew: c.N, s: c.sampling(), seed: c.Seed,
		tr: &chat.Translation{TargetLang: c.To, SourceLang: c.From, Form: c.Form}}
	t.reset(c.System)
	if c.Prompt == "" {
		return t.talk(c.System)
	}
	console.Running() // the times of a conversation in the terminal are not told
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

// talker holds a conversation with a chat model: one loaded here (local) or one hwnpud holds
// (remote).
type talker struct {
	r           replier
	msgs        []chat.Message
	think       bool
	thinkBudget int
	maxNew      int
	s           chat.Sampling
	seed        uint64
	tr          *chat.Translation
}

// replier answers a conversation, writing the answer to out as it comes.
type replier interface {
	reply(ctx context.Context, msgs []chat.Message, o turn, out *printer) (chat.Message, turnStats, error)
	// features are what the model reads besides text, and whether it translates.
	features() features
}

type features struct{ images, audio, translator bool }

// turn is how a turn is answered.
type turn struct {
	think       bool
	thinkBudget int
	maxNew      int
	s           chat.Sampling
	seed        uint64
	tr          *chat.Translation
}

// turnStats are a turn's statistics, for its line.
type turnStats struct {
	prompt, reused, generated int
	prefill, decode           time.Duration
	thinkCut, thinkLooped     bool
	looped                    bool
	drafted, accepted         int
	seed                      uint64
}

// printer writes an answer as it comes: the thinking to standard error in dim, the answer to
// standard output.
type printer struct{ thought, answered bool }

func (p *printer) write(dr, dc string) {
	if dr != "" {
		fmt.Fprint(os.Stderr, dimmed(dr))
		p.thought = true
	}
	if dc != "" {
		if !p.answered && p.thought {
			fmt.Fprint(os.Stderr, "\n\n")
		}
		fmt.Print(dc)
		p.answered = true
	}
}

func (p *printer) end() {
	if p.answered || p.thought {
		fmt.Println()
	}
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
	t.msgs = append(t.msgs, chat.Message{Role: "user", Content: content})
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	var p printer
	msg, st, err := t.r.reply(ctx, t.msgs, turn{think: t.think, thinkBudget: t.thinkBudget, maxNew: t.maxNew, s: t.s, seed: t.seed, tr: t.tr}, &p)
	stop()
	p.end()
	if err != nil && !errors.Is(err, context.Canceled) {
		t.msgs = t.msgs[:len(t.msgs)-1]
		return err
	}
	t.msgs = append(t.msgs, msg)
	if errors.Is(err, context.Canceled) && st.prompt == 0 {
		console.Notef("[stopped]\n") // through hwnpud: a stream cut short has no statistics
		return nil
	}
	cut := ""
	if st.thinkCut {
		cut = fmt.Sprintf("; thinking cut at %d tokens", t.thinkBudget)
	}
	if st.thinkLooped {
		cut = "; thinking cut going round in circles"
	}
	if st.looped {
		cut += "; stopped repeating itself"
	}
	if st.drafted > 0 {
		cut += fmt.Sprintf("; %d of %d drafts accepted", st.accepted, st.drafted)
	}
	if t.s.Temperature > 0 && st.seed != 0 {
		cut += fmt.Sprintf("; seed %d", st.seed)
	}
	if errors.Is(err, context.Canceled) {
		cut += "; stopped"
	}
	console.Notef("[prompt %d tokens, %d new in %.1fs; %d tokens at %.1f tokens/s%s]\n",
		st.prompt, st.prompt-st.reused, st.prefill.Seconds(), st.generated, float64(st.generated)/max(st.decode.Seconds(), 1e-9), cut)
	if console.Profile {
		decode.WriteProfile(os.Stderr)
		if b, ok := mem.NPUWired(); ok {
			fmt.Fprintf(os.Stderr, "profile: the Neural Engine holds %.2f GiB\n", float64(b)/(1<<30))
		}
	}
	return nil
}

// local is a chat model loaded in this process.
type local struct{ e *chat.Engine }

func (l local) features() features {
	return features{images: l.e.Images(), audio: l.e.Audio(), translator: l.e.Template == chat.Plamo3}
}

func (l local) reply(ctx context.Context, msgs []chat.Message, o turn, out *printer) (chat.Message, turnStats, error) {
	e := l.e
	prompt, thinking, err := e.Template.Render(msgs, nil, chat.Options{NoThink: !o.think, Translation: o.tr})
	if err != nil {
		return chat.Message{}, turnStats{}, err
	}
	var media []chat.Media
	for _, m := range msgs {
		media = append(media, m.Content.Media...)
	}
	ids, err := e.Tokens(ctx, prompt, media)
	if err != nil {
		return chat.Message{}, turnStats{}, err
	}
	var sent chat.Stream
	var gen []int32
	st, err := e.Generate(ctx, chat.Request{Prompt: ids, MaxTokens: o.maxNew, Sampling: o.s, Seed: o.seed, Think: e.ThinkLimit(o.thinkBudget, thinking)}, func(g []int32) (bool, error) {
		gen = g
		a := e.Template.Parse(e.Tok.Decode(g, false), thinking, nil, false)
		out.write(sent.Next(a.Reasoning, a.Content))
		return false, nil
	})
	a := e.Template.Parse(e.Tok.Decode(gen, false), thinking, nil, true)
	r := a.Reasoning
	msg := chat.Message{Role: "assistant", Content: chat.Content{Text: a.Content}, Reasoning: &r}
	return msg, turnStats{prompt: st.Prompt, reused: st.Reused, generated: st.Generated, prefill: st.Prefill, decode: st.Decode,
		thinkCut: st.ThinkCut, thinkLooped: st.ThinkLooped, looped: st.Looped, drafted: st.Drafted, accepted: st.Accepted, seed: st.Seed}, err
}

// talk is the conversation in the terminal.
func (t *talker) talk(system string) error {
	f := t.r.features()
	media := ""
	if f.images {
		media += "/image FILE [text] sends an image, "
	}
	if f.audio {
		media += "/audio FILE.wav [text] sends a clip, "
	}
	translator := f.translator
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
