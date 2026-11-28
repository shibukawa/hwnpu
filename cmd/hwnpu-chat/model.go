package main

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"text/tabwriter"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/text"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/catalog"
	"github.com/shibukawa/hwnpu/internal/console"
)

// /model changes the model of a conversation in the terminal, as the other chat programs do:
// alone it lists the chat models of the definitions to choose one from (with ↑ ↓ on a terminal),
// with a name it loads that one. A model that is not here is listed with what it downloads and
// keeps on the disk, and asked for before it is downloaded.

// modelChoice is a chat model /model lists: its definition and, from its plan with the
// command's flags, whether it is here and what making it ready takes.
type modelChoice struct {
	info ai.ModelInfo
	item ai.Item
	err  error // why it cannot be planned (the model policy refuses it)
}

// state is the choice's column of whether it is here: yes, or what making it ready takes (what
// it downloads and keeps on the disk), short enough for a terminal of 80 columns.
func (m modelChoice) state() string {
	it := m.item
	switch {
	case m.err != nil:
		return "no: " + strings.TrimPrefix(m.err.Error(), "ai: ")
	case it.Ready:
		return "yes"
	case it.Download > 0 && it.Disk > 0:
		return fmt.Sprintf("no: %s download, %s disk", cache.FormatSize(it.Download), cache.FormatSize(it.Disk))
	case it.Download > 0:
		return fmt.Sprintf("no: %s download", cache.FormatSize(it.Download))
	case it.Disk > 0:
		return fmt.Sprintf("no: %s disk (checkpoint here)", cache.FormatSize(it.Disk))
	}
	return "no"
}

// choices are the chat models of the definitions, each planned as /model would load it.
func (t *talker) choices(ctx context.Context) []modelChoice {
	var out []modelChoice
	var cms []chatModel
	for _, info := range t.cat.OfKind(string(catalog.KindChat)) {
		cf, ef := t.cmd.flagsFor(info.Name)
		cm, err := chatModelOf(t.st, t.cat, info.Name, cf, ef)
		out, cms = append(out, modelChoice{info: info, err: err}), append(cms, cm)
	}
	// one plan of them all; when a model fails it (a policy that refuses one), one of each
	var uses []ai.Requirement
	var defs []ai.Definition
	var of []int // the choice of each use
	for i, cm := range cms {
		if out[i].err != nil {
			continue
		}
		uses, defs, of = append(uses, cm.use()), append(defs, cm.defs...), append(of, i)
	}
	all := ai.NewApp("hwnpu-chat", uses...)
	all.Store = t.st
	all.Store.Definitions = defs
	if p, err := all.Plan(ctx); err == nil && len(p.Items) == len(of) {
		for n, i := range of {
			out[i].item = p.Items[n]
		}
		return out
	}
	for _, i := range of {
		p, err := cms[i].app().Plan(ctx)
		switch {
		case err != nil:
			out[i].err = err
		case len(p.Items) == 1:
			out[i].item = p.Items[0]
		}
	}
	return out
}

// table is the choices as lines of aligned columns under a line of headings: the model at hand
// marked, each with its memory running, whether it is here or what it takes, and its features
// (last: a narrow terminal cuts them, not the sizes).
func (t *talker) table(cs []modelChoice) (head string, rows []string) {
	var b strings.Builder
	tw := tabwriter.NewWriter(&b, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "MODEL\tMEMORY\tHERE\tFEATURES")
	for _, c := range cs {
		name := c.info.Name
		if t.at(name) {
			name += " *"
		}
		mem := "-"
		if gb := cmp.Or(c.item.MemoryGB, c.info.MemoryGB); gb > 0 { // the plan's: of the variant for this machine's memory
			mem = fmt.Sprintf("%.1f GB", gb)
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\n", name, mem, c.state(), strings.Join(c.info.Features, ","))
	}
	tw.Flush()
	lines := strings.Split(strings.TrimRight(b.String(), "\n"), "\n")
	return lines[0], lines[1:]
}

// model is /model: with a name, the model to load; alone, the list to choose from.
func (t *talker) model(arg string) {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if arg != "" {
		t.change(ctx, arg)
		return
	}
	cs := t.choices(ctx)
	switch {
	case ctx.Err() != nil:
		console.Notef("[stopped]\n")
		return
	case len(cs) == 0:
		printError(errors.New("the definitions have no chat model"))
		return
	}
	head, rows := t.table(cs)
	cur := 0
	for i, c := range cs {
		if t.at(c.info.Name) {
			cur = i
		}
	}
	i, err := t.ed.Select(head, rows, cur)
	if errors.Is(err, console.ErrNoTerminal) {
		// not a terminal to choose on: the list, to name one of
		fmt.Println("  " + head)
		for _, r := range rows {
			fmt.Println("  " + r)
		}
		console.Notef("* is the model at hand; /model NAME loads another\n")
		return
	}
	if i < 0 || t.at(cs[i].info.Name) {
		console.Notef("[still %s]\n", t.name())
		return
	}
	t.change(ctx, cs[i].info.Name)
}

// errNotFetched is ask's error when the user does not want the download.
var errNotFetched = errors.New("not downloaded")

// ask says what the models of the plan that are not here take, and asks whether to go on: on a
// terminal, where a download of gigabytes is not started by a slip of the keys.
func (t *talker) ask(p *ai.Plan) error {
	lines := toFetch(p)
	if len(lines) == 0 {
		return nil
	}
	var need int64
	for _, it := range p.Items {
		if !it.Ready {
			need += it.Download + it.Disk
		}
	}
	for i, line := range lines {
		if i == len(lines)-1 && p.DownloadBytes() > 0 && need > p.DownloadBytes() {
			line += fmt.Sprintf(" (%s free needed while it is made ready)", cache.FormatSize(need))
		}
		console.Notef("%s\n", line)
	}
	if !console.Terminal(os.Stdin) {
		return nil
	}
	what := "Download it?"
	if p.DownloadBytes() == 0 {
		what = "Make it ready?"
	}
	line, err := t.ed.Ask(what + " [y/N] ")
	if a := strings.ToLower(strings.TrimSpace(line)); err != nil || (a != "y" && a != "yes") {
		return errNotFetched
	}
	return nil
}

// change loads the model name names in place of the one at hand, which is let go once the new
// one is ready (downloaded and converted) and before it is loaded: the two are not in memory
// at once. When the new one fails to load, the one before it is loaded again. The conversation
// goes on with the new model, unless one of the two is a translation model or the conversation
// has images or clips the new one does not read: then it starts again.
func (t *talker) change(ctx context.Context, name string) {
	if !isPath(name) {
		// a name the definitions do not have: said in a line (the command line's lists them all)
		var none *ai.NoModelError
		if _, _, err := ai.Normalize(t.cat, text.Chat(name)); errors.As(err, &none) {
			printError(fmt.Errorf("no chat model, family or use %q (/model alone lists them)", name))
			return
		}
	}
	cf, ef := t.cmd.flagsFor(name)
	cm, err := chatModelOf(t.st, t.cat, name, cf, ef)
	if err != nil {
		printError(err)
		return
	}
	old := t.run
	if old != nil && cm.chat.Name() == old.cm.chat.Name() && (cm.translate == nil) == (old.cm.translate == nil) {
		console.Notef("[already %s]\n", t.name())
		return
	}
	released := false
	run, err := t.cmd.start(ctx, cm, t.ask, func() {
		t.use(nil)
		old.close()
		released = true
	})
	if err != nil {
		switch {
		case errors.Is(err, errNotFetched):
			console.Notef("[still %s]\n", t.name())
		case errors.Is(err, context.Canceled):
			console.Notef("[stopped]\n")
		default:
			printError(err)
		}
		if !released || old == nil {
			return
		}
		// the model before it again, whatever stopped the new one
		back, stop := signal.NotifyContext(context.Background(), os.Interrupt)
		defer stop()
		run, err = t.cmd.start(back, old.cm, said, nil)
		if err != nil {
			printError(fmt.Errorf("%s is not loaded again either: %w (/model NAME loads one)", old.cm.base, err))
			return
		}
		t.use(run)
		console.Notef("[back to %s]\n", t.name())
		return
	}
	wasTranslator := old != nil && old.tl != nil
	t.use(run)
	cm.keep(ctx)
	kept := "the conversation goes on"
	switch {
	case wasTranslator || t.tl != nil:
		// a translation model's history is its pairs, a chat model's its messages
		t.restart()
		kept = "a new conversation"
	case !t.reads():
		t.restart()
		kept = "a new conversation, as it does not read what the last had"
	case len(t.msgs) == 0 || (len(t.msgs) == 1 && t.msgs[0].Role == "system"):
		kept = ""
	}
	if kept != "" {
		kept = "; " + kept
	}
	console.Notef("[%s loaded%s]\n", t.name(), kept)
	if h := strings.TrimSuffix(t.hints(), ", "); h != "" {
		console.Notef("%s\n", h)
	}
}

// restart starts the conversation again for another model, keeping the message sent last for
// /retry: the same question, asked of the new model.
func (t *talker) restart() {
	last, sent := t.last, t.sent
	t.reset()
	t.last, t.sent, t.lastAt = last, sent, len(t.msgs)
}

// reads reports whether the model at hand reads the images and clips of the conversation.
func (t *talker) reads() bool {
	for _, m := range t.msgs {
		for _, md := range m.Media {
			if t.c == nil || (md.Audio && !t.c.Audio()) || (!md.Audio && !t.c.Images()) {
				return false
			}
		}
	}
	return true
}
