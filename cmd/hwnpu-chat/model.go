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
	"unicode/utf8"

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

// dl is the choice's column of what loading it downloads first: ready when it is here, else
// about how much (0 when only its conversion is missing), - when it cannot be planned.
func (m modelChoice) dl() string {
	switch {
	case m.err != nil:
		return "-"
	case m.item.Ready:
		return "ready"
	case m.item.Download <= 0:
		return "0"
	}
	return shortSize(m.item.Download)
}

// shortSize is n bytes in five characters at most, for a column of the list: 26.3G, 830M; - for
// none (not known).
func shortSize(n int64) string {
	switch {
	case n <= 0:
		return "-"
	case n < 1_000_000:
		return "<1M"
	case n < 999_500_000:
		return fmt.Sprintf("%.0fM", float64(n)/1e6)
	}
	return fmt.Sprintf("%.1fG", float64(n)/1e9)
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

// table is the choices as lines of aligned columns under a line of headings, all of it within
// the 80 columns of a terminal (78 with the longest names and features of models.yaml, beside
// the two of the list's mark): the model at hand marked with a *, the memory it takes running
// (MEM), what loading it downloads first (DL: ready when it is here), what it keeps on the disk
// (DISK), and its features, none cut. A model that cannot be planned says why in their place.
//
// paint colors a row's features as tags, each feature in its color, on a terminal that takes
// colors (colors): on one of cols columns that has the room, tags with a background and a space
// on each side (the rows have those spaces); else the features' words, a space in place of the
// comma between them. Without colors the features are as models.yaml writes them, and paint
// changes nothing.
func (t *talker) table(cs []modelChoice, cols int, colors bool) (head string, rows []string, paint func(string) string) {
	build := func(chips bool) []string {
		var b strings.Builder
		tw := tabwriter.NewWriter(&b, 0, 0, 2, ' ', 0)
		if chips {
			fmt.Fprintln(tw, " MODEL\tMEM\tDL\tDISK\t FEATURES")
		} else {
			fmt.Fprintln(tw, " MODEL\tMEM\tDL\tDISK\tFEATURES")
		}
		for _, c := range cs {
			name := " " + c.info.Name
			if t.at(c.info.Name) {
				name = "*" + c.info.Name
			}
			mem := "-"
			if gb := cmp.Or(c.item.MemoryGB, c.info.MemoryGB); gb > 0 { // the plan's: of the variant for this machine's memory
				mem = fmt.Sprintf("%.1fG", gb)
			}
			features := strings.Join(c.info.Features, ",")
			switch {
			case c.err != nil:
				features = notHad + strings.TrimPrefix(c.err.Error(), "ai: ")
			case chips && len(c.info.Features) > 0:
				features = " " + strings.Join(c.info.Features, "   ") + " "
			}
			fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\n", name, mem, c.dl(), shortSize(c.item.Disk), features)
		}
		tw.Flush()
		return strings.Split(strings.TrimRight(b.String(), "\n"), "\n")
	}
	chips := false
	lines := build(false)
	if colors {
		// the tags with their spaces, when every model's fit beside the list's mark
		wide, fits := build(true), true
		for i, l := range wide[1:] {
			fits = fits && (cs[i].err != nil || utf8.RuneCountInString(l)+2 < cols)
		}
		if fits {
			chips, lines = true, wide
		}
	}
	at := utf8.RuneCountInString(lines[0][:strings.Index(lines[0], "FEATURES")])
	if chips {
		at-- // the space before the first tag
	}
	paint = func(row string) string {
		if !colors {
			return row
		}
		r := []rune(row)
		if len(r) <= at {
			return row
		}
		left, features := string(r[:at]), string(r[at:])
		switch {
		case strings.HasPrefix(strings.TrimSpace(features), strings.TrimSpace(notHad)):
			return left + "\x1b[31m" + features + "\x1b[39m"
		case chips:
			return left + paintChips(features)
		}
		words := strings.Split(features, ",")
		for i, w := range words {
			if c := tagColor(w).word; c != "" {
				words[i] = "\x1b[" + c + "m" + w + "\x1b[39m"
			}
		}
		return left + strings.Join(words, " ")
	}
	return lines[0], lines[1:], paint
}

// notHad begins what a model that cannot be planned says in place of its features.
const notHad = "not to be had: "

// tagColors are the colors of the features in /model's list (SGR parameters): of a tag, its
// text on a background, and of the feature's word alone.
var tagColors = map[string]struct{ chip, word string }{
	"tools":       {"30;46", "36"},
	"thinking":    {"30;45", "35"},
	"vision":      {"30;42", "32"},
	"audio":       {"30;43", "33"},
	"translation": {"97;44", "94"},
}

// tagColor is the colors of the feature a word of the list names, a word the terminal's width
// cut (thin…) too; a feature without colors of its own is a gray tag and a plain word.
func tagColor(word string) struct{ chip, word string } {
	name := strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(word), "…"))
	if c, ok := tagColors[name]; ok {
		return c
	}
	if name != "" {
		for n, c := range tagColors {
			if strings.HasPrefix(n, name) {
				return c
			}
		}
	}
	return struct{ chip, word string }{"30;47", ""}
}

// paintChips colors the tags of a row's features: each word with the space on each side of it,
// the spaces between the tags left as they are.
func paintChips(features string) string {
	var b strings.Builder
	for i := 0; i < len(features); {
		if features[i] == ' ' && (i+1 == len(features) || features[i+1] == ' ') {
			b.WriteByte(' ') // between two tags, or after the last
			i++
			continue
		}
		start := i
		if features[i] == ' ' {
			i++ // the tag's own space
		}
		for i < len(features) && features[i] != ' ' {
			i++
		}
		word := features[start:i]
		if i < len(features) {
			i++ // the space that closes it
		}
		b.WriteString("\x1b[" + tagColor(word).chip + "m" + features[start:i] + "\x1b[39;49m")
	}
	return b.String()
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
	colors := console.Colors(os.Stdout)
	head, rows, paint := t.table(cs, console.Columns(os.Stdout), colors)
	cur := 0
	for i, c := range cs {
		if t.at(c.info.Name) {
			cur = i
		}
	}
	i, err := t.ed.Select(head, rows, cur, paint)
	if errors.Is(err, console.ErrNoTerminal) {
		// not a terminal to choose on: the list, to name one of
		fmt.Println("  " + head)
		for _, r := range rows {
			fmt.Println("  " + paint(r))
		}
		console.Notef("* is the model at hand; DL is what loading a model downloads first (ready: it is here), DISK what it keeps on the disk; /model NAME loads another\n")
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
		if t.unread(m) != "" {
			return false
		}
	}
	return true
}

// unread is what of a message the model at hand does not read: "an image", "a clip", "" when it
// reads it all.
func (t *talker) unread(m text.Message) string {
	for _, md := range m.Media {
		switch {
		case md.Audio && (t.c == nil || !t.c.Audio()):
			return "a clip"
		case !md.Audio && (t.c == nil || !t.c.Images()):
			return "an image"
		}
	}
	return ""
}
