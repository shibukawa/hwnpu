package text

import (
	"cmp"
	"context"
	"errors"
	"io"
	"strings"
	"sync"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/flow"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
)

// Translate is the use of a translation model (PLaMo 3 Translate) for an app: a model, a family
// or a use of the definitions ("translate", "plamo3translate:2b").
func Translate(name string) ai.Use[*Translator] {
	return ai.Define[*Translator](translateRole, name)
}

var translateRole = &airole.Role{
	Name:    "translate",
	Kinds:   []chat.Kind{chat.KindChat},
	Feature: "translation",
	Prepare: prepareEngine,
	Load: func(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
		eng, err := loadEngine(ctx, f, e)
		if err != nil {
			return nil, err
		}
		return &Translator{e: eng, model: e.Model}, nil
	},
}

// TranslateOptions are how a text is translated (PLaMo 3 Translate's settings).
type TranslateOptions struct {
	// To is the language to translate into, a lowercase code (en, ja, zh-hans, ja(easy)); ""
	// is English for a text with Japanese characters and Japanese for any other.
	To string
	// From is the text's language; "" lets the model find it.
	From string
	// Form is polite or plain, for Japanese; "" is the model's choice.
	Form string
	// Context is reference text: what the conversation is about, names.
	Context string
	// History are earlier texts and their translations, oldest first, which the translation
	// follows in its words and style.
	History []Pair
	// Glossary are terms to translate so.
	Glossary []Pair
	// MaxTokens is the most tokens of the translation (0: 1024).
	MaxTokens int
}

// Pair is a text and its translation.
type Pair struct{ Source, Target string }

// Translator translates texts (PLaMo 3 Translate). It runs one request at a time; others wait.
type Translator struct {
	mu    sync.Mutex
	e     *chat.Engine
	model string
}

// Model is the name of the model in the definitions.
func (t *Translator) Model() string { return t.model }

// OnNPU reports whether the programs were verified to run on the NPU.
func (t *Translator) OnNPU() bool { return t.e.OnNPU() }

// Translate translates a text.
func (t *Translator) Translate(ctx context.Context, text string, o TranslateOptions) (string, error) {
	if strings.TrimSpace(text) == "" {
		return "", errors.New("text: nothing to translate")
	}
	var msgs []chat.Message
	for _, p := range o.History {
		msgs = append(msgs, chat.Message{Role: "user", Content: chat.Content{Text: p.Source}},
			chat.Message{Role: "assistant", Content: chat.Content{Text: p.Target}})
	}
	msgs = append(msgs, chat.Message{Role: "user", Content: chat.Content{Text: text}})
	tr := &chat.Translation{TargetLang: o.To, SourceLang: o.From, Form: o.Form, Context: o.Context}
	for _, g := range o.Glossary {
		tr.Glossary = append(tr.Glossary, chat.Term{Source: g.Source, Target: g.Target})
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	c, err := t.e.Complete(ctx, msgs, nil, chat.Options{Translation: tr}, chat.Sampling{}, cmp.Or(o.MaxTokens, 1024), 0, nil)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(c.Content), nil
}

// historyTurns is how many earlier pairs TranslateStream gives each translation.
const historyTurns = 4

// TranslateStream translates the final chunks of a stream of text (a transcript's settled
// segments), each with the last few pairs before it as its history after o.History, into final
// chunks of the same times. Provisional chunks are left out.
func (t *Translator) TranslateStream(ctx context.Context, s flow.Stream[flow.Text], o TranslateOptions) flow.Stream[flow.Text] {
	return t.translateStream(ctx, s, func() TranslateOptions { return o })
}

// TranslateNode is the node of a translator in a network (flow, ai.App.Connect): texts in,
// their translations out, as TranslateStream gives them. Its settings may be changed while the
// network runs (Update): the next text is translated with them, and a change of the language
// to translate into starts the history of earlier pairs anew.
func TranslateNode(u ai.Use[*Translator], o TranslateOptions) *ai.ModelNode[flow.Text, flow.Text, TranslateOptions] {
	return ai.NewModelNode(u, o, func(ctx context.Context, t *Translator, in flow.Stream[flow.Text], options func() TranslateOptions) flow.Stream[flow.Text] {
		return t.translateStream(ctx, in, options)
	})
}

// translateStream is TranslateStream with the options asked for at each text.
func (t *Translator) translateStream(ctx context.Context, s flow.Stream[flow.Text], options func() TranslateOptions) flow.Stream[flow.Text] {
	return func(yield func(flow.Chunk[flow.Text], error) bool) {
		var hist []Pair
		to := options().To
		for c, err := range s {
			if err != nil {
				yield(flow.Chunk[flow.Text]{}, err)
				return
			}
			if !c.Final || strings.TrimSpace(c.Data.Text) == "" {
				continue
			}
			o := options()
			if o.To != to { // pairs of another language are no history for this one
				to, hist = o.To, nil
			}
			oo := o
			oo.History = append(append([]Pair(nil), o.History...), hist...)
			if oo.From == "" {
				oo.From = c.Data.Lang
			}
			out, err := t.Translate(ctx, c.Data.Text, oo)
			if err != nil {
				yield(flow.Chunk[flow.Text]{}, err)
				return
			}
			hist = append(hist, Pair{c.Data.Text, out})
			if len(hist) > historyTurns {
				hist = hist[len(hist)-historyTurns:]
			}
			lang := o.To
			if lang == "" {
				lang = chat.AutoTarget(c.Data.Text)
			}
			if !yield(flow.Chunk[flow.Text]{Data: flow.Text{Text: out, Lang: lang}, Start: c.Start, Len: c.Len, Final: true}, nil) {
				return
			}
		}
	}
}

// Close unloads the model. An app leaves it to the Runtime (its Close, or the use's Release).
func (t *Translator) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.e.Close()
}
