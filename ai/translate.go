package ai

import (
	"cmp"
	"context"
	"errors"
	"strings"
	"sync"

	"github.com/shibukawa/hwnpu/internal/chat"
)

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
		return "", errors.New("ai: nothing to translate")
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
func (t *Translator) TranslateStream(ctx context.Context, s Stream[Text], o TranslateOptions) Stream[Text] {
	return func(yield func(Chunk[Text], error) bool) {
		var hist []Pair
		for c, err := range s {
			if err != nil {
				yield(Chunk[Text]{}, err)
				return
			}
			if !c.Final || strings.TrimSpace(c.Data.Text) == "" {
				continue
			}
			oo := o
			oo.History = append(append([]Pair(nil), o.History...), hist...)
			if oo.From == "" {
				oo.From = c.Data.Lang
			}
			out, err := t.Translate(ctx, c.Data.Text, oo)
			if err != nil {
				yield(Chunk[Text]{}, err)
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
			if !yield(Chunk[Text]{Data: Text{Text: out, Lang: lang}, Start: c.Start, Len: c.Len, Final: true}, nil) {
				return
			}
		}
	}
}

// Close unloads the model.
func (t *Translator) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.e.Close()
}
