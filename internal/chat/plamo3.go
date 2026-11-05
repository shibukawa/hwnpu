package chat

import (
	"fmt"
	"strings"
	"unicode"
)

// Plamo3 is the format of PLaMo 3's translation models (pfnet/plamo-3-translate-*): the PFML3
// prompt its chat_template.jinja writes. The user messages are the texts to translate and the
// assistant messages their translations, alternating and ending with the text to translate now;
// the earlier pairs are the translation history. A system message, which the template refuses,
// is taken as its context (reference text). The target language and the other settings are
// Options.Translation (the template's target_lang, source_lang, form, context and glossary);
// without a target language, a text with Japanese characters is translated into English and any
// other into Japanese (AutoTarget), each pair of the history labelled by its own text, where the
// template labels them all with the one target_lang. The model has no thinking and no tools.
var Plamo3 Template = plamo3Format{}

// Translation is what a translation model is asked besides the texts: the variables of PLaMo 3's
// chat template (chat_template_kwargs in the API).
type Translation struct {
	TargetLang string `json:"target_lang"` // a lowercase language code: en, ja, zh-hans, ja(easy); "": by the text
	SourceLang string `json:"source_lang"` // "": the model finds it
	Form       string `json:"form"`        // polite or plain, for Japanese; "": the model's choice
	Context    string `json:"context"`     // reference text
	Glossary   []Term `json:"glossary"`
}

// Term is an entry of a translation's glossary.
type Term struct {
	Source string `json:"source"`
	Target string `json:"target"`
}

type plamo3Format struct{}

func (plamo3Format) Stops() []string {
	return []string{"<|plamo:tag|>", "<|plamo:bos|>", "<|plamo:eos|>"}
}

func (plamo3Format) Think() (open, close, end string) { return "", "", "" }

func (plamo3Format) Render(msgs []Message, tools []Tool, o Options) (string, bool, error) {
	if len(tools) > 0 {
		return "", false, fmt.Errorf("chat: a translation model calls no tools")
	}
	var t Translation
	if o.Translation != nil {
		t = *o.Translation
	}
	if len(msgs) > 0 && msgs[0].Role == "system" {
		if t.Context == "" {
			t.Context = msgs[0].Content.Text
		}
		msgs = msgs[1:]
	}
	if len(msgs)%2 != 1 {
		return "", false, fmt.Errorf("chat: a translation is user texts and assistant translations in turn, ending with a user text")
	}
	for i, m := range msgs {
		want := "user"
		if i%2 == 1 {
			want = "assistant"
		}
		if m.Role != want {
			return "", false, fmt.Errorf("chat: message %d is %s, not %s: a translation is user texts and assistant translations in turn", i+1, m.Role, want)
		}
		if len(m.Content.Media) > 0 || len(m.ToolCalls) > 0 {
			return "", false, fmt.Errorf("chat: a translation model reads text only")
		}
		if err := pfmlText(m.Content.Text, "message", false); err != nil {
			return "", false, err
		}
	}
	auto := t.TargetLang == ""
	if auto {
		t.TargetLang = AutoTarget(msgs[len(msgs)-1].Content.Text)
	}
	if err := pfmlLang(t.TargetLang, "target_lang"); err != nil {
		return "", false, err
	}
	if t.SourceLang != "" {
		if err := pfmlLang(t.SourceLang, "source_lang"); err != nil {
			return "", false, err
		}
	}
	if err := pfmlText(t.Context, "context", true); err != nil {
		return "", false, err
	}
	if t.Form != "" && (t.Form != "polite" && t.Form != "plain" || t.TargetLang != "ja") {
		return "", false, fmt.Errorf("chat: form must be polite or plain, and needs target_lang ja")
	}
	for _, g := range t.Glossary {
		if err := pfmlText(g.Source, "glossary source", false); err != nil {
			return "", false, err
		}
		if err := pfmlText(g.Target, "glossary target", false); err != nil {
			return "", false, err
		}
	}
	var sb strings.Builder
	input := func() {
		sb.WriteString("<|plamo:tag|>input")
		if t.SourceLang != "" {
			sb.WriteString("<|plamo:key|>lang<|plamo:val|>" + t.SourceLang)
		}
	}
	sb.WriteString("<|plamo:bos|><|plamo:tag|>dataset<|plamo:msg|>translation")
	if t.Context != "" {
		sb.WriteString("<|plamo:tag|>context<|plamo:msg|>" + t.Context)
	}
	for i, m := range msgs[:len(msgs)-1] {
		if m.Role == "user" {
			input()
		} else {
			lang := t.TargetLang
			if auto {
				lang = AutoTarget(msgs[i-1].Content.Text)
			}
			sb.WriteString("<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>" + lang)
		}
		sb.WriteString("<|plamo:msg|>" + m.Content.Text)
	}
	if len(t.Glossary) > 0 {
		sb.WriteString("<|plamo:tag|>glossary")
		for _, g := range t.Glossary {
			sb.WriteString("<|plamo:index|>" + g.Source + "<|plamo:item|>" + g.Target)
		}
	}
	input()
	sb.WriteString("<|plamo:msg|>" + msgs[len(msgs)-1].Content.Text + "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>" + t.TargetLang)
	if t.Form != "" {
		sb.WriteString("<|plamo:key|>form<|plamo:val|>" + t.Form)
	}
	sb.WriteString("<|plamo:msg|>")
	return sb.String(), false, nil
}

// Parse is the translation: the text up to a special token (the stop tokens end it).
func (plamo3Format) Parse(text string, thinking bool, tools []Tool, final bool) Answer {
	if i := strings.Index(text, "<|plamo:"); i >= 0 {
		text = text[:i]
	} else if !final {
		// hold back what may be the start of a special token
		for k := len("<|plamo:") - 1; k > 0; k-- {
			if strings.HasSuffix(text, "<|plamo:"[:k]) {
				text = text[:len(text)-k]
				break
			}
		}
	}
	return Answer{Content: text}
}

// AutoTarget is the language a text is translated into when not told: English for a text with
// hiragana, katakana or kanji, Japanese for any other.
func AutoTarget(text string) string {
	for _, r := range text {
		if unicode.In(r, unicode.Hiragana, unicode.Katakana, unicode.Han) {
			return "en"
		}
	}
	return "ja"
}

// pfmlText checks a text of the prompt as the template does: not empty unless allowed, no PFML
// tokens.
func pfmlText(s, name string, empty bool) error {
	if !empty && strings.TrimSpace(s) == "" {
		return fmt.Errorf("chat: %s must not be empty", name)
	}
	if strings.Contains(s, "<|plamo:") {
		return fmt.Errorf("chat: %s must not contain PFML tokens (<|plamo:...)", name)
	}
	return nil
}

// pfmlLang checks a language code as the template does: ja(easy), or a lowercase code of two or
// three letters with an optional script of four (zh-hans).
func pfmlLang(s, name string) error {
	if s == "ja(easy)" {
		return nil
	}
	parts := strings.Split(s, "-")
	ok := len(parts) <= 2 && (len(parts[0]) == 2 || len(parts[0]) == 3) && (len(parts) == 1 || len(parts[1]) == 4)
	for _, r := range s {
		if !(r >= 'a' && r <= 'z' || r == '-') {
			ok = false
		}
	}
	if !ok {
		return fmt.Errorf("chat: %s %q must be a lowercase language code, e.g. en, ja, zh-hans or ja(easy)", name, s)
	}
	return nil
}
