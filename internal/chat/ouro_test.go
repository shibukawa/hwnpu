package chat

import (
	"encoding/json"
	"os"
	"testing"
)

// Ouro.Render writes what Ouro's chat template does with enable_thinking. The expected prompts in
// testdata/ouro_template.json were rendered by transformers' apply_chat_template with
// ByteDance/Ouro-1.4B-Thinking's tokenizer (add_generation_prompt, enable_thinking).
func TestOuroRenderMatchesTemplate(t *testing.T) {
	raw, err := os.ReadFile("testdata/ouro_template.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Name     string    `json:"name"`
		Messages []Message `json:"messages"`
		Want     string    `json:"want"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			got, thinking, err := Ouro.Render(c.Messages, nil, Options{})
			if err != nil {
				t.Fatal(err)
			}
			if got != c.Want || !thinking {
				t.Errorf("got (thinking %v)\n%s\nwant\n%s", thinking, got, c.Want)
			}
		})
	}
	// without thinking, an empty thinking after the template's prompt
	got, thinking, err := Ouro.Render(cases[0].Messages, nil, Options{NoThink: true})
	want := cases[0].Want[:len(cases[0].Want)-len("<think>\n")] + "<think>\n\n</think>\n\n"
	if err != nil || thinking || got != want {
		t.Errorf("no thinking: %q, %v, %v", got, thinking, err)
	}
	if _, _, err := Ouro.Render(cases[0].Messages, []Tool{{}}, Options{}); err == nil {
		t.Error("tools: no error")
	}
	if tmpl, err := TemplateFor("ouro"); err != nil || tmpl != Ouro {
		t.Errorf("TemplateFor(ouro): %v, %v", tmpl, err)
	}
	if a, err := archOf("Ouro"); err != nil || a != "ouro" {
		t.Errorf("archOf(Ouro): %q, %v", a, err)
	}
}
