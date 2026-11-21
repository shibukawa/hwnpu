package chat

import (
	"fmt"
	"strings"
)

// Ouro is the chat format of ByteDance's Ouro Thinking models (the chat_template of
// ByteDance/Ouro-1.4B-Thinking's tokenizer_config.json, the same for 2.6B): ChatML turns, a system
// turn of its own when the messages have none ("You are a helpful assistant."), the messages'
// content as it is, and no tools. The template ends the prompt at "<|im_start|>assistant\n" and,
// with enable_thinking, "<think>\n"; the model thinks either way, so Render writes it unless
// o.NoThink, and then the empty thinking Qwen3's enable_thinking=false writes, which Ouro's
// template has no switch for.
var Ouro Template = ouroFormat{}

// ouroSystem is the system message Ouro's template writes when the messages have none.
const ouroSystem = "You are a helpful assistant."

type ouroFormat struct{ qwen3Format }

func (ouroFormat) Render(msgs []Message, tools []Tool, o Options) (string, bool, error) {
	if len(msgs) == 0 {
		return "", false, fmt.Errorf("chat: no messages")
	}
	if len(tools) > 0 {
		return "", false, fmt.Errorf("chat: Ouro calls no tools")
	}
	var sb strings.Builder
	if msgs[0].Role == "system" {
		sb.WriteString("<|im_start|>system\n" + msgs[0].Content.Text + "<|im_end|>\n")
		msgs = msgs[1:]
	} else {
		sb.WriteString("<|im_start|>system\n" + ouroSystem + "<|im_end|>\n")
	}
	for _, m := range msgs {
		switch m.Role {
		case "system", "user", "assistant":
		default:
			return "", false, fmt.Errorf("chat: unexpected message role %q", m.Role)
		}
		if len(m.ToolCalls) > 0 {
			return "", false, fmt.Errorf("chat: Ouro calls no tools")
		}
		sb.WriteString("<|im_start|>" + m.Role + "\n" + m.Content.Text + "<|im_end|>\n")
	}
	sb.WriteString("<|im_start|>assistant\n")
	if o.NoThink {
		sb.WriteString("<think>\n\n</think>\n\n")
		return sb.String(), false, nil
	}
	sb.WriteString("<think>\n")
	return sb.String(), true, nil
}
