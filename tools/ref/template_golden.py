"""Prompts rendered by a model's chat template, for checking internal/chat's Go port of it.

    python template_golden.py MODEL_DIR OUT.json

MODEL_DIR holds the tokenizer and chat_template.jinja of google/gemma-4-E2B-it. Each conversation
below is rendered by transformers' apply_chat_template (add_generation_prompt, enable_thinking
unless no_think, preserve_thinking with keep_thinking) and written with the prompt as "want".
The conversations are in the OpenAI form the server receives: a tool call's arguments are a JSON
string, which is parsed into an object for the template, as the server does. Needs transformers
and jinja2, not torch.
"""
import argparse
import copy
import json

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Gets the weather of a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city's name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                "days": {"type": "integer", "description": "How many days", "nullable": True},
            },
            "required": ["city"],
        },
    },
}

EDIT = {
    "type": "function",
    "function": {
        "name": "edit",
        "description": "Edits files.",
        "parameters": {
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array",
                    "description": "The edits",
                    "items": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}, "Old": {"type": "string"}, "new": {"type": "string"}},
                        "required": ["path", "new"],
                    },
                },
                "options": {"type": "object", "properties": {"dry_run": {"type": "boolean"}}, "required": ["dry_run"]},
                "meta": {"type": "object", "owner": {"type": "string"}},
                "tags": {"type": "array", "items": {"type": ["string", "null"]}},
            },
            "required": ["edits"],
        },
    },
}

CASES = [
    {"name": "plain", "no_think": True, "messages": [{"role": "user", "content": "Hi"}]},
    {"name": "system thinking", "messages": [{"role": "system", "content": "  You are terse.\n"}, {"role": "user", "content": "Hi "}]},
    {
        "name": "turns with reasoning",
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "<|channel>thought\nadd<channel|>4", "reasoning_content": "Two and two."},
            {"role": "user", "content": "And 3+3?"},
            {"role": "assistant", "content": "6", "reasoning_content": "Three and three."},
        ],
    },
    {"name": "tools", "tools": [WEATHER, EDIT], "messages": [{"role": "user", "content": "Weather in Tokyo?"}]},
    {
        "name": "call and responses",
        "tools": [WEATHER],
        "messages": [
            {"role": "user", "content": "Weather in Tokyo and Osaka?"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "Two cities.",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Tokyo", "days": 2, "unit": "celsius"}'}},
                    {"id": "c2", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Osaka"}'}},
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "Sunny, 20 C"},
            {"role": "tool", "tool_call_id": "c2", "content": "Rain"},
        ],
    },
    {
        "name": "call and responses without thinking",
        "no_think": True,
        "tools": [WEATHER],
        "messages": [
            {"role": "user", "content": "Weather in Tokyo?"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "Sunny"},
        ],
    },
    {
        "name": "call pending",
        "no_think": True,
        "messages": [
            {"role": "user", "content": "Weather?"},
            {"role": "assistant", "content": "Let me look.", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Kyoto"}'}}]},
        ],
    },
    {
        "name": "answer after a response and another question",
        "keep_thinking": True,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "Be brief."}, {"type": "text", "text": " Use tools. "}]},
            {"role": "user", "content": [{"type": "text", "text": "Edit it"}]},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "I will edit.",
                "tool_calls": [
                    {
                        "id": "e1",
                        "type": "function",
                        "function": {
                            "name": "edit",
                            "arguments": '{"edits": [{"path": "a.go", "new": "x := 1.0", "Old": "x := 1"}], "options": {"dry_run": false}, "count": 3, "ratio": 0.5, "big": 1e20, "none": null, "Zeta": true, "alpha": [1, "two", {"k": "v"}]}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "e1", "content": [{"type": "text", "text": "done"}]},
            {"role": "assistant", "content": "Edited a.go."},
            {"role": "user", "content": "Thanks"},
        ],
    },
    {
        "name": "tool message by name",
        "no_think": True,
        "messages": [
            {"role": "user", "content": "Run it"},
            {"role": "assistant", "content": "Running.", "tool_calls": [{"id": "r1", "type": "function", "function": {"name": "run", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "other", "name": "runner", "content": "ok"},
            {"role": "assistant", "content": "It ran."},
        ],
    },
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("out")
    a = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model_dir)
    out = []
    for c in CASES:
        msgs = copy.deepcopy(c["messages"])
        for m in msgs:
            for tc in m.get("tool_calls") or []:
                tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
        want = tok.apply_chat_template(
            msgs,
            tools=c.get("tools"),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=not c.get("no_think", False),
            preserve_thinking=c.get("keep_thinking", False),
        )
        out.append({**c, "want": want})
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"{a.out}: {len(out)} conversations")


if __name__ == "__main__":
    main()
