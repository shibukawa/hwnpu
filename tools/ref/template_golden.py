"""Prompts rendered by a model's chat template, for checking internal/chat's Go port of it.

    python template_golden.py MODEL_DIR OUT.json

MODEL_DIR holds the tokenizer and chat_template.jinja of google/gemma-4-E2B-it. Each conversation
below is rendered by transformers' apply_chat_template (add_generation_prompt, enable_thinking
unless no_think, preserve_thinking with keep_thinking) and written with the prompt as "want".
The conversations are in the OpenAI form the server receives: a tool call's arguments are a JSON
string, which is parsed into an object for the template, as the server does. Needs transformers
and jinja2, not torch.

--harmony renders gpt-oss's harmony template: a message's reasoning_content goes in as its
thinking (the template's name for it), no_think as reasoning_effort "low" (else "medium"), the
date is fixed at --date, and HARMONY_CASES (defaults, arrays, a final answer after a call) are
added.

--text-content joins the text parts of a content given as a list into a string first, as vLLM does
for a template that takes content as a string (Qwen3's, which renders a list as nothing or fails
on it), and as internal/chat's Content does.
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

SEARCH = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Searches the web.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "topn": {"type": "number", "description": "Results", "default": 10},
                "sites": {"type": "array", "items": {"type": "string"}},
                "scores": {"type": "array", "items": {"type": "integer"}, "nullable": True},
                "mode": {"type": "string", "enum": ["fast", "deep"], "default": "fast"},
                "safe": {"type": "boolean", "default": True},
                "note": {"type": "string", "nullable": True},
            },
            "required": ["query"],
        },
    },
}
NOARGS = {"type": "function", "function": {"name": "now", "description": "The time.", "parameters": {"type": "object", "properties": {}}}}

HARMONY_CASES = [
    {"name": "harmony: tools with defaults", "tools": [SEARCH, NOARGS], "messages": [
        {"role": "system", "content": "Answer in Japanese."}, {"role": "user", "content": "東京の天気"}]},
    {"name": "harmony: answer after a call", "tools": [SEARCH], "messages": [
        {"role": "user", "content": "Find hwnpu"},
        {"role": "assistant", "content": "", "reasoning_content": "Search it.",
         "tool_calls": [{"id": "s1", "type": "function", "function": {"name": "search", "arguments": '{"query": "hwnpu \\"npu\\"", "topn": 3}'}}]},
        {"role": "tool", "tool_call_id": "s1", "content": "{\"hits\": [\"github.com/shibukawa/hwnpu\"]}\nline 2"},
        {"role": "assistant", "content": "It is on GitHub.", "reasoning_content": "Found."},
        {"role": "user", "content": "Thanks!"},
    ]},
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--text-content", action="store_true", help="join a content's text parts into a string")
    ap.add_argument("--harmony", action="store_true", help="gpt-oss's template (module docstring)")
    ap.add_argument("--date", default="2026-11-13", help="--harmony: the date strftime_now gives")
    a = ap.parse_args()
    cases = CASES + (HARMONY_CASES if a.harmony else [])
    if a.harmony:
        import datetime as dt

        import transformers.utils.chat_template_utils as ctu

        fixed = dt.datetime.fromisoformat(a.date)

        class Fixed(dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed

        ctu.datetime = Fixed

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model_dir)
    out = []
    for c in cases:
        msgs = copy.deepcopy(c["messages"])
        for m in msgs:
            if a.harmony and "reasoning_content" in m:
                m["thinking"] = m.pop("reasoning_content")
            if a.harmony and m.get("content") is None:
                m["content"] = ""  # the template fails on null (an "in" test); the server reads it as ""
            if a.text_content and isinstance(m.get("content"), list):
                m["content"] = "".join(p["text"] for p in m["content"] if p.get("type") == "text")
            for tc in m.get("tool_calls") or []:
                tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
        want = tok.apply_chat_template(
            msgs,
            tools=c.get("tools"),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=not c.get("no_think", False),
            preserve_thinking=c.get("keep_thinking", False),
            **({"reasoning_effort": "low" if c.get("no_think") else "medium"} if a.harmony else {}),
        )
        out.append({**c, "want": want})
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"{a.out}: {len(out)} conversations")


if __name__ == "__main__":
    main()
