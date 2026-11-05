"""Prompts rendered by PLaMo 3 Translate's chat template, for checking internal/chat's Plamo3.

    python plamo3_template_golden.py MODEL_DIR OUT.json

MODEL_DIR holds the tokenizer and chat_template.jinja of pfnet/plamo-3-translate-2b (every size
ships the same). Each case is rendered by transformers' apply_chat_template with
add_generation_prompt and its variables (target_lang, source_lang, form, context, glossary) and
written with the prompt as "want". Needs transformers, jinja2 and numba, not torch.
"""
import argparse
import json

U = lambda t: {"role": "user", "content": t}  # noqa: E731
A = lambda t: {"role": "assistant", "content": t}  # noqa: E731

CASES = [
    {"messages": [U("The meeting has been rescheduled to next Monday.")], "kwargs": {"target_lang": "ja"}},
    {"messages": [U("会議は来週の月曜日に変更されました。")], "kwargs": {"target_lang": "en", "source_lang": "ja"}},
    {"messages": [U("Thank you for your help.")], "kwargs": {"target_lang": "ja", "form": "plain"}},
    {"messages": [U("Thank you for your help.")], "kwargs": {"target_lang": "ja(easy)"}},
    {"messages": [U("今日はいい天気ですね。")], "kwargs": {"target_lang": "zh-hans"}},
    {
        "messages": [
            U("Open the project dashboard."),
            A("プロジェクトの管理画面を開きます。"),
            U("It lists all active tasks."),
            A("進行中のタスクがすべて表示されます。"),
            U("You can also filter them by owner."),
        ],
        "kwargs": {"target_lang": "ja", "source_lang": "en"},
    },
    {
        "messages": [U("The NPU runs the model at low power.\n\n  Indented line, two  spaces.")],
        "kwargs": {
            "target_lang": "ja",
            "context": "hwnpu is a Go library for the NPUs built into CPUs.",
            "glossary": [{"source": "NPU", "target": "NPU"}, {"source": "model", "target": "モデル"}],
            "form": "polite",
        },
    },
    {"messages": [U("Hello."), A("こんにちは。"), U("Goodbye.")], "kwargs": {"target_lang": "ja", "context": ""}},
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("out")
    a = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
    out = []
    for c in CASES:
        want = tok.apply_chat_template(c["messages"], tokenize=False, add_generation_prompt=True, **c["kwargs"])
        out.append({**c, "want": want})
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"{a.out}: {len(out)} conversations")


if __name__ == "__main__":
    main()
