"""Golden token ids of a SentencePiece tokenizer, for checking internal/tokenizer.

    python tokenizer_golden.py MODEL_DIR OUT.json [--random N]

MODEL_DIR holds tokenizer.model, tokenizer_config.json and the tokenizer's remote code
(tokenization_plamo.py for PLaMo), or a tokenizer.json converted from SentencePiece (Gemma).
Each text is encoded by the Hugging Face tokenizer without special tokens (no BOS), and, when it
contains no special token and there is a tokenizer.model, also by sentencepiece alone; the two
must agree. OUT.json holds {"cases": [{"text": ..., "ids": [...]}]}. --random N adds N
random strings drawn from the texts' characters, spaces, tabs and newlines (for a wider local
check; the committed golden file has none). Needs transformers and sentencepiece, not torch.
"""
import argparse
import json
import os
import random

TEXTS = [
    "",
    "a",
    " ",
    "PLaMo-Embedding-1Bは、Preferred Networks, Inc. によって開発された日本語テキスト埋め込みモデルです。",
    "次の文章に対して、関連する文章を検索してください: 東京の天気は晴れです。",
    "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。",
    "The Neural Engine runs small language models at low power.",
    "Hello, world! It's 2026-10-30 and the price is $1,234.56 (approx. 10%).",
    "日本語とEnglishが混ざったmixed text、全角ＡＢＣ１２３と半角ABC123。",
    "カタカナ・ひらがな・漢字、そして「かぎ括弧」と『二重かぎ括弧』。",
    "  leading spaces",
    "trailing spaces   ",
    "two  spaces,   three,    four,        eight,                 sixteen and more",
    "tab\tseparated\t\tvalues\t\t\t\tend",
    "line one\nline two\n\nline four\r\nwindows",
    "\n\n\nstarts with newlines",
    "    def f(x):\n        return x * 2\n",
    "絵文字 😀🎉👍🏽 と 👨‍👩‍👧‍👦 家族",
    "珍しい字: 𠮷野家、𩸽、髙橋、﨑、〆、ヶ",
    "combining: é ä が",
    "control chars: \x01\x07 and zero width​space",
    "literal ▁ and ▁▁ lower blocks",
    "Ελληνικά, Русский, العربية, हिन्दी, 한국어, ไทย",
    "math ∑∫√∞ ≠ ≤ ≥ and arrows → ⇒ ↔",
    "1234567890 ０１２３４５６７８９ 一二三四五",
    "https://huggingface.co/pfnet/plamo-embedding-1b?tab=readme#usage",
    "<|startoftext|>with bos written in the text",
    "before<|endoftext|>after",
    "a <|pad|> b<|unknown|>c",
    "<|startoftext|><|endoftext|>",
    "全角スペース　を含む　文",
    "ｶﾀｶﾅ半角 と ① ② ㈱ ㌔",
    "<|turn>user\nこんにちは<turn|>\n<|turn>model\n",
    "<|think|>思考<|channel>thought\n考え<channel|>答え",
    "<|tool_call>call:get_weather{city:<|\"|>Tokyo<|\"|>}<tool_call|>",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--random", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
    sp = None
    if os.path.exists(os.path.join(a.model_dir, "tokenizer.model")):
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor(model_file=os.path.join(a.model_dir, "tokenizer.model"))
    special = list(tok.added_tokens_encoder)

    texts = list(TEXTS)
    if a.random:
        rng = random.Random(a.seed)
        chars = sorted(set("".join(t for t in TEXTS if not any(s in t for s in special)))) + [" "] * 8 + ["\t", "\n"] * 2
        for _ in range(a.random):
            texts.append("".join(rng.choice(chars) for _ in range(rng.randint(1, 40))))

    cases = []
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        if sp is not None and not any(s in t for s in special):
            alone = sp.encode(t)
            if alone != ids:
                raise SystemExit(f"transformers and sentencepiece disagree on {t!r}:\n{ids}\n{alone}")
        cases.append({"text": t, "ids": ids})
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"tokenizer": os.path.basename(os.path.normpath(a.model_dir)), "bos": tok.bos_token_id, "cases": cases}, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"{a.out}: {len(cases)} cases, {sum(len(c['ids']) for c in cases)} tokens")


if __name__ == "__main__":
    main()
