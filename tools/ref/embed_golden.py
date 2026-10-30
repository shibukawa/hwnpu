"""Reference sentence embeddings of PLaMo-Embedding-1B, for checking hwnpu-chat's embedder.

    python embed_golden.py MODEL_DIR OUT.json

MODEL_DIR holds the model with its remote code (modeling_plamo.py, tokenization_plamo.py).
OUT.json holds {"queries": [...], "documents": [...]}, each {"text", "ids", "embedding"}: the
embedding from the model's own encode_query or encode_document (as the model card uses them),
and the ids of the tokenizer with its special tokens (the BOS), for a query with the instruction
before the text. internal/chat's TestPlamoEmbeddingGolden compares hwnpu-chat's with them on a
Mac:

    HWNPU_PLAMO_DIR=MODEL_DIR HWNPU_PLAMO_GOLDEN=OUT.json go test -run PlamoEmbeddingGolden ./internal/chat/

Needs torch, transformers and sentencepiece; modeling_plamo.py is written for transformers 4.47
(see docs/design.md 16.18).
"""
import argparse
import json

# encode_query's instruction, as in internal/chat.QueryPrefix
QUERY_PREFIX = "次の文章に対して、関連する文章を検索してください: "

QUERIES = [
    "PLaMo-Embedding-1Bとは何ですか？",
    "日本の首都はどこですか？",
    "How fast is the Neural Engine?",
]

DOCUMENTS = [
    "PLaMo-Embedding-1Bは、Preferred Networks, Inc. によって開発された日本語テキスト埋め込みモデルです。",
    "東京は日本の首都で、世界有数の大都市です。",
    "最近は随分と暖かくなりましたね。",
    "The Neural Engine runs a 1.7B language model at about 15 tokens per second on an M3.",
    "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("out")
    a = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(a.model_dir, trust_remote_code=True, torch_dtype=torch.float32)
    model.eval()
    out = {}
    with torch.inference_mode():
        for kind, texts, encode, prefix in [
            ("queries", QUERIES, model.encode_query, QUERY_PREFIX),
            ("documents", DOCUMENTS, model.encode_document, ""),
        ]:
            embs = encode(texts, tok)
            out[kind] = [
                {"text": t, "ids": tok(prefix + t)["input_ids"], "embedding": e.float().tolist()}
                for t, e in zip(texts, embs)
            ]
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(QUERIES)} queries and {len(DOCUMENTS)} documents to {a.out}")


if __name__ == "__main__":
    main()
