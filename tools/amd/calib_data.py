#!/usr/bin/env python3
"""Write the calibration and evaluation texts q4_gptq.py reads (CALIB_DATA, ~/models/calib).

  ~/.cache/hwnpu/dev/ref-venv/bin/python tools/amd/calib_data.py [OUT]

From the parquet files on Hugging Face, read with pyarrow:

- wikitext2-train.txt: wikitext-2-raw-v1's train split (Salesforce/wikitext), calibration
- wikitext2-test.txt: its test split from character 200000, evaluation only
- wiki40b-ja-calib.txt: the first half of wiki40b-ja's validation articles (range3/wiki40b-ja),
  cleaned of the markers (_START_ARTICLE_ and the title line, _START_SECTION_, _START_PARAGRAPH_,
  _NEWLINE_), calibration
- wiki40b-ja-eval.txt: its last 200 articles, evaluation only
"""

import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


def column(repo: str, name: str) -> list[str]:
    path = hf_hub_download(repo, name, repo_type="dataset")
    return pq.read_table(path, columns=["text"]).column("text").to_pylist()


def article(raw: str) -> str:
    """A wiki40b article as plain text: its sections' titles and paragraphs, a line each."""
    out, title = [], False
    for line in raw.split("\n"):
        if line == "_START_ARTICLE_":
            title = True  # the next line is the article's title
            continue
        if line in ("_START_SECTION_", "_START_PARAGRAPH_"):
            continue
        if title:
            title = False
            continue
        if line:
            out.append(line.replace("_NEWLINE_", "\n"))
    return "\n".join(out)


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "models" / "calib"
    out.mkdir(parents=True, exist_ok=True)
    wt = "wikitext-2-raw-v1/{}-00000-of-00001.parquet"
    (out / "wikitext2-train.txt").write_text("".join(column("Salesforce/wikitext", wt.format("train"))))
    (out / "wikitext2-test.txt").write_text("".join(column("Salesforce/wikitext", wt.format("test")))[200000:])
    arts = [a for a in map(article, column("range3/wiki40b-ja", "validation.parquet")) if a]
    (out / "wiki40b-ja-calib.txt").write_text("\n\n".join(arts[: len(arts) // 2]))
    (out / "wiki40b-ja-eval.txt").write_text("\n\n".join(arts[-200:]))
    for f in sorted(out.glob("*.txt")):
        print(f"{f}: {len(f.read_text())} characters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
