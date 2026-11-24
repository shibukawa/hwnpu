# サポートしていないモデルのメモ

ここには、BonsaiAI がいまサポートしていないモデルのメモを置きます。対象は次の三つです。

- 実装したあとで外したモデル
- コードは残っていても models.yaml の一覧から外したモデル
- 調べただけで実装しなかったモデル

一つのモデルに一つのファイルです。あとで再開するときに、次のことが分かるように書きます。

- どの config で動かしていたか
- どこまでできていたか
- どのコミットにロジックが残っているか
- どんな問題があったか
- 再開するときの手順

設計文書 (`docs/design/`) の `## Not adopted` の節には 1 行の理由だけを書き、くわしいことはここに書きます ([docs/design/README.ja.md](../design/README.ja.md))。

## 状態の書き方

| 状態 | 意味 |
|---|---|
| 検討のみ | 調べたが、実装する前に見送った |
| 削除済み | 実装したが、コードごと消した。ロジックはファイルに書いたコミットにある |
| 一覧から外した | コードはいまもあり、名前やリポジトリで指せば動くことがある。models.yaml の一覧には載せていない |

## 一覧

| ファイル | モデル | 状態 | 日付 | 理由 |
|---|---|---|---|---|
| [chroma1.md](chroma1.md) | Chroma1-Flash、Chroma1-HD | 削除済み | 2026-11-24 | M1 で Apple のコンパイラが DiT のプリビルドで落ちる (1 部品ごとに端末で約 2 時間)。用途は FLUX.2 klein で足りる |
| [sana.md](sana.md) | Sana 600M、SANA 1.5 1.6B、Sana-Sprint | 削除済み | 2026-11-18 (Sprint は 2026-11-11) | Qwen-Image と一緒に外し、8 GB の枠は FLUX.2 klein 4B に。Sprint は 2 ステップで速いが、絵がかなり悪い |
| [qwen-image.md](qwen-image.md) | Qwen-Image-2.1、-Turbo | 削除済み | 2026-11-18 | 7B の DiT と 8B の文の符号器は NPU には大きすぎる (プログラム 17 GB が ANE のキャッシュに入らない)。ライセンスも非商用 |
| [ouro.md](ouro.md) | Ouro-1.4B-Thinking ほか (ループ型の言語モデル) | 一覧から外した | 2026-11-22 | NPU では遅く (3.6 トークン/秒)、KV キャッシュが 4 倍になり、4 ビットにできない |
| [bert.md](bert.md) | BERT の系列 (all-MiniLM-L6-v2 など) | 一覧から外した | 一覧に載せたことはない (移植は 2026-10-29) | WordPiece のトークナイザーがない。埋め込みは PLaMo-Embedding と用途が重なる |
| [kotoba-whisper.md](kotoba-whisper.md) | kotoba-whisper v2.2 | 一覧から外した | 一覧に載せたことはない (確認は 2026-11-08) | 長い音声で崩れるので、書き起こしは whisper-large-v3-turbo にした |

## 検討のみ (ファイルなし)

調べて見送ったモデルの理由は、元の文書に 1 行ずつ書いてあります。再開するときにここへファイルを足します。

| モデル | 書いてある場所 | 見送った理由 |
|---|---|---|
| Anima-Base v1.0 | [docs/design/image-models.ja.md](../design/image-models.ja.md) の「採らなかったもの」 | LoRA の土台として使うもので、BonsaiAI は LoRA を読まない。`iron_anima.py` ではいまも書き出せる |
| LongCat-Image-Edit | 同じ表 | 6.3B の DiT と 16.6 GB の文の符号器で重すぎる |
| OmniGen2 | 同じ表 | 絵の質が低い |
| Inkling 2 Small、Mini | [docs/model-estimation.ja.md](../model-estimation.ja.md) | Small は 594B の MoE で NPU には大きすぎる。Mini は見積もりだけで後回し |
| d1-omni-600M | [docs/design/llm-models.ja.md](../design/llm-models.ja.md) の「採らなかったもの」 | 対象外 |
| MMAudio | [docs/design/audio-models.ja.md](../design/audio-models.ja.md) の「採らなかったもの」、[docs/small-models-survey.md](../small-models-survey.md) | 文の特徴に研究専用の DFN5B-CLIP を使う |
| DA3NESTED、DA3-LARGE、DA3-GIANT | [docs/design/vision-models.ja.md](../design/vision-models.ja.md) | CC BY-NC |
| BiRefNet | 同じ | deformable conv を使う。切り抜きは EfficientSAM にした |

[docs/small-models-survey.md](../small-models-survey.md) で見送ったものは次のとおりです。

| モデル | 見送った理由 |
|---|---|
| SoulX-Singer | 日本語の歌詞を受けない |
| MT3 | T5-small 級 |
| FCPE、RMVPE、CREPE | SwiftF0 を入れた |
| DiffSinger | 声のデータごとにライセンスが違う |
| ACE-Step 1.5 | 1B を超える |
| MusicGen-small | 非商用 |
| Stable Audio Open Small | Stability AI Community License |
| Woosh | 非商用 |
| MiDashengLM-Gen | 約 3B |
| ViTPose、DWPose、RTMW | RTMO を入れた |
| YOLOX、RT-DETR、D-FINE | RTMDet を入れた |
| OWLv2、Grounding DINO | 重い |
| Ultralytics YOLO | AGPL-3.0 |
| PP-OCRv5 | v6 を入れた |

## 評価の記録

モデルを NPU に載せる前の評価 (見積もり、実測、カタログに入れなかった理由) は、[docs/model-estimation.ja.md](../model-estimation.ja.md) に 1 モデル 1 節で書いています。ループ型のモデルの評価は、もとは `docs/looped-models.md` にあり (`9a4f0048`)、いまはそこにまとめています。ここにある一覧から外したモデルのメモは、その評価の結果を受けて書いたものです。
