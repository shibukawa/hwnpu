# モデルのライセンスと内容、使ってよいモデルの指定 (設計案)

作成: 2026-11-15。状態: 1 段 (調査) と 2 段 (models.yaml の `license` と `content`、`BONSAIAI_MODEL_ALLOWLIST` の指定) を実装 ([docs/design/distribution.md](design/distribution.md))。3 段と 4 段は設計案。

## 0. 実装したもの (2026-11-15)

- models.yaml の全モデルに `license` (`class`: `open`、`conditional`、`non-commercial`)、Anima と Chroma1 (のちに外した: [not-adopted/chroma1.md](not-adopted/chroma1.md)) に `content: [nsfw]`
- `BONSAIAI_MODEL_ALLOWLIST` と利用者の models.yaml の `policy:` (2 節の決まりのとおり)。用途は許されたモデルに回し、なければ同じ種類のほかのモデル、名前で指したものはエラー。`bonsaiai-chat list` などが弾いたモデルと理由を出す
- 既定で Anima と Chroma1 を選ばなくなった (`+nsfw` でオプトイン)

お客さんの規程でモデルの選び方がきびしいことがある (商用で使えないライセンスの禁止、NSFW を出すモデルの禁止)。用途の既定 (`categories` の `picks`) が機械のメモリで変わるので、「承認したモデル」を利用者が手で守るのは難しい。そこで、モデルごとにライセンスと内容の印を models.yaml に持たせ、利用者が allow と deny で絞れるようにする。

## 1. 進め方

1. 全モデルのライセンスを調べる (この文書の 3 節、4 節)
2. models.yaml に `license` (全モデル) と `content` を足し、指定 (2 節) で用途の既定と `-m` を絞る。[docs/design/distribution.md](design/distribution.md) に今の状態として書く
3. 第三者の写しを本家の取得元に寄せる (5 節)。まず写しがバイト単位で同じもの
4. RTMO と RTMDet の ONNX への書き出しを自前で持つ (5 節)

## 2. 指定の書き方

環境変数 `BONSAIAI_MODEL_ALLOWLIST` (と `.env`、利用者の models.yaml の `policy:`) に、`+` (許す) と `-` (弾く) の項目をカンマで並べる。

```sh
BONSAIAI_MODEL_ALLOWLIST=-license:non-commercial,-license:conditional
BONSAIAI_MODEL_ALLOWLIST=+nsfw
BONSAIAI_MODEL_ALLOWLIST=+license:open,+license:plamo-community-license
```

| 項目 | 当たるもの |
|---|---|
| `license:ID` | ライセンスの id (`apache-2.0`、`lfm-open-1.0`)、または区分 (`open`、`conditional`、`non-commercial`) |
| `nsfw` | `content: [nsfw]` のモデル |

決まり:

- `-` は `+` より強い (`+license:open,-license:mit` は MIT を弾く)
- `+license` が 1 つでもあれば許可リストになる。どれにも当たらないモデルと、ライセンスの分からないモデル (Hugging Face のリポジトリを `OWNER/NAME` で指したもの) は弾く
- 「重みは非商用、生成物は商用可」(Anima) も `license:non-commercial` に入れる (2026-11-15 に決定)。重みを商用の製品で動かせないため
- `nsfw` だけは既定で弾く。`+nsfw` でオプトインする。`+nsfw` は NSFW を出すモードではなく、「頼んでいなくても NSFW を出しやすいモデル (`content: [nsfw]`) も選んでよい」という意味。生成の振る舞い (ネガティブプロンプトなど) は変えない
- 弾かれたとき: 用途の既定は次の候補に回し、一言出す。候補がなければエラー。`-m`、`-use`、API のリクエストで名前を指したときは、差し替えずにエラー。一覧 (`bonsaiai-chat list` など) は弾かれた理由を出す
- 利用者の models.yaml や `aliases` で足したモデルにも同じ判定をかける

## 3. 調査の結果 (2026-11-15)

Hugging Face の API (`cardData.license`)、モデルカード、リポジトリの LICENSE、部品の config と、開発元のページを読んだ。✓ は一次資料で確かめたもの、△ は一部を推定で補ったもの。

### 3.1 チャット、翻訳、埋め込み、書き起こし、読み上げ

| モデル | ライセンス | 商用 | 確度 |
|---|---|---|---|
| LFM3-1.7B、LFM3-VL-2B、LFM3-Audio-2B (TTS も)、LFM3-Omni-2B、d1-2B、d1-Omni-2B | `lfm-open-1.0` | 条件付き: 年商 1,000 万ドル未満 | ✓ |
| gemma-4-E2B-it、gemma-4-E4B-it | `apache-2.0` | 可 | ✓ |
| Qwen3.5-0.8B、2B、4B、Qwen3-0.6B、1.7B | `apache-2.0` | 可 | ✓ |
| gpt-oss-20b | `apache-2.0` と `USAGE_POLICY` | 可 (法令の遵守のみ) | ✓ |
| plamo-3-translate-0.8b、2b、8b | `plamo-community-license` | 条件付き: 登録、グループの年商 10 億円以下、表示 | ✓ |
| plamo-embedding-1b | `apache-2.0` | 可 | ✓ |
| whisper-large-v3-turbo | `mit` (カードの表記だけ。LICENSE のファイルはない) | 可 | △ |

### 3.2 画像と音の生成

| モデル | ライセンス | 商用 (重み / 生成物) | 部品から引き継ぐもの | NSFW | 確度 |
|---|---|---|---|---|---|
| Anima-Turbo、Anima-Aesthetic | `circlestone-labs-non-commercial` (Anima は v1.2、Anima-Base-v1.0-Diffusers は v1.0) | 不可 / 可 (v1.2 の 2.e。v1.0 にはこの例外がない) | NVIDIA Open Model License (Cosmos-Predict2-2B の派生): 「Built on NVIDIA Cosmos」の表示、安全機構を外すと権利が切れる | likely | △ |
| Chroma1-Flash、Chroma1-HD (外した) | `apache-2.0` | 可 / 可 | FLUX.1 schnell、T5-XXL、FLUX の VAE (どれも Apache-2.0) | likely (「Fully uncensored」) | ✓ |
| FLUX.2-klein-4B | `apache-2.0` | 可 / 可 | Qwen3-4B、FLUX.2 の VAE (Apache-2.0)。9B は非商用なので差し替えない | filtered (学習データの選別、出入りの NSFW フィルタ、透かしと C2PA) | ✓ |
| magenta-realtime-2-small | `cc-by-4.0` (重み)、コードは Apache-2.0 | 可 (表示の義務: 「Magenta RealTime 2 by Google DeepMind」) / 可 | なし | — | ✓ |
| MOSS-SoundEffect-v2.0 (lut8 も) | `apache-2.0` | 可 / 可 | Qwen3-1.7B (Apache-2.0)、DAC 形の VAE (コードは Descript の MIT、重みを誰が学習したかは不明) | — | △ |

### 3.3 認識 (画像、音)

| モデル | ライセンス | 商用 | 今の取得元 | 確度 |
|---|---|---|---|---|
| swift-f0 | `mit` | 可 | 第三者の写し (PyPI の wheel とバイト単位で同じ) | ✓ |
| basic-pitch | `apache-2.0` (NOTICE の同梱が要る) | 可 | 第三者の写し (GitHub と PyPI とバイト単位で同じ) | ✓ |
| RTMO-s | `apache-2.0` | 可 | 第三者の変換 (Xenova)。公式の ONNX とバイト単位で違う | ✓ (元の重みは未確認) |
| RTMDet-tiny、s、m | `apache-2.0` (カードの表記) | 可 | 第三者の書き出し (作り方の記述なし) | △ |
| PP-OCRv6-small | `apache-2.0` | 可 | 公式 | ✓ |
| DA3-SMALL、DA3-BASE、DA3MONO-LARGE、DA3METRIC-LARGE | `apache-2.0` | 可 | 公式 | ✓ |
| moge-2-vits、vitb、vitl | `mit` | 可 | 公式 (第一著者のアカウント) | ✓ |
| YuNet | `mit` | 可 | 公式 (opencv_zoo とバイト単位で同じ) | ✓ |
| siglip2-base-patch16-224、siglip2-large-patch16-256 | `apache-2.0` | 可 | 公式 | ✓ |
| EfficientSAM-Ti、S | `apache-2.0` | 可 | 公式 (第一著者のアカウント、GitHub とバイト単位で同じ) | ✓ |

## 4. 気をつけること

### 4.1 既定のモデルが引っかかる指定

- `-license:conditional` (大きな会社で LFM の年商の上限を超える): 8 GB の機械の chat、vision、audio、speech、decision の既定がすべて LFM3 と d1。chat は同じ種類のほかのモデル (8 GB で Qwen3-1.7B) に回るが、vision、audio、speech、decision は代わりがなくエラーになる。translate (PLaMo 3 Translate、登録と年商 10 億円の上限) も同じ。Gemma 4 E2B の画像と音声 (変換はまだない) が入ると、vision と audio の代わりになる
- `-license:non-commercial`: image の既定 (Anima) が当たる (`+nsfw` のとき。既定では nsfw で弾かれる)

### 4.2 ライセンスの義務

- **PLaMo Community License**: 商用は事前の登録が要る。関係するサイトや UI に「Built with PLaMo」と表示し、派生のモデルの名前に「PLaMo」を入れる。生成物で学習したモデルも同じライセンスになる (第 5 条 3 項、蒸留への継承)。変換したもの (`converted/.../vN`) を取得元で配るときは、ライセンスの同梱が要る
- **LFM Open License**: 年商の上限はグループ全体 (Legal Entity に関連会社を含む)。違反すると自動で終了し、コピーを消す義務がある
- **Anima**: 落とす 2 つのリポジトリのライセンスの版が違い、Diffusers 側の v1.0 には「生成物は商用可」の例外がない。Cosmos から NVIDIA Open Model License の表示の義務も引き継ぐ
- **magenta-realtime-2**: 生成した音楽を使う所に表示が要る (CC-BY 4.0)
- **basic-pitch**: NOTICE の同梱が要る

### 4.3 NSFW

- likely: Anima (学習に `safe, sensitive, nsfw, explicit` のタグ。短いプロンプトで出やすいとカード自身が書く)、Chroma1 (検閲なしをうたう。外した)
- filtered: FLUX.2-klein (学習データの選別と、入出力のフィルタ)
- `content: [nsfw]` (`+nsfw` がないと選ばない) を付けるのは likely の Anima だけ (外した Chroma1 にも付けていた)。頼んでいないのに出してくる (Anima は `safe` を付けても出る) のが問題で、filtered の FLUX.2-klein には付けない

## 5. 本家に寄せる

| モデル | 本家の取得元 | 寄せ方 |
|---|---|---|
| swift-f0 | PyPI の `swift_f0-0.3.0-py3-none-any.whl` の `swift_f0/model.onnx` (今と同じ sha256) | `from:` に PyPI の wheel (zip) の中のファイルを指す形を足す |
| basic-pitch | GitHub の `spotify/basic-pitch` の `basic_pitch/saved_models/icassp_2022/nmp.onnx`、PyPI の `basic_pitch-0.4.0` の wheel (今と同じ sha256) | GitHub の固定のコミットの raw か、wheel の中のファイル |
| RTMO-s | download.openmmlab.com の COCO の `.pth`。公式の ONNX は body7 版だけ (学習データが違い、そのデータセットのライセンスは要確認) | 自前で書き出す (MMDeploy、tools/ に) |
| RTMDet-tiny、s、m | download.openmmlab.com の COCO の `.pth` (公式の ONNX はない) | 自前で書き出す |

- YuNet、EfficientSAM、MoGe-2、DA3、PP-OCR、SigLIP 2 は今の取得元が公式なので変えない。MoGe-2 には公式の ONNX (`Ruicheng/moge-2-*-normal-onnx`) もある
- swift-f0 と basic-pitch の URL:
  - `https://files.pythonhosted.org/packages/07/21/ac6c76119a776e4281da61e22828f992bb2e34b74edfcd3987dca97a6180/swift_f0-0.3.0-py3-none-any.whl`
  - `https://files.pythonhosted.org/packages/99/0e/3a36d22562daeb0ae3c78ac78da3f8dba96543c5576ed57d4acb8ddbab5b/basic_pitch-0.4.0-py2.py3-none-any.whl`
- RTMO と RTMDet の `.pth`:
  - `https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-s_8xb32-600e_coco-640x640-8db55a59_20231211.pth`
  - `https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth`
  - `https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth`
  - `https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth`

## 6. 確かめられなかったこと

- MOSS-SoundEffect の VAE の重みを誰が学習したか、DiT が Wan の重みから始めたか
- Whisper large-v3-turbo のライセンス (カードは MIT、元の large-v3 は Apache-2.0。ファイルはない)
- PLaMo Community License の日本語版と英語版のどちらが正か (日本語版は読んでいない)
- Xenova の RTMO-s がどの重み (COCO か body7 か) から作られたか
