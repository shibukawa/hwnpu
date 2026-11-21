# hwnpu の設計

[English](README.md)

hwnpu は、CPU に内蔵された NPU でモデルを動かす Go のライブラリとコマンドの集まりである。対象は
Apple の Neural Engine (Core ML、Core AI)、Intel の NPU (Level Zero、OpenVINO IR)、AMD の XDNA
(amdxdna の ioctl、IRON)。cgo なし、GPU なし。ここのファイルは今の設計と今の決定だけを書く。経緯は git に残る。

## 目標

- **NPU で動き、それを保証する。** ONNX Runtime の EP や OpenVINO の AUTO は、未対応の演算を黙って
  CPU や GPU に回す。hwnpu は既定で厳格モードとし、NPU だけで完結しないグラフは実行前にエラーにして、
  載らなかった演算を返す ([API](api.ja.md))。
- **電力を少なく。** 電力ポリシーを第一級の設定にし、バックエンドが許す範囲でエネルギーを読む
  ([メモリと電力](memory-and-power.ja.md))。
- **小さなモデルをうまく。** 1 つのモデルの複数の形 (プロンプト / デコード)、重みの共有、再起動をまたぐ
  コンパイルキャッシュ、メモリに応じた常駐と解放。
- **Hugging Face のモデルを Go で変換。** 対応アーキテクチャは `modeling_*.py` から手で移し、各 NPU の
  形式へ直接書き出す ([アーキテクチャ](architecture.ja.md))。
- **cgo なし、依存は少なく。** ルートのパッケージは Linux と Windows では purego と golang.org/x/sys だけ。

## 非目標

- GPU (NVIDIA、Intel XMX、AMD RDNA、Apple の GPU)。GPU が要るなら別のエンジンを使う。
- データセンター向け NPU (AWS Inferentia、Google TPU など)。
- 変換器への汎用の取り込み (ONNX、TorchScript)。ホワイトリスト外のモデルはベンダーの形式
  (`.mlpackage`、OpenVINO IR、xclbin / ELF) で持ち込む。

## 原則

| 原則 | 意味 |
|---|---|
| NPU だけ | 一部を CPU に移すと少し速くなる場合でも、仕事は NPU に置く。CPU は NPU ができないこと (サンプリング、トークナイザー、小さなホストの手順) だけを行い、NPU の方が CPU より遅いモデルは `models.yaml` (`device: {amd: cpu}`) で CPU に置く |
| GPU なし | バックエンドにも逃げ道にもしない。Core ML の compute units は既定で GPU を含まない |
| 速さより電力 | 合計時間が同じなら NPU に置く。読む重みのバイトを減らす (ビット数、使わない層を省く、MoE)。NPU の無駄を減らす (載りきらず読み直すプログラム、NPU と CPU の交互の実行)。`hwnpu-dev powerbench` で mJ/トークンを測り、測れないときは読む重みのバイト、NPU のメモリ (`mem.NPUWired`)、CPU 使用率で比べる |
| cgo なし | purego、`golang.org/x/sys` の ioctl、実行時に読む小さな C / Swift の shim (`native/xrtshim`、Core AI の `libcaishim.dylib`)。`CGO_ENABLED=0` で全部ビルドできる |
| 既定は厳格 | `PlacementStrict` は NPU で動くと確かめられないグラフを拒む |
| 変換は一度、Go で | 公式の bf16 のチェックポイント (`config.json` + safetensors / GGUF) から各 NPU の形式へ直接。間に ONNX を置かない。量子化済みの公開物は劣化なしに並べ替えられるときだけ使い、量子化し直しはしない |
| 参照と照合する | 移したモデルはどれも、NPU に載せる前に CPU で PyTorch (fp64/fp32) と層ごとに照合する ([新しいモデルのチェックリスト](new-model-checklist.ja.md)) |
| 決定を記録する | 合格条件を満たさないモデルは、理由を書いてから先へ進む。やらないと決めたことも理由と共に残す |

## ファイル

| ファイル | 範囲 |
|---|---|
| [README.ja.md](README.ja.md) | この索引、目標と非目標、原則 |
| [architecture.ja.md](architecture.ja.md) | ベンダーのソフトウェアスタック、設計方針、パッケージ構成、purego / ABI 規約、テストと診断、Go での変換の流れ、モデルのホワイトリスト、残るマイルストーン、参考資料 |
| [api.ja.md](api.ja.md) | 公開の Go API (デバイス、バッファ、GraphRunner、KernelRunner、厳格モード、電力、エラー、オプション)、モデルを部品として組み合わせる、`npuinfo`、`modelstore` |
| [apple-neural-engine.ja.md](apple-neural-engine.ja.md) | どちらのバックエンドでも当てはまる Apple の Neural Engine: 上限、動かさない演算、数値の落とし穴、プロンプトのプログラム、プロセスどうしのロック、メモリ |
| [apple-coreml.ja.md](apple-coreml.ja.md) | Core ML のバックエンド (ベストエフォート): 使っているところ、大きさとキャッシュの上限、層ごとの分割、いくつかの数値 |
| [apple-coreai.ja.md](apple-coreai.ja.md) | Core AI のバックエンド (macOS 27): MIL から `.aimodel`、関数 2 つの `.aimodel`、キャッシュ、Apple の書き出し済みプログラム、NE のコンパイル、Core AI の落とし穴 |
| [intel.ja.md](intel.ja.md) | Intel の NPU: Level Zero、OpenVINO IR、上限、常駐 |
| [amd.ja.md](amd.ja.md) | AMD XDNA: amdxdna ドライバ、ioctl、ERT、xrtshim と Windows、NPU1、全モデルが共有する IRON の基盤とカーネル、電力 |
| [amd-llm.ja.md](amd-llm.ja.md) | IRON による AMD の NPU の言語モデル: 密なモデル、多モーダルの塔、プロンプトとデコードのプログラム、AMD の重み形式 |
| [llm-engine.ja.md](llm-engine.ja.md) | KV キャッシュと段、注意の分割 (paged)、疎な注意、プリフィル、最後の行のロジット、プロンプトの保存、長い文脈 |
| [llm-models.ja.md](llm-models.ja.md) | LFM3 の一族 (VL、Audio、d1)、Gemma 4、PLaMo 3 / Translate / Embedding、MiniLM / BERT、トークナイザー、チャットの形式 |
| [llm-qwen.ja.md](llm-qwen.ja.md) | Qwen3、Qwen3.5 (Gated DeltaNet)、Qwen2.5-Coder |
| [llm-decoding.ja.md](llm-decoding.ja.md) | 投機的デコード (プロンプトの照合、MTP)、考える長さの上限、ループの検出 |
| [moe.ja.md](moe.ja.md) | 専門家の混合: Qwen3.6-35B-A3B (Mac と AMD)、gpt-oss-20b |
| [image-models.ja.md](image-models.ja.md) | NPU をまたぐ画像生成: 一覧、サンプラー、CFG、編集、まとめての生成、アスペクト比、書き出し、残すモデル |
| [image-amd.ja.md](image-amd.ja.md) | AMD の NPU の画像モデル: DiT、VAE、テキストエンコーダの形とカーネル |
| [image-apple.ja.md](image-apple.ja.md) | Apple (Core ML / Core AI) と Intel の画像モデル |
| [vision-models.ja.md](vision-models.ja.md) | 検出、姿勢、OCR、領域分割、深度、顔、SigLIP 2、ONNX だけのモデル |
| [audio-models.ja.md](audio-models.ja.md) | Whisper、TTS、Magenta RT2、MOSS-SoundEffect、SwiftF0、Basic Pitch |
| [distribution.ja.md](distribution.ja.md) | `models.yaml`、キャッシュの置き場、取得元 (フォルダ、WebDAV、S3)、ダウンロード、利用条件つきのリポジトリ、ライセンス、書き出し済みの配布、ディスクの共有 (clone / reflink)、掃除 |
| [hwnpud.ja.md](hwnpud.ja.md) | hwnpud (設計、未実装): Mac と Linux で NPU とモデルをユーザーごとの 1 つのプロセスが持つ。モデルとキャッシュの管理、ソケット、監督役とエンジン、アプリをまたぐ割り振り、入れ方と配り方、署名 |
| [cli.ja.md](cli.ja.md) | コマンド (`hwnpu`、`hwnpu-chat`、`hwnpu-image`、`hwnpu-audio`、`hwnpu-dev` …)、OpenAI の API、OpenCode、端末の表示、Ctrl+C |
| [memory-and-power.ja.md](memory-and-power.ja.md) | エネルギーの測り方、電力ポリシー、メモリの測り方、機械のメモリの区分、常駐 |
| [quantization.ja.md](quantization.ja.md) | 重みの形式 (lut4/6/8、int8、q4/q6/q8、mxfp4)、GPTQ、回転、モデルごとの選択 |
| [new-model-checklist.ja.md](new-model-checklist.ja.md) | 新しいモデルを入れるときのチェックリスト |
| [performance.ja.md](performance.ja.md) | 各改善で得たもの、組み合わせるときの注意、速さの上限、まだ確かめていないこと |

## これらのファイルの直し方

- **記録ではなく今の状態を書く**。変更が書いてあることを覆したら、その文を書き換える。日付つきの項、「X だったが Y にした」、PR の番号は足さない。経緯は git に残る
- **やらないと決めたことも残す**。試してやめた形、外したモデル、採らなかった手法は、ファイルの `## 採らなかったもの` の節に 1 行の理由 (数値が理由ならその数値) と書く
- **変更が属する話題のファイルを直す** (上の表)。新しいファイルは新しい話題のときだけ足し、表にも足す
- **英語を先に、日本語はその訳**。`NAME.md` と `NAME.ja.md` を同じコミットで、同じ見出しと表で直す
- **数値には機械を添える** (「M1 16 GB」「Ryzen AI 9 365」など)。新しい数値が出たら古い数値と並べず置き換える
- コードのコメントやほかの文書からは**ファイルで指す** (`docs/design/llm-engine.md` など)。指す節の番号はない

## docs/ のほかの文書

| ファイル | 内容 |
|---|---|
| [../developer-guide.ja.md](../developer-guide.ja.md) ([English](../developer-guide.md)) | `ai` のパッケージでアプリを作る: 登録、準備、実行、モデルのつなぎ方、制約 |
| [../app-api.md](../app-api.md)、[../app-api-tasks.md](../app-api-tasks.md) | `ai` のライブラリの設計と作業の一覧 |
| [../kv-cache.md](../kv-cache.md) | KV キャッシュの省メモリ化と高速化の手法の調査と、Neural Engine への当てはめ |
| [../model-fit.md](../model-fit.md) | Neural Engine に合うモデルの条件と、少ないメモリで速く賢くする手法 (調査と提案) |
| [../model-table.md](../model-table.md) | `models.yaml` の設計 (モデルの定義、選び方、取得元、NPU ごとの書き出し) |
| [../model-provenance.md](../model-provenance.md) | モデルのライセンスと内容、使ってよいモデルの指定 (`HWNPU_MODEL_ALLOWLIST`) |
| [../cache-sharing.ja.md](../cache-sharing.ja.md) ([English](../cache-sharing.md)) | モデルのキャッシュとその共有の利用者向けの説明 |
| [../cache-sharing-design.md](../cache-sharing-design.md) | 変換済みモデルの配布とキャッシュの共有の設計の記録 (層、取得元、lock) |
| [../prebuilt.md](../prebuilt.md) | 取得元 (NAS) の変換済みモデルを Hugging Face に上げて読み戻す設計 |
| [../known-issues.md](../known-issues.md) | Core AI / coreai-build の既知の問題、hwnpu の回避策、Apple が直したときにすること |
| [../reference-paper.ja.md](../reference-paper.ja.md) ([English](../reference-paper.md)) | 参考にした NPU の論文と、hwnpu の実機で当てはまったか |
| [../amd-npu-models.md](../amd-npu-models.md) | 変換したモデルを AMD の NPU (IRON) で動かす設計 |
| [../amd-xdna2-bench.md](../amd-xdna2-bench.md) | XDNA2 (Strix Point) のデータ型と、IRON のサンプルでの実測 |
| [../intel-npu-models.md](../intel-npu-models.md) | 変換したモデルを Intel の NPU で動かす設計 |
| [../model-estimation.md](../model-estimation.md) | NPU で評価したモデルの記録。models.yaml に入れなかったものも、判断と理由とともに |
| [../small-models-survey.md](../small-models-survey.md) | チャット以外の小さな (1B 以下) モデルの調査: 音程、歌声、音楽、効果音、姿勢 |
| [../image-models-survey-qwen-chroma.md](../image-models-survey-qwen-chroma.md) | Qwen-Image-2.1 と Chroma1 を AMD の NPU へ入れるための下調べ |
| `../handoff-*.md` | 別の機械 (M1〜M4、AMD、Intel) での確認の引き継ぎ: 何を動かし何を記録するか |
