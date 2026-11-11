# クラウドの AI API との対応

[English](CLOUD_AI_COMPATIBILITY.md)

2026-11-11 時点です。クラウドの API が受け持つ機能について、hwnpu のどのコマンドとモデルが代わりになるかをまとめました。チップごとの対応状況と、まだ代わりのない機能、その候補も載せています。比べた相手は次のとおりです。

- Google Cloud (Vision、Speech-to-Text、Text-to-Speech、Natural Language、Translation、Document AI、Vertex AI)
- AWS (Rekognition、Transcribe、Polly、Comprehend、Textract)
- Azure AI (Vision、Speech、Language、Document Intelligence、Content Safety)
- OpenAI API
- Apple の端末上のフレームワーク (Vision、SoundAnalysis)

対応はこの順で進めます: 画像 → 音声 → 言語 → API の口。

## 記号

| 記号 | 意味 |
|---|---|
| ◎ | 実機で動かし、参照と照合して測った |
| ○ | 経路はあるが、実機では確かめていない (Intel は全体がこれ。README の表を参照) |
| CPU | NPU の形がなく、CPU で動く (`--cpu` の CPU インタプリタ、または models.yaml の `device: {amd: cpu}`) |
| — | その NPU では動かない |

チップの欄は次の 3 つです。

- **Apple**: M1〜M4 の Neural Engine (Core ML)
- **Intel**: Core Ultra の NPU (OpenVINO IR、Level Zero)
- **AMD**: Ryzen AI の NPU (IRON、xdna)

## 1. 今あるもの

| 機能 | クラウドでの例 | hwnpu | モデル | Apple | Intel | AMD |
|---|---|---|---|---|---|---|
| チャット、コーディング | OpenAI Chat Completions、Gemini、Bedrock | `hwnpu-chat`、`agent`、`serve` (`/v1/chat/completions`) | LFM3、Gemma 4、Qwen3.5 ほか | ◎ | ○ | ◎ (models.yaml の `amd_export`) |
| 画像についての質問 | GPT の画像入力、Gemini | `hwnpu-chat -m vision` | LFM3-VL-2B、Gemma 4 E2B 画像と音声つき | ◎ | ○ | ◎ |
| 音声についての質問 | GPT の音声入力、Gemini | `hwnpu-chat -m audio` | LFM3-Audio / Omni、Gemma 4 E2B | ◎ | ○ | ◎ |
| 翻訳 | Cloud Translation、Azure Translator、DeepL | `hwnpu-chat -m translate` | PLaMo 3 Translate 0.8B / 2B | ◎ | ○ | ◎ |
| 文の埋め込み | OpenAI Embeddings、Vertex の text embeddings | `hwnpu-chat search`、`/v1/embeddings` | PLaMo-Embedding-1B | ◎ | ○ | — (書き出しがない) |
| 書き起こし | Speech-to-Text、Transcribe、Azure Speech、OpenAI `/v1/audio/transcriptions` | `hwnpu-audio transcribe` (API の口はまだない) | Whisper large-v3-turbo | ◎ | — (Core ML のパッケージだけを書く) | — |
| 読み上げ | Text-to-Speech、Polly、Azure TTS、OpenAI `/v1/audio/speech` | `hwnpu-audio tts` (API の口はまだない) | LFM3-Audio-2B | ◎ | — | — |
| 状態の判定 | (llama.cpp の System One API) | `hwnpu-chat systemone` | d1-2B、d1-Omni-2B | ◎ | — | — |
| 画像生成 | OpenAI Images、Imagen | `hwnpu-image generate` (API の口はまだない) | Anima-Turbo | ◎ | ○ | ◎ |
| 〃 | 〃 | 〃 | Sana / Sana-Sprint | ◎ | — | ◎ |
| 深度、点群、法線 | (クラウドに定番の API はない) | `hwnpu-image depth` | Depth Anything 3、MoGe-2 | ○ (変換はする、測っていない) | — | ◎ |
| 顔の検出 (枠と 5 点) | Cloud Vision の顔検出、Rekognition DetectFaces、Azure Face、Apple Vision | `hwnpu-image face-detect` | YuNet | ○ (測っていない) | — | CPU (NPU でも動くが、エネルギーが CPU の 16〜40 倍。docs/amd-npu-models.md 17) |
| 人の姿勢 (17 点) | Apple Vision の体の姿勢、Video Intelligence の人物検出 | `hwnpu-image pose` | RTMO-s | ◎ | ○ | ◎ (初回に書き出す、docs/amd-npu-models.md 21) |
| 物の検出 (枠とラベル) | Cloud Vision の物体の位置検出、Rekognition DetectLabels、Azure の物体検出 | `hwnpu-image detect` | RTMDet-tiny / s / m | ◎ | ○ | ◎ m (初回に書き出す、docs/amd-npu-models.md 20)。tiny と s は CPU |
| 文字 (OCR) | Cloud Vision の TEXT / DOCUMENT_TEXT_DETECTION、Textract、Azure Read、Apple Vision | `hwnpu-image ocr` | PP-OCRv6 small | ◎ | ○ | CPU |
| ラベル付け (ゼロショット)、画像と文の埋め込み | Cloud Vision の LABEL_DETECTION、Rekognition DetectLabels、Vertex と Azure のマルチモーダルの埋め込み、Apple Vision の分類と特徴量 | `hwnpu-image label`、`hwnpu-image embed` | SigLIP 2 Base / Large | — | — | ◎ (docs/amd-npu-models.md 18) |
| 音高 (F0) | (なし) | `hwnpu-audio pitch` | SwiftF0 | ◎ | ○ | CPU |
| 音符 (MIDI) | (なし) | `hwnpu-audio notes` | Basic Pitch | ◎ | ○ | CPU |
| 音楽生成 | Vertex の Lyria | `hwnpu-audio generate` | Magenta RealTime 2 small | ◎ | — | — (`--cpu` のみ) |
| 効果音 | ElevenLabs Sound Effects | `hwnpu-audio generate -m sfx` | MOSS-SoundEffect v2.0 | ◎ | — | — (`--cpu` のみ) |

AMD の「CPU」の 5 つ (姿勢、検出、OCR、音高、音符) は、どれも ONNX から `internal/onnx` で MIL に下ろす経路のモデルです。この経路は Core ML と OpenVINO IR を書きますが、AMD 向けには書き出しません。AMD の機械では CPU で動きます。hwnpu の CPU 実行器 (外部依存なし。RTMDet-m は 1 スレッドで 1 枚 2.3 秒) か、ONNX Runtime の共有ライブラリがあればそれ (ONNX のファイルの同じ切り口、`onnxrt.Cut`。1.4 秒) で (docs/amd-npu-models.md 19)。ONNX の畳み込みのグラフを IRON のグラフにする `tools/amd/iron_onnxcnn.py` ができ、RTMDet-m と RTMO-s はその書き出し (初回に作る) で AMD の NPU で動きます (docs/amd-npu-models.md 20、21)。RTMDet の tiny と s は CPU のまま (和の打ち消し合いが bfp16 を超える、20.4)、OCR、音高、音符も CPU です。

## 2. まだないもの

### 2.1 画像 (次に進める)

| 機能 | クラウドでの例 | 候補 (ライセンス) | hwnpu での形 (案) | AMD への向き |
|---|---|---|---|---|
| **物のマスク** (インスタンスのセグメンテーション) | Apple Vision の前景のインスタンスのマスク、Vertex AutoML | RTMDet-Ins (Apache-2.0) | `hwnpu-image detect --mask` | △ 小さな CNN で演算が多い。YuNet と同じく CPU に負けるおそれがある。汎用の ONNX → IRON を作ってから測る |
| **背景の除去**、被写体の切り抜き | Apple の被写体の切り抜き、remove.bg | BiRefNet / BiRefNet_lite (MIT)。RMBG-2.0 は非商用なので使えない | `hwnpu-image cutout` (透明な PNG) | △ 骨格 (Swin) は GEMM が中心だが、デコーダのどのブロックにも ASPPDeformable があり (BiRefNet_lite の設定も)、torchvision の deform_conv2d を使う。タップごとに、ネットワークが計算したずれの位置を双線形で読むので、決まった格子のない gather になる。コマンドの間のホストの処理か、密な書き直しが要る |
| 点や箱で指定する切り抜き | Meta SAM | SAM 2、EfficientSAM (Apache-2.0) | `hwnpu-image segment --point x,y` / `--box` | ○ 画像の符号器は ViT で向く。軽いデコーダはプロンプトごとに CPU でもよい |
| **文書のレイアウト、表、ページと行の向き** | Document AI、Textract AnalyzeDocument、Azure Document Intelligence の Layout | PP-DocLayout、SLANet、PP-LCNet の向き判定 (Apache-2.0) | `hwnpu-image ocr --layout` / `--table`。向きは自動で直す | △ CNN。物のマスクと同じく、汎用の経路の後に |
| 顔の照合 (同じ人か) | Rekognition CompareFaces / SearchFaces、Azure Face Verify | SFace (OpenCV Zoo、Apache-2.0)。insightface の重みは非商用なので使えない | `hwnpu-image face-detect --embed`、`face-compare` | CPU の見込み (小さな CNN) |
| 不適切な画像の判定 | Cloud Vision の SafeSearch、Rekognition DetectModerationLabels、Azure Content Safety | Falconsai/nsfw_image_detection (Apache-2.0、精度は要確認)、または SigLIP 2 のゼロショット | `hwnpu-image moderate` | ◎ 向く (ViT)。SigLIP 2 は動く (そのラベルで `hwnpu-image label`)。閾値はまだデータで決めていない |
| 高解像度化 | Vertex の Imagen の拡大 | Real-ESRGAN (BSD-3) | `hwnpu-image upscale` | ○ 大きな地図の畳み込みで、im2col の GEMM が大きい |
| 物を消す (インペイント) | Vertex の Imagen の編集 | LaMa (Apache-2.0) | `hwnpu-image erase --mask` | △ FFT がある。DFT を行列積に直す必要がある (Apple も同じ) |
| バーコード、QR | Apple Vision、ML Kit | gozxing (Apache-2.0、Go。ディープラーニングではない) | `hwnpu-image barcode` | チップによらない (ホスト) |
| 動画での追跡 | Video Intelligence、Rekognition Video | ByteTrack (MIT。新しいモデルは要らない) | `hwnpu-image detect --track` (連番の画像) | detect が動くチップで |

扱わないもの:

- ロゴ、名所、有名人、Web 上の同じ画像の検索: 大きなデータベースが要るので、端末のモデルには向きません。
- 顔の表情、年齢、性別の推定: 偏りの問題があり、Azure も提供をやめたので入れません。

### 2.2 音声

| 機能 | クラウドでの例 | 候補 (ライセンス) | hwnpu での形 (案) |
|---|---|---|---|
| **声のある区間 (VAD)** | Speech-to-Text と Transcribe の内部、WebRTC VAD | Silero VAD (MIT) | `hwnpu-audio vad`。書き起こしの前に区間を切る |
| **話者の分離** (誰が話したか) | Speech-to-Text のダイアライゼーション、Transcribe の話者ラベル、Azure Speech | pyannote segmentation-3.0 (MIT、ダウンロードに同意が要る)、WeSpeaker / 3D-Speaker (Apache-2.0) | `hwnpu-audio transcribe --speakers` |
| 音の分類 (拍手、犬、サイレン) | Apple の SoundAnalysis | YAMNet (Apache-2.0、TFLite で `internal/tflite` に乗る) | `hwnpu-audio classify` |
| 音源の分離 (声と伴奏) | (専門のサービス) | Demucs (MIT、大きさは要確認) | `hwnpu-audio separate` |
| 雑音の除去 | (NVIDIA Maxine ほか) | DeepFilterNet (MIT / Apache-2.0) | `hwnpu-audio denoise` |

### 2.3 言語

| 機能 | クラウドでの例 | 候補 (ライセンス) | hwnpu での形 (案) |
|---|---|---|---|
| **リランク** | Cohere Rerank、Vertex の Ranking API、Bedrock の rerank | ruri の reranker (要確認)、bge-reranker-v2-m3 (Apache-2.0) | `hwnpu-chat search --rerank`、`/v1/rerank` |
| 固有表現、感情、分類 | Cloud Natural Language、Comprehend、Azure Language | チャットモデルで足りる。小さく速くするなら GLiNER (Apache-2.0、版ごとに要確認) | 急がない |
| 文の不適切さの判定 | OpenAI Moderations、Azure Content Safety | チャットモデルに判定させる | 急がない |

### 2.4 API の口

`hwnpu-chat serve` の口は `/v1/models`、`/v1/chat/completions`、`/v1/embeddings` の 3 つです (`internal/chat/server.go`)。モデルはあるのに、口がない機能があります。

- `/v1/audio/transcriptions` (Whisper)、`/v1/audio/speech` (LFM3-Audio)、`/v1/images/generations` (Anima、Sana)
- 画像の解析の口 (Cloud Vision の `images:annotate` のように、1 回で検出、OCR、顔、姿勢、ラベルを返す)
- `/v1/rerank` (2.3 のリランクを入れた後)

## 3. AMD から進めるときの注意

AMD には公開されたグラフ用のコンパイラがなく、モデルごとに IRON のグラフを Python で組みます (`tools/amd/iron_*.py`。docs/amd-npu-models.md 1、2.3)。測ってわかった速さの決まり方 (同 17.2):

- GEMM とほかの演算を交互にすると、切り替え 1 回に約 250 µs、エネルギーで約 1 mJ かかります。同じ演算が続くなら 1 つ 5〜30 µs です。
- そのため、演算が多く 1 つずつが小さい CNN は CPU に負けます。YuNet (約 70 演算) は 1 枚 127 mJ で、CPU の 1 スレッドなら 7.7 mJ でした。
- ViT や Transformer、大きな畳み込み (im2col の GEMM) は向いています。DA3、MoGe-2、Sana、Anima はこの形で動いています。

画像の項目を AMD から進めるなら、次の順を勧めます。

1. **SigLIP 2** (ラベル付け、画像の埋め込み、不適切な画像の判定): ViT なので DA3 の部品が使えます。これ 1 つで、2.1 の 3 つの項目が埋まります。ラベル付けと埋め込みはできました (2026-11-11、docs/amd-npu-models.md 18)。その間に、1 節の ONNX の 5 つは AMD の CPU で ONNX Runtime で動くようにしました (同 19)。
2. **SAM 系の符号器と BiRefNet** (切り抜き): 骨格が重く GEMM が中心です。BiRefNet の deformable conv はデコーダのどのブロックにもある (2.1) ので、EfficientSAM (1024 × 1024 の素の ViT-S / T、Apache-2.0) から始めます。
3. **ONNX → IRON の汎用の経路**: 1 節の 5 つ (姿勢、検出、OCR、音高、音符) と、RTMDet-Ins、文書のレイアウトが載ります。モデルごとに NPU と CPU の mJ を比べ、負けるものは models.yaml を `device: {amd: cpu}` のままにします (YuNet と同じ扱い)。RTMDet-m と RTMO-s はできました (`tools/amd/iron_onnxcnn.py`、2026-11-11、docs/amd-npu-models.md 20、21)。RTMDet の tiny と s は負けました (20.4)。
4. **バーコード、追跡**: チップによらないので、いつでも入れられます。

## 4. 確かめていないこと

- 候補のライセンスと大きさは、調べた時点の記載と記憶にもとづきます。入れる前に、重みのファイルとモデルカードで 1 つずつ確かめます (docs/new-model-checklist.md 1)。
- Intel の「○」は、変換の経路があるだけです。実機では動かしていません。
- Apple の深度と顔の検出は変換しますが、M1 で速さとエネルギーを測っていません。
