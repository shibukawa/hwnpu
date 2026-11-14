# 作業の分担: アプリに組み込む API と同時通訳のサンプル

作成: 2026-11-11。状態: B0 と B2 の v0、B1 の一部を実装 (2026-11-14、app-api.md の 11 節)。計画は [app-api.md](app-api.md)。この文書は、その計画を複数の機械で手分けして実装するための作業の一覧。

読む順: [app-api.md](app-api.md) → [prebuilt.md](prebuilt.md) の 1〜4 節 → [handoff-ane-cache.md](handoff-ane-cache.md) の 4 節。

## 0. 進め方

- 作業ごとに `main` からブランチを切り、PR を 1 つ出す。終わったら [design.md](design.md) の 16 章に節を足し、この文書の表の「状態」を更新する
- 機械の呼び方:
  - **どこでも**: Go が動く機械。NPU は要らない。CPU の実行器 (`--cpu`、`internal/milcpu`) とモックで確かめる
  - **Mac**: Neural Engine のある Mac。主に M3 (16 GB)。8 GB での確認には M1
  - **aix1**: Ryzen AI 9 365 (Linux、XDNA 2)。IRON の環境がある
- **同時に触らないファイル**: `internal/chat/model.go` (`Prepare`)、`internal/transcribe/prepare.go`、`internal/speech/prepare.go` は、A3、B1、E2 が触る。この 3 つは順に進める (1 節の依存のとおり)
- NPU の確認は、そのとき Neural Engine を使っているほかのセッションがないときに行う。コンパイルが重なると、キャッシュの当たり外れも時間も狂う (handoff-ane-cache.md)

## 1. 一覧

| ID | 作業 | 機械 | 依存 | 規模 | 状態 |
|---|---|---|---|---|---|
| **A1** | Core ML のパッケージを決定的に書く | どこでも、確認は Mac | — | 小 | 未着手 |
| A2 | OpenVINO IR と AMD の書き出しが決定的かを確かめる | どこでも、aix1 | — | 小 | 未着手 |
| **A3** | 変換し直すかどうかを、入力のハッシュ (ビルドのキー) で決める | どこでも | A1 | 中 | 未着手 |
| A4 | models.yaml: 全ファイルの sha256、ライセンスの項目、固定する道具 | どこでも | — | 中 | 未着手 |
| **B1** | `ai`: 登録、計画、準備、置き場の目録、取得元 | どこでも | A3 | 大 | 一部 (登録、計画、準備、目録。取得元と、取得と変換の分割は残り) |
| **B2** | `ai`: 実行 (`Open`、文字起こし、翻訳、読み上げ、音量の VAD) | どこでも、確認は Mac | B1 | 中 | v0 済み (NPU での確認は残り) |
| **B0** | 流れの型 (`Chunk`、`Stream`) と塊をまとめる道具 | どこでも | — | 小 | 済み |
| B3 | `hwnpu-audio` と `hwnpu-chat` を `ai` の上に載せ替える | どこでも | B2 | 中 | 未着手 |
| B4 | アプリのための Neural Engine のロックの方針 | Mac | B2 | 小 | 未着手 |
| B5 | 文字起こしを流しながら (話している途中の仮の字幕と確定) | どこでも、確認は Mac | B2 | 中 | 未着手 |
| B6 | 読み上げを流しながら (できた音から返す、文を受けるたびに読む) | どこでも、確認は Mac | B2 | 小 | 未着手 |
| **D3** | 読み上げを実時間より速くする (今は lut8 で実時間の約 0.5 倍) | Mac | — | 中 | 未着手 |
| C1 | `internal/ws`: WebSocket (RFC 6455) の最小限の実装 | どこでも | — | 小 | 未着手 |
| C2 | `ai/server`: 共通の部分 (互換の口をまとめる Handler、トークン、許すディレクトリ) と独自のストリーム | どこでも | B2、C1 | 中 | 未着手 |
| C3 | `ai/server/openai`: 今の 3 つを移し、音声の 3 つを足す | どこでも | C2 | 中 | 未着手 |
| C4 | `ai/server/googlecloud`: Speech-to-Text、Text-to-Speech、Translation v2、Vision の REST | どこでも | C2 | 中 | 未着手 |
| **D1** | `samples/interpreter`: 同時通訳のサンプル | Mac | C2 | 中 | 未着手 |
| D2 | 測定: 日本語の読み上げ、3 つのモデルの同居、1 発話の遅れ | Mac (M3、M1) | B2 | 小 | 未着手 |
| E1 | ビルドタグと `hwnpu.Target()`、`hwnpu.Built()` | どこでも | — | 小 | 未着手 |
| E2 | `hwnpu_noconvert`: 変換をタグで外せるようにする | どこでも | B1、E1 | 大 | 未着手 |
| E3 | バンドルの形式、`hwnpu-dev bundle`、実行時の照合 | どこでも、確認は Mac | A1、B1、E1 | 中 | 未着手 |
| F1 | `internal/onnx` に LSTM と If | どこでも | — | 中 | 未着手 |
| F2 | Silero VAD を Neural Engine で | Mac | F1、B2 | 小 | 未着手 |
| G1 | Whisper を AMD の NPU で | aix1 | — | 大 | 未着手 |
| G2 | LFM3-Audio の読み上げを AMD の NPU で | aix1 | — | 大 | 未着手 |
| G3 | Silero VAD を AMD の機械で (CPU の見込み) | aix1 | F1 | 小 | 未着手 |
| H1 | 文書: README、CLOUD_AI_COMPATIBILITY、英語版、godoc の例 | どこでも | B3、C2 | 小 | 未着手 |
| **I1** | 決まった形で答える生成 (JSON Schema、`response_format`) | どこでも、確認は Mac | — | 中 | 未着手 |
| I2 | 仕事の API (`/v1/jobs`、`/v1/capabilities`、`/v1/tools`) | どこでも | C2 | 中 | 未着手 |
| **I3** | A2A の口 (Agent Card、JSON-RPC、SSE、push notification) | どこでも | I2 | 中 | 未着手 |
| I5 | `hwnpu-mcp`: MCP しか話さない頭脳のための口 | どこでも | I2 | 小 | 未着手 |
| I4 | 仕事の種類ごとの時間と電力量の目安 | Mac、aix1 | I2 | 小 | 未着手 |

依存のつながり:

```
B0 (済み) ───────→ B2 (流れの形)、B5、B6、C2
A1 ─→ A3 ─→ B1 ─→ B2 ─┬→ B3 ─→ H1
                │       ├→ B4
                │       ├→ D2
C1 ─────────────┼───────┴→ C2 ─┬→ D1
                │              ├→ C3
                │              └→ C4
E1 ─────────────┴→ E2、E3 (A1 も要る)
F1 ─→ F2 (B2 も要る)、G3
G1、G2: 単独で進められる (AMD で同時通訳を動かすのに要る)
```

B0 (流れの型) は単独で始められ、B2 の流れの形、B5、B6、C2 がこれを使う。B5 と B6 は B2 のあと。D3 (読み上げの速さ) は単独で進められる。

I1 (決まった形で答える生成) は単独で進められる。I2 は C2 のあと。I3、I4、I5 は I2 のあと。

すぐに並べて始められるもの: **A1、A2、A4、C1、D3、E1、F1、G1、G2、I1**。B2 の NPU での確認 (Mac)、B5、B6 も、`ai` の v0 の上で始められる。

## 2. A: 決定的な書き出しとビルドのキー

### A1. Core ML のパッケージを決定的に書く

- 問題: 同じ入力で変換し直しても、パッケージのバイトが毎回変わる。コンパイルのキャッシュのキーは中身のハッシュ (`internal/backend/coreml/graph_darwin.go` の `compiledModel`、`cache.TreeDigest` と OS の版) なので、hwnpu の `.mlmodelc` のキャッシュも ANE 自身のキャッシュも外れる。Gemma 4 は起動に 100 秒、SANA の書き出しの作り直しでは 18 分かかった
- 原因は 2 つ (`internal/mlpackage/package.go` の `write`)
  - `Manifest.json` の 2 つの UUID が乱数 (`uuid()` は `rand.Read`)。handoff-ane-cache.md 4 節に記録がある
  - `proto.Marshal` を決定的なモードなしで呼んでいる。Go の protobuf は map のフィールドを順不同で書く。MIL の属性や関数は map なので、`model.mlmodel` も揃わないおそれがある
- 直し方
  - UUID を中身から作る。`model.mlmodel` のバイトと weights の要約のハッシュから、UUID の版 5 の形にする
  - `proto.MarshalOptions{Deterministic: true}.Marshal`
  - `BlobWriter` (重みのファイル) に、map の順や時刻に依るところがないかを確かめる
  - `json.MarshalIndent` の map は鍵の順に並ぶので、そのままでよい
- 合格の条件
  - 小さなモデル (LFM3 の小さい構成、YuNet の ONNX から、Anima の VAE の段の 1 つ) を同じ入力で 2 回変換し、`cache.TreeDigest` が一致するテスト (`internal/mlpackage`、`internal/onnx`、`internal/imagegen/milpart`)
  - Mac: 1 つのモデルを変換し直し、2 回目の読み込みで hwnpu のキャッシュに当たること (`HWNPU_COREML_TRACE=1`)
- 注意: 直したあとの最初の変換は、前の (乱数入りの) 書き出しと別物になる。利用者の機械では一度だけコンパイルし直すことになる。design.md に書く

### A2. OpenVINO IR と AMD の書き出しが決定的かを確かめる

- `internal/ovir/write.go`: 層と辺はスライスの順で書いている。`rtInfo` などに map の順が入っていないかを確かめ、2 回書いて一致するテストを足す
- AMD (`tools/amd/iron_*.py`): 2 回書き出してファイルごとの sha256 を比べる。ELF や xclbin に時刻や乱数が入るなら、書き出しのあとで消すか固定する方法を調べる。直せないものは prebuilt.md の 4 節に書く
- 合格の条件: 違いがあればその場所の一覧と、直したか、直せない理由

### A3. 変換し直すかどうかを、入力のハッシュで決める

- 今: プログラムの名前に形式、設定、版 (`Arch.Revision` の `-r1`、`ExportVersion` の `-v3`) を入れて区別している。出力の変わる修正をしたら、版を手で上げる運用
- 直し方: 「ビルドのキー」を、元のファイルの sha256、変換の設定、アーキテクチャごとの変換器の版から作る。それをプログラムの横の目録 (`*.build.json`) に書き、キーが変わったときだけ変換し直す
  - プログラムの名前は今のまま残す (互換のため)。キーは名前を変えずに変換し直す判断にだけ使う
  - A1 と合わせると、変換し直しても中身が同じならコンパイルのキャッシュに当たる
- 対象: `chat.Prepare`、`transcribe.Prepare`、`speech.Prepare`、`anima.Prepare`、`sana.Prepare`、ONNX のモデル (`detect.Prepare` など)
- 合格の条件: 版を上げずに変換の設定だけを変えたときに変換し直し、何も変えなければ変換しないことを、Prepare ごとのテストで確かめる

### A4. models.yaml: 全ファイルの sha256、ライセンス

- sha256 が付いているのは、今は 43 モデルのうち 15 だけ (Qwen3-1.7B、Sana と SANA 1.5、Anima の 3 つ、MOSS-SoundEffect、ONNX のモデル 7 つ)。LFM3、Gemma 4、Qwen3.5、PLaMo 3、Whisper、DA3、MoGe-2、SigLIP 2 などには付いていない。チェックポイントの全ファイルに付ける
- 固定する道具 `hwnpu-dev pin NAME`: リポジトリのファイルを落とすか、HF の API の LFS の oid を読み、`sha256:` と `from: ...@REVISION` を models.yaml に書く
- ライセンスの項目を足す (`chat.Model` と docs/model-table.md)
  - `license`: SPDX の名前か、`gemma`、`apple-amlr` などの独自の名前
  - `redistribute`: `yes`、`no`、`terms` (規約に従えば可)
  - `attribution`: 表示が要る文 (Magenta RealTime 2 の「Magenta RealTime 2 by Google DeepMind」など)
  - gated: 同意の要るリポジトリかどうか。今は取得の時に分かる
- 合格の条件: `TestFilesPinned` を全モデルに広げる。全モデルに `license` と `redistribute` が付いている

## 3. B: `ai` パッケージ

### B0. 流れの型と塊をまとめる道具

- [app-api.md](app-api.md) の 6.1 節。`Chunk[T]` (中身、標本の数から計算した位置と長さ、区切りの `Final`)、`Stream[T] = iter.Seq2[Chunk[T], error]`
- 道具: `Rechunk` (標本の数や時間で切り直す)、`Sentences` (日本語の句点と読点、英語の文末)、`Finals`、`Tee`、`Async` (前の段を goroutine で回してためる)。必要になったものから足し、増やしすぎない
- 一括の形は、塊が 1 つの流れとして書けることを確かめる
- 合格の条件: 道具ごとのテスト。`Async` は、ctx を取り消すか読む側が `break` したときに、goroutine が残らないこと (`goleak` に当たる確かめ方を、依存を足さずにテストに書く)。`Rechunk` は、入る塊の大きさがどうであっても、標本の数と時刻がずれないこと

### B1. 登録、計画、準備、置き場の目録、取得元

- [app-api.md](app-api.md) の 2、3 節。`ai.NewRegistry`、`Require`、`Plan` (`DownloadBytes`、`Licenses`、変換が要るか)、`Run` (進み具合のコールバック、取り消し、途中からの再開)
- 置き場の目録 `store.json`: 用途の名前をどのモデルにしたか、各モデルのプログラムの道筋、ビルドのキー (A3)
- 取得元: `ai.HuggingFace()` (`chat.Fetch` を包む。同意の要るリポジトリはデバイスフローの URL とコードを、コールバックでアプリに渡す。今は端末に書いている)、`ai.Mirror(baseURL)`、`ai.Source` インタフェース
- 今の `Prepare` を「取得」と「変換」に分ける。transcribe、speech、chat (PLaMo 3 Translate) から始める
- 進み具合は、今の `io.Writer` へのログ (`chat.Fetch` の `progress`) をやめ、構造のある値 (段階、ファイル、バイト) で返す
- 合格の条件: HF とローカルの HTTP サーバー (ミラーのテスト) で、計画 → 準備 → 2 回目の計画で何も要らないことを確かめる。途中で切ってからの再開

### B2. 実行

- `ai.Open` (NPU を 1 回だけ開き、モデルどうしで共有する)、`Transcriber`、`Translator` (PLaMo 3 の文脈: 前の発話と訳の組、`internal/chat/plamo3.go`)、`Speaker`、`VAD` (Go の音量の VAD: しきい値、発話の終わりを決めるまでの待ち時間、前後の余白)
- 準備されていなければ `ErrNotPrepared`。ネットワークにも変換にも触らない
- 各機能の `OnNPU()`、`ai.WithPowerPolicy`
- 合格の条件: CPU の実行器 (`dev == nil`) で、小さな音声の文字起こし、翻訳、読み上げが通るテスト。Mac で 3 つとも `all on the NPU`

### B3. CLI の載せ替え

- `hwnpu-audio transcribe`、`tts`、`hwnpu-chat -m translate` を `ai` の上に載せる。CLI は計画と準備を自動で行ってから動かす (今と同じ動き)
- 合格の条件: 出力が今と同じ (同じ種で、読み上げのバイトが一致する)

### B4. アプリのための Neural Engine のロックの方針

- 今: Core ML のときは、プロセスが生きている間、ファイルのロック (`internal/llm/npulock.go`、`HWNPU_NPU_LOCK=off` で外せる) を持ち続ける。同時通訳のアプリを開いたままにすると、ほかの hwnpu のコマンドやアプリはずっと待つ
- ロックの理由 (`npulock.go` のコメント): 2 つのプロセスが同時に Neural Engine を使うと、M1 の Mac が固まった。予測が返らず、Ctrl+C でも kill -9 でもプロセスが終わらず、次のコマンドもその後ろで止まった。ロックを外すのは危ない
- 決めること (案)
  - ロックはそのまま、プロセスに 1 つとする。その代わり、常駐するアプリには「NPU を手放す」API (`rt.Suspend()` でモデルを閉じてロックを放し、`rt.Resume()` で取り直す) を用意する
  - ほかのプロセスが待っていることをアプリに知らせる。ロックのファイルに待ち手の印を書く
  - 同時に使って固まるのが M1 だけか、M2 と M3 でも起きるかを、壊れてもよい状態で確かめるかどうか (固まると再起動が要る)
- 合格の条件: 方針が決まり、`ai` の文書に書いてある。常駐するサンプル (D1) が、ほかの hwnpu のコマンドの実行をずっと止めないこと

### B5. 文字起こしを流しながら

- Whisper の符号器は 30 秒の窓 (1500 位置) で固定の形。NPU のプログラムも固定の形なので、音を少しずつ足して読むことはできない
- 方式 (最初): **読み直して、一致した先頭を確定する** (whisper_streaming の LocalAgreement)。話している間、まだ確定していない音 (30 秒まで) を 1〜2 秒ごとに読み直す。続けて 2 回同じだった先頭を確定し、残りは仮の字幕として出す。確定した文は、次の読み直しのプロンプト (`<|startofprev|>`) に入れる。これは今の `transcribe` にないので足す
- 費用: 読み直すたびに 30 秒の窓の符号器を回す (M3 で、符号器と交差の段が窓あたり約 0.6 秒)。話している間は ANE の半分ほどを使う。翻訳や読み上げと ANE を取り合うので、間隔は D2 の測定で決める。VAD が静かと言っている間は読まない
- 後の候補
  - 短い窓の符号器: 5 秒と 10 秒のプログラムも作る (whisper.cpp の `audio_ctx` と同じ考え)。速くなるが、短い窓では精度が落ち、プログラムの数だけメモリとコンパイルが増える
  - 流しながら読む前提のモデル (日本語が読めるもの) を調べる。
- 合格の条件: 録音を実時間で流し、仮の字幕の遅れと、確定までの遅れを測る。確定した文を全部つなげたものと、一括で読んだ文の違い (字の誤り率) を出す

### B6. 読み上げを流しながら

- **できた音から返す**: 1 フレームは 80 ms。コーデックの復号は、今も重なりを持たせた塊ごとに回している (`speech.DecodeCodec`、「流しながら復号したのと同じ結果」)。生成の途中で塊ができるたびに音を返すようにする (`Speak` にコールバックかチャネルを足す)。最初の音までの時間は、プロンプトの読み込みと最初の塊の分だけになる
- **文を受けるたびに読む**: LFM3-Audio の TTS のプロンプトは、文が全部そろってから `<|audio_start|>` に進む形。翻訳が流れてくる途中では読めないので、句点、読点、文字数で区切り、区切りごとに読む。区切りをまたぐ抑揚は少し不自然になる。声がそろうよう、会話の間は声と種を同じにする
- 合格の条件: 最初の音までの時間と、途中で音が途切れないかを測る (D3 の速さ次第)

### 読み上げの速さ (D3)

- design.md 16.31 の測定では、LFM3-Audio-2B の読み上げは lut8 で**実時間の約 0.5 倍** (1 秒の音に 2 秒)、lut4 で約 1.4 倍だった。ただし lut4 の音は、ASR に戻すと崩れる。このままでは、流しながら鳴らすと途中で途切れる
- 1 フレーム (80 ms) に、テキスト部 1 回と depthformer 3 回 (12 のコードブックを 4 つずつ)、合わせて 4 回プログラムを呼ぶ。プログラムの切り替えは 1 回 15〜35 ms かかる (16.36) ので、切り替えだけでフレームの時間を使い切るおそれがある
- まずその後の変更 (ロジットを最後の行だけ、GPTQ の lut6 など) を入れた今の速さを測り直す。そのうえで候補を試す
  - depthformer の 3 回を 1 つのプログラムにまとめる
  - テキスト部を GPTQ の lut6 にする。チャットでは KL が下がった (16.87)
  - フレームの中の呼び出しの重なり
- 合格の条件: lut8 の音質 (ASR に戻して一致する) のまま、実時間の 1 倍を超える

## 4. C: ローカルサーバー

### C1. `internal/ws`

- RFC 6455 のサーバーの側: ハンドシェイク、テキストとバイナリのフレーム、分割されたフレーム、ping と pong、close、マスク。拡張 (圧縮) はなし
- 合格の条件: `net/http/httptest` と、テストの中で書いた最小限のクライアントで往復するテスト。大きなフレーム (64 KiB 以上) と分割

### C2. `ai/server`

- [app-api.md](app-api.md) の 7 節のパッケージの分け方。`server.New(rt, opts..., openai.Mount(), ...)` が `http.Handler` を返す。互換のパッケージは `Mount` で口を足すだけにし、共通の部分 (トークン、許すディレクトリ、エラーの形、ログ) はここに置く
- 独自のストリーム `GET /v1/audio/stream` (WebSocket) の決まり (サーバーの文書に書く): クライアントは 16 kHz、モノラル、16 ビットの PCM をバイナリで送る。サーバーは JSON のテキストで、`speech_start`、`speech_end`、`transcript`、`translation` を返し、読み上げは 24 kHz の PCM をバイナリで返す。設定 (言語、訳す先、読み上げるか) は最初のテキストのメッセージで送る。中身は `ai` の流れ (VAD → `TranscribeStream` → `TranslateStream` → `SpeakStream`) をつなぐだけ
- 合格の条件: ストリームは C1 のテストのクライアントで、録音のファイルを流して結果が返る

### C3. `ai/server/openai`

- 今の `internal/chat/server.go` の 3 つ (`/v1/models`、`/v1/chat/completions`、`/v1/embeddings`) を移し、`hwnpu-chat serve` はこれを使う
- 足すもの: `POST /v1/audio/transcriptions` (`response_format` の json、text、verbose_json、srt、vtt)、`/v1/audio/translations`、`/v1/audio/speech` (`response_format` の wav と pcm。mp3 などは符号器が要るので断る)
- モデルの名前: `whisper-1` → `transcribe`、`tts-1` と `gpt-4o-mini-tts` → `speech`。それ以外は models.yaml の名前か用途
- 合格の条件: OpenAI の公式の SDK (Python か JS) で `base_url` だけを変えて、文字起こしと読み上げが呼べる

### C4. `ai/server/googlecloud`

- REST の形で: Speech-to-Text v1 `POST /v1/speech:recognize` (`config.languageCode`、`audio.content` の base64)、Text-to-Speech v1 `POST /v1/text:synthesize` (`LINEAR16` だけ)、Translation v2 `POST /language/translate/v2` (`q`、`target`、`source`)、Vision `POST /v1/images:annotate` (`FACE_DETECTION`、`LABEL_DETECTION`、`TEXT_DETECTION`、`OBJECT_LOCALIZATION`。画像の機能を `ai` に入れてから)
- Google の公式のクライアントは gRPC が既定のものが多い。REST の転送とエンドポイントの差し替えができるクライアント (Python の `client_options={"api_endpoint": ...}` と `transport="rest"`、認証はダミー) で確かめ、使えるクライアントを文書に書く。gRPC は作らない
- 合格の条件: curl の例と、REST の転送の Python のクライアントから、書き起こし、読み上げ、翻訳が返る

## 5. D: 同時通訳のサンプル

### D1. `samples/interpreter`

- Go のプログラムが Web ページを配る。ブラウザでは AudioWorklet で 16 kHz に落とし、WebSocket で送る。返ってきた音声は順に鳴らす
- 画面: 元の言語の文字起こし、訳、状態 (聞いている、訳している、話している)、NPU に載っているか
- 同じことを `ai` パッケージで直接書いた例 (`samples/interpreter/cli`: マイクの代わりに WAV を読み、訳した音声を書く) も置く
- 合格の条件: M3 で、日本語を話すと英語の音声が返る (と、その逆)。遅れを測って README に書く

### D2. 測定

- LFM3-Audio-2B の日本語の読み上げ: 品質 (聞いてみる、Whisper で書き起こして元の文と比べる) と速さ (実時間の何倍か)
- 3 つのモデルの同居: models.yaml の `memory` では合わせて約 6.7 GB (Whisper 1.9、PLaMo 3 Translate 0.8B 2.4、TTS 2.4)。M3 (16 GB) で常駐できるか、M1 (8 GB) で読み込み直しが要るか。M1 の ANE が一度に持てる量 (2.8 GiB、`internal/chat/model.go` の `m1HoldBytes`) を超えて遅くならないか
- 1 発話の遅れの内訳: VAD の終わりの判定、文字起こし、翻訳、読み上げの最初の音まで
- 結果は design.md の節と、`ai.WithReload` (読み込み直し) を既定にするかの判断に使う

## 6. E: ビルドタグとバンドル

### E1. ビルドタグ

- [app-api.md](app-api.md) の 5 節。`backends.go` と `backends_darwin.go` を、バックエンドごとのファイルに分ける。例: `backend_coreml.go` は `//go:build darwin && (hwnpu_apple || !(hwnpu_amd || hwnpu_intel))`
- `hwnpu_apple` を darwin 以外で付けたら、コンパイルを失敗させる
- `hwnpu.Target()` (`apple`、`amd-npu1`、`amd-npu2`、`intel`、`""`) と `hwnpu.Built()`
- 合格の条件: 次の組み合わせで `go build ./...` と `go vet` が通る (GitHub のホストランナーで回せる)。darwin/arm64 の (なし)、`hwnpu_apple`。linux/amd64 の (なし)、`hwnpu_amd`、`hwnpu_intel`。windows/amd64 の (なし)、`hwnpu_amd,hwnpu_intel`

### E2. `hwnpu_noconvert`

- 変換の処理 (`internal/llm` の変換、`internal/onnx` の書き換え、`internal/mlpackage` と `internal/ovir` の書き出し、IRON の書き出しの起動) を、タグの付いたファイルに分ける。タグがあるときは `ErrNoConvert` を返す代わりのファイルを置く
- 合格の条件: `hwnpu_apple,hwnpu_noconvert` のバイナリでバンドルから文字起こしが動く。`go version -m` と `go tool nm` で、変換のパッケージが入っていないこと。バイナリの大きさを比べて書く

### E3. バンドル

- [app-api.md](app-api.md) の 4 節。目録 `hwnpu-bundle.json`、`hwnpu-dev bundle -target T -use ... -memory 8,16 -o DIR`、`ai.Open` の照合 (`ErrBundleTarget`)
- 中身の並びは prebuilt.md の 4 節と同じにする
- `apple` で `.mlmodelc` を入れるのは Mac で作ったときだけ。ほかの機械で作るときは `.mlpackage`
- 再配布できないライセンス (A4 の `redistribute: no`) は失敗にする (`-allow-license` で明示したときだけ通す)
- 合格の条件: Linux で作った `apple` のバンドルを Mac にコピーし、`hwnpu_apple,hwnpu_noconvert` のサンプルが変換もダウンロードもせずに動く

## 7. F: Silero VAD

### F1. `internal/onnx` に LSTM と If

- LSTM: ONNX の `LSTM` (前向き、1 層、ゲートの順 iofc) を MatMul、sigmoid、tanh に分解する。系列の長さを固定して展開する
- If: 条件が定数に畳めるときは、その分岐に置き換える (Silero は標本化の周波数で分かれる。16 kHz に固定する)
- 合格の条件: Silero VAD v5 の ONNX を CPU の実行器で動かし、ONNX Runtime (`internal/onnxrt`) の出力と比べる。一致の度合いを書く

### F2. Silero VAD を Neural Engine で

- 32 ms ごとに NPU を呼ぶと、呼び出しのコストが勝つ。0.5 秒分 (16 ステップ) を 1 つのプログラムに展開し、LSTM の状態を入出力にする
- 合格の条件: `all on the NPU`。音量の VAD と、録音での区切りの違いを比べる。電力を CPU の実行器と比べる (`hwnpu-dev powerbench`)。NPU が負けるなら、AMD の YuNet と同じく CPU を既定にする判断を書く

## 8. G: AMD

AMD では翻訳 (PLaMo 3 Translate) は NPU で動く。しかし文字起こしと読み上げは、今はどちらも AMD で動かない (CLOUD_AI_COMPATIBILITY.md 1 節)。同時通訳を AMD で動かすには G1 と G2 が要る。

- **G1. Whisper**: 符号器は ViT に近い (畳み込み 2 つと Transformer)。DA3 や SigLIP 2 の部品が使える。復号器は交差注意のある LM なので、`iron_export_lm.py` の流れを使う。docs/amd-npu-models.md の形で記録する
- **G2. LFM3-Audio の読み上げ**: 文の部分は LFM3-Audio の書き出し (`lfm3audio`) がある。深さの Transformer (depthformer) とコーデックの復号を足す
- **G3. Silero VAD**: 小さなモデルなので、CPU (自前の実行器か ONNX Runtime) の見込み。電力で NPU と比べる (CLOUD_AI_COMPATIBILITY.md 3 節の YuNet と同じ測り方)

## 9. H: 文書

- README (日本語と英語)、CLOUD_AI_COMPATIBILITY.md (音声のエンドポイントが入ったこと)、MODEL.md
- `ai` パッケージの godoc の例 (`Example_interpreter` など)。外のモジュールから import して使う例を `samples/` に置く

## 10. I: エージェント向けの口

[app-api.md](app-api.md) の 7.1 節。

### I1. 決まった形で答える生成

- `/v1/chat/completions` の `response_format: {type: "json_schema", ...}` と、`ai` の `Chat` の同じ設定
- JSON Schema を文法 (受け付ける状態の機械) に直し、生成の各段で、文法に合わないトークンのロジットを消す。トークナイザーの語彙から「その状態で受け付けるトークン」を前もって引けるようにする (語彙が 25 万の Qwen3.5 や Gemma 4 でも、1 段の上乗せを 1 ms 程度に抑えたい)
- 対象のスキーマ: object、array、string、number、integer、boolean、enum、required。`$ref` や再帰は後で
- ツールの呼び出しの引数にも同じ仕組みを使える (今は書式の解析だけ)
- 合格の条件: CPU のテストで、どのスキーマの出力もスキーマに合う。Mac で、1 トークンの時間の上乗せを測る

### I2. 仕事の API

- 仕事の置き場 (ディレクトリに 1 仕事 1 ファイル)、待ち行列、優先度、取り消し、SSE の進み具合
- 最初の種類は、すでにある機能から: `transcribe`、`translate`、`speak`、`ocr`、`label`、`embed`、`image.generate`。入力は 1 件でも、フォルダと glob でもよい
- 読み書きしてよいディレクトリの設定、トークン
- 合格の条件: 100 枚の画像の OCR を投げ、途中で取り消し、もう一度投げて残りが進む。サーバーを止めて起動し直しても、仕事の状態が残っている

### I3. A2A の口

- [app-api.md](app-api.md) の 7.1 節の対応表のとおり。A2A 1.0 の仕様 (a2a-protocol.org) を読んで、必須の項目を確かめてから作る
- Agent Card: 技能ごとの ID、説明、入出力の種類、例、時間と電力量の目安。既定は 127.0.0.1 だけで待ち受けるので、認証の方式は「なし」と「トークン」の 2 つ
- JSON-RPC over HTTP と SSE。gRPC と HTTP+JSON は作らない。A2A の Go の SDK にも頼らない
- 技能の引数は DataPart (JSON Schema は I2 の `/v1/tools` と同じ)。自然言語だけのメッセージは `rejected`
- `input-required`: 同意の要るモデルのサインイン (デバイスフローの URL とコードを渡す)、ライセンスへの同意
- 合格の条件: 公式の Python の SDK (`a2a-sdk`) のクライアントから、Agent Card を読み、書き起こしの仕事を送り、SSE で進み具合を受け、成果物の文字とファイルを取れる。途中で取り消せる。Google ADK の `RemoteA2aAgent` から呼べるかも確かめる

### I5. `hwnpu-mcp`

- MCP しか話さない頭脳 (Claude Code、エディタ) のための口
- stdio の MCP サーバー。ツールは `capabilities`、`submit_job`、`get_job`、`wait_job` (最大 50 秒待つ)、`cancel_job`。結果のファイルは resource で返す
- Go の MCP の実装は、依存を増やさない方針なら JSON-RPC を自分で書く (stdio なら小さい。I3 の JSON-RPC と共通にできる)
- 合格の条件: Claude Code から `hwnpu-mcp` を登録し、「この録音を書き起こして要点を 3 つ」を頼むと、書き起こしは手元で、要点はクラウドのモデルが作る

### I4. 時間と電力量の目安

- 仕事の種類とモデルごとに、1 件 (1 枚、1 分の音声、1000 文字) の時間と電力量を測って、`/v1/capabilities` で返す。実際に回した仕事の測定で更新する
- 合格の条件: Mac と aix1 で、すでにある機能の目安が埋まっている

## 11. この後の候補 (今回の範囲の外)

部品の追加の順と、サンプルとの対応。

| サンプル | 要る部品 | 状態 |
|---|---|---|
| 擬似立体 (顔の位置と深度で視差) | YuNet、DA3 / MoGe-2 (ある)。頭の位置の推定と深度からメッシュ (Go)。カメラの映像のストリーム (C2 の映像版) | 新しいモデルは要らない |
| 多レイヤー分割 | EfficientSAM (Apache-2.0)、穴埋め (MI-GAN か Anima の VAE の符号器)、RGBA と PSD の書き出し | CLOUD_AI_COMPATIBILITY.md 2.1 節 |
| 超解像 | Real-ESRGAN (BSD-3)。Anima の VAE の符号器があれば、タイルごとの img2img も | 同上 |
| 3D 化 | 1 枚の画像の PLY 出力と、深度からメッシュ (Go)。TripoSR (MIT) | 新しいモデルは TripoSR だけ |
| ボイスチェンジャー | kNN-VC (WavLM、HiFiGAN)。SwiftF0 (ある) | 学習は要らない |
