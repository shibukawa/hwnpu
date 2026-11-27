# hwnpud: NPU とモデルを 1 つのプロセスに

[English](hwnpud.md)

状態: Mac と、AMD の NPU の Linux で段取りの 1 と 3 を実装 (下の「実装したもの」)。残りは設計。hwnpud は macOS と Linux のユーザーごとのデーモンで、NPU とモデルを持つ: 変換済みのモデルを落として持ち、
キャッシュを持ち、プログラムを読み込んで動かす。hwnpu のコマンド (`hwnpu`、`hwnpu-chat`、`hwnpu-image`、`hwnpu-audio`) と
`ai` のパッケージで作ったアプリは Unix ドメインソケットでこれにつなぐ。自分の実行に要るものを渡し、返ってきたものを描くだけで、
自分のプロセスで NPU を動かすこともモデルを管理することもしなくなる。Windows は実機で調べてから。

## 実装したもの

Mac で段取りの 1 (`cmd/hwnpud`、`internal/daemon`):

- `hwnpud serve` は監督役: launchd のソケットを受け取り (`launch_activate_socket`)、なければ
  `HWNPU_DAEMON_SOCKET` か `~/Library/Caches/hwnpu/hwnpud.sock` で待つ (モード 0600、相手の uid を
  `LOCAL_PEERCRED` で確かめる)。`GET /hwnpud/v1/hello` と `status`、`POST /hwnpud/v1/quit` は自分で答え、
  残りはエンジンに渡す。エンジンはそういうリクエストの最初に起こし、終わったらまた起こす。launchd が起こしたときは
  エンジンもリクエストもないまま `HWNPU_DAEMON_IDLE` (5m) たつと終わり、実行ファイルが入れ替わっていれば手が空いた時点で終わる
- エンジン (`hwnpud engine`) はチャットのモデルを持つ: `POST /hwnpud/v1/load` は hwnpu-chat の既定のフラグと同じに
  モデルを用意し、ログ (JSON の行。クライアントの端末がバーを描き直せるかつき)、次にモデルを返す。`/v1/chat/completions`
  はそのモデルの `chat.Server`。usage に hwnpu の統計 (`chat.UsageStats`) を足す。待ち時間で終わり、Neural Engine の
  メモリとロックを返す
- `hwnpu service install` は `~/Library/LaunchAgents/com.github.shibukawa.hwnpud.plist` (ログは `~/Library/Logs/hwnpud.log`)
  を書いて bootstrap する。シェルだけが設定している hwnpu の変数を警告する。`uninstall`、`status`、`stop`
- `hwnpu-chat [chat]` は `HWNPU_DAEMON=on` (か `auto`) でこれを通る: 読み込みのログ、会話、同じ出力と統計の行。
  モデルの変換と動かし方のフラグはそこでは断る (段取り 3 で `ai` のパッケージに移した。下)
- 設定 ([設定](#設定)): `hello` は hwnpud の設定と読んだ `.env` を返す。ファイルが変わると次のエンジンは新しい設定で
  起きる。設定の違うクライアントには知らせ (`on`)、自分で動く (`auto`)。`hwnpu service status` に出す。クライアントの
  `HWNPU_MODEL_ALLOWLIST` は hwnpud の方針を狭める
- M3: PLaMo 3 Translate 2B のプログラムは `hwnpud` の名前で初回 22 秒 (Core AI の項目 2.5 GB)。次の実行は
  モデルがエンジンに残っていて、始めから終わりまで 0.6 秒

Mac で段取りの 3 (`internal/airpc`、`internal/aihost`、`internal/cachecmd`、`ai` のパッケージ):

- `ai` のパッケージを使うプログラムはすべて hwnpud を通る: `HWNPU_DAEMON` は設定しなければ `on`
  (`daemon.DefaultMode`)。hwnpud が答えなければ、どうすればよいかを添えてエラーにする。`HWNPU_DAEMON=off` は
  今までどおり自分のプロセスで動かす。テスト (`go test`) と hwnpud のエンジン自身 (自分で off にする) も自分のプロセス。
  `hwnpu-chat [chat]` も既定で hwnpud を通る
- セッション (`internal/airpc`) はソケットへの接続を HTTP から切り替えたもの (`GET /hwnpud/v1/ai`、
  `Upgrade: hwnpu-ai/1`。監督役のプロキシを通ってエンジンへ)。両方向に gob の流れを 1 本ずつ持ち、電文を ID で
  多重化する。クライアントはエンジンのオブジェクトのメソッドを名前で呼び、型は自分の側の同じメソッドを持つ型
  (モデルの型、`ai.Host`) から取る。引数と結果は gob (`image.Image` の絵は具体的な型で)、`context.Context` は
  取り消しに、関数の引数と、struct の引数の関数の欄 (`imagegen.Options` の `Progress`、`Drawn` ほか) は
  クライアントへのコールバックに、iterator (`flow.Stream`、`iter.Seq`) はどちら向きにもストリームになる
  (回されるにつれて読み、読む側がやめれば止まる)。エラーは登録した番兵と型 (`ai.ErrNotPrepared`、
  `*ai.LicenseError`、`text.ErrContext`、`*text.FormatError`) を保つので、`errors.Is` と `errors.As` が通る。
  エンジンのハンドラーはセッションが終わるまで戻らないので、アプリが使っている間エンジンは待ち時間で終わらない
- エンジン (`internal/aihost`。`cmd/hwnpud` が `engine.Engine.AI` (`internal/daemon/engine`) に付ける。`internal/daemon` はモデルの
  パッケージを import しない) はセッションごとに `ai.Host` を出す: `App.Plan` と `Plan.Run` は、クライアントが送った
  もの (`App.Spec`、`ai.AppFromSpec`: 役の名前による use (`airole.Lookup`)、`InTurns` のグループ、メモリ、
  定義つきのストア) からアプリを作り直して、エンジンで計画し実行する。`Run` の段階とモデルのログは出るそばから
  返り、クライアントが `Run` と同じに描く。`App.Open` はセッションの Runtime を開き、`Use.Get` はそこで use を
  読み込んで、アプリの側の型のモデル (`airole.Role.Remote`) を返す。そのメソッドはエンジンのものを呼ぶ。
  `Release` と `Close` は降ろし、終わったセッションのモデルも降ろす。すべてのセッションの読み込みと実行は順番に。
  クライアントの `HWNPU_MODEL_ALLOWLIST` は `LoadRequest` と同じく hwnpud の方針を狭める
- だから準備、コンパイル (`Run` の読み込みと `Warm`、`CompileOnLoad` の最初の読み込み)、実行は `hwnpud` の名前で
  動く: Core AI のキャッシュは `coreai-cache/<OS>/hwnpud/` だけになり、ストアの `compiled_by` はどのプログラムでも
  `hwnpud`。ある use の名前で役に用意したモデルは、同じモデルの別の use でも用意済みで、どちらかを hwnpud が
  コンパイルしていればコンパイル済み (hwnpu-audio の `whisper-large-v3-turbo` は interpreter の `transcribe`)
- すべての役が hwnpud を通る: depth、faces、label、pose、detect、segment、ocr、image と image-edit、transcribe、
  speak、translate、chat、embed、decide、music、notes、pitch、sfx (それぞれ `airole.Register` で登録し、クライアントの
  側 `Role.Remote` を持つ)。モデルの 1 回ずつの呼び出しでできたストリームのメソッド (`TranscribeStream`、
  `SpeakStream`、`DetectStream`、`CompleteStream`、`Labeler.Label`、`Segmenter.Segment`、ノード) はクライアントで
  その呼び出しの上に動く
- モデルの呼び出しがエンジンに残すもの: 絵のエンコード (`Segmenter.Encode` の `*PictureEncoding`。`Masks` が
  デコードする画像エンコーダーの出力) はハンドルの向こうのエンジンに置いたまま (`airpc.RegisterRef`)。クライアントの
  値は公開の欄 (時間) とハンドルを持ち、それを渡した `Masks` ではエンジンのものに戻る。クライアントの値が GC で
  回収されると (かセッションが終わると) エンジンのものを手放す。ラベルの候補と埋め込み (`Candidates`、`Embedding`)
  はただのデータなのでそのまま渡す。`Rank` と `Probability` はモデルと一緒にエンジンで動く。ストリームが進むにつれて
  埋まる結果 (`MusicGenerator.Generate` の `*MusicStats`) はストリームの終わりと一緒にもう一度送るので、ストリームの
  ループが終わったときには揃っている。途中でやめたループは、iterator のループと同じく作る側が止まるのを待つ
  (作る側は数個先まで進んでいる: ストリームに流量の制御はない)。可変長引数のメソッド (`MusicGenerator.Style`)
  はスライスを送る
- コマンドとサンプルのモデルの仕事はすべて `ai` を通る。自分のプロセスに残るもの: `hwnpu-image convert` (Anima の
  変換。NPU は使わない)、`hwnpu-dev` (開発者のプロセス)、`hwnpu issue`、Neural Engine のメモリのログ
  (`mem.NPUWired` は機械全体のものなので、コマンドはどれだけ増えたかを今も出せる)。コマンドの `--backend` は
  このプロセスの `HWNPU_BACKEND` を設定するが、hwnpud はそれを取らない (自分のものを使うと言う)
- `hwnpu-chat chat` もほかのコマンドと同じく `ai` でモデルを動かす: モデルの変換と動かし方のフラグは断らずに
  アプリの定義 (`ai.Definition`) にして hwnpud が用意し、モデルのファイルはキャッシュのコマンドにわかる。エンジンの
  チャットの道 (`/hwnpud/v1/load`、`/v1/chat/completions`) は OpenAI の API のクライアントのために残す。あきらめた
  もの: その道が読み込んだモデルは待ち時間までエンジンに残り、次の実行はすぐ答えた (0.6 秒)。`ai` では最後の
  クライアントが手放すとモデルを降ろすので、次の実行は Apple のコンパイラーのキャッシュから読み込み直す
  (Qwen3-0.6B で 0.5 秒、始めから終わりまで 1.2 秒)
- モデルはクライアントがいくつでも 1 つ: エンジンの Runtime (セッションに 1 つ) は、同じに読み込むモデル (同じ役、
  モデル、ファイル、動かし方の設定 `LoadSettings`、`AllowOffNPU`) を共有する。最初の `Get` が読み込み、次からは
  それを使い、最後の Runtime が手放すと (`Release`、`Close`、終わったセッション) 降ろす。呼び出しはその型のとおり
  順番に (どれも 1 回に 1 つ)。アプリの `InTurns` は今までどおり、グループのほかの自分の分を手放す。モデルの
  ログは、それを持つ最後のクライアントのものに行く。`hwnpu service status` は持っているモデルと、持つクライアントの数を出す
- キャッシュのコマンドはエンジンで動く (`internal/cachecmd`、同じセッションの `cachecmd.Host`): `hwnpu cache`
  (info、ls、prune、clear、clean) と `hwnpu memory`。エンジンの設定 (シェルのではなく、その `HWNPU_CACHE` とソース)
  で動き、出力と問い (「remove them?」) はクライアントに。`hwnpu status` とアプリの `status` はエンジンのキャッシュを
  出す。だからキャッシュから消すのは hwnpud だけ: `ls` は持っているモデルの読み込み元のユニットに「in use」と書き、
  `clear` と `prune` はそれを残し (「kept: hwnpud holds Qwen3-0.6B loaded, for 2 clients」)、場所を空けるための
  エンジン自身の刈り込みもそれを避ける (`cache.SetInUse`: エンジンは長く動くので、コマンドのプロセスのように一度
  触れたものをずっと残しはしない)。消すものは進んでいる読み込みと実行を待ち、終わるまで次を始めさせない。
  `HWNPU_DAEMON=off` なら今までどおりここで動く
- ゼロの値へのポインタはそのまま渡る (gob は nil として送る): `ChatOptions.Think` が false なら考えない。モデルの
  選びにはならない (以前は `--no-think` でも考えていた)
- まだのもの: アプリに同梱したモデル (`Store.Models.Bundle`)、状態に出すアプリ自身の `Store.Models`
  (エンジンのキャッシュを出す)
- M3 (macOS 27、worktree の hwnpud を別のソケットで、NAS の書き出し): `hwnpu-image depth` の絵 1 枚は初回 60 秒
  (hwnpud が DA3-BASE のプログラムを取ってきて展開し 35 秒、絵の大きさを最初の呼び出しでコンパイルして 24 秒)、
  次は 0.2 秒。`hwnpu-audio transcribe` の日本語 6 秒は初回 75 秒 (Whisper の読み込みのコンパイルが 57 秒)、
  音声は 0.7 秒。そのあと `interpreter models` は `transcribe` をコンパイル済みと出し (hwnpu-audio の
  `whisper-large-v3-turbo`)、`models prepare` は PLaMo 3 Translate 2B と LFM3-Audio-2B-TTS を 3 分 10 秒で用意した。
  `interpreter run` の録音は 3 つのモデルを 0.3、1.3、0.6 秒で読み込み、全部で 7.5 秒。その後の `hwnpu-audio tts`
  は interpreter の音声のモデルを 0.2 秒で読み込み、何もコンパイルしなかった。`parallax models` と `layers models`
  は hwnpu-image が動かした深度と顔のモデルをコンパイル済みと出した。Core AI のキャッシュで増えたのは `hwnpud/`
  だけ (18 項目 2.5 GB から 28 項目 5.5 GB)。`hwnpu-image/`、`hwnpu-audio/`、`interpreter/` は変わらず、
  `parallax/` や `layers/` もできなかった。同じエンジンを通る `hwnpu-chat` は始めから終わりまで 1.3 秒で答えた
- M3、ほかの役 (それぞれ初回、モデルは NAS から): `hwnpu-image label` (SigLIP 2 large) 67 秒、`detect` (RTMDet-m)
  27 秒、`pose` (RTMO-s) 46 秒、`ocr` (PP-OCRv6 small) 14 秒、`segment` (EfficientSAM-S、点 1 つ: エンコードは
  エンジンに置いたままマスクに) 35 秒。`hwnpu-audio pitch` と `notes` はそれぞれ 11 秒、音楽 5 秒の `generate`
  (Magenta RealTime 2。統計はストリームのあとで揃う) 31 秒。`hwnpu-chat search` (PLaMo-Embedding-1B) 61 秒、
  次は 0.3 秒。`hwnpu-chat systemone` (d1-2B) 3 分 34 秒、次は 0.45 秒。`voice-agent models prepare` (Whisper、
  LFM3-1.7B、Supertonic 3) 2 分 49 秒。Core AI の `hwnpud/` は 18 項目から 54 項目に (2.5 GB から 8.9 GB)。
  ほかのプログラムのディレクトリは変わらなかった
- M3、モデル 1 つとキャッシュ: 同時に開いた 2 つの `hwnpu-chat -m Qwen3-0.6B` はモデルを 1 つだけ持った
  (`hwnpu service status`: 「Qwen3-0.6B (chat): loaded once, for 2 clients」。2 つ目の `Get` は 1 つ目のものを使った)。
  `hwnpu cache ls Qwen3-0.6B` はその 2 つのユニットに in use と書き、`clear -y` は残した。2 つとも終わったあとは
  同じ `clear` が消せる

AMD の NPU の Linux でも、同じ監督役とエンジン:

- `hwnpu service install` は systemd のユーザーのユニット `~/.config/systemd/user/hwnpud.socket`
  (`ListenStream=$XDG_RUNTIME_DIR/hwnpud.sock`、モード 0600、`FileDescriptorName=hwnpud`) と `hwnpud.service`
  (`ExecStart=… serve`、`KillMode=mixed`: エンジンは監督役が自分で止める) を書き、`systemctl --user enable --now hwnpud.socket`
  する。監督役は `LISTEN_FDS` からソケットを受け取り (この変数はエンジンに渡さない)、相手の uid を `SO_PEERCRED` で確かめる。
  ログはジャーナル (`journalctl --user -u hwnpud`)。systemd のユーザーのマネージャーは `~/.config/environment.d` の変数を
  サービスに渡すので、`install` が警告するのはそこにも `.env` にもない変数だけ
- AMD の NPU では、エンジンのチャットの道 (`/hwnpud/v1/load`、`/v1/chat/completions`。OpenAI の API のクライアントのために残す) は
  チャットのモデルを一度に 1 つ持つ: 別のモデルの読み込みは持っているモデルを、そのリクエストが終わってから閉じる
  (その間の新しいリクエストには渡さない)。1 つのモデルへのリクエストは順番に動く (`chat.Engine` は生成を一度に 1 つ)。順は来た順。
  `ai` のセッションのモデル (段取りの 3、下) はこれに入らない: エンジンはそれらを並べて持ち (チャットのモデルどうしも)、
  ほかのために閉じることはまだしない (段取りの 4 の予算)
- NPU のロック (`neural-engine.lock`) は AMD でも、チャット、埋め込み、判定のモデルの IRON の書き出しを読むときに取る
  (hwnpud のエンジンでも、`HWNPU_DAEMON=off` で動くプログラムでも)。そのプログラムは `waiting for the NPU: process N (hwnpud) is using it`
  と出して、エンジンが待ち時間で終わるのを待つ。エンジンではどの役のモデルも 1 つのプロセスにある。`HWNPU_DAEMON=off` で
  Whisper、読み上げ、画像のモデルを動かすプログラムはロックなしで NPU を開く (段取りの 4)
- `hwnpu service status` はエンジンの NPU のバッファ (固定される) を出す (`/proc/self/fdinfo` の `drm-total-memory`、`mem.DRMMemory`)
- Ryzen AI 9 365 (`hwnpu service install` のユニットと、別のソケットで `systemd-socket-activate`): LFM3-1.7B の書き出しは初回 2.6 秒で読み、
  17 tok/s で答える。次の実行は読み込みなしで始めから終わりまで 3.1 秒 (固定 1.62 GiB)。3 つのクライアントを同時に出すと
  1 つずつ順に答える (12 秒、24 秒、43 秒)。Qwen3.5-0.8B を読むと先に LFM3-1.7B を閉じる (その後 1.02 GiB)。`hwnpu service stop` でサービスが終わり、次の接続でまた起きる

AMD の NPU の Linux の段取りの 3: Mac と同じコードで、Mac だけのものはない。

- ここでも `HWNPU_DAEMON` は既定で `on`: hwnpud が答えなければ、`ai` パッケージのコマンドはどれも失敗し、することを言う
  (`hwnpu service install`、または `HWNPU_DAEMON=off`)
- エンジンはどのセッションのモデルも、役によらず並べて持つ (チャットのモデル 2 つ、チャットと Whisper など)。コマンドは NPU の上で
  交互に流れ、あるモデルの呼び出しの間に別のモデルの呼び出しが入る。障壁の要るモデルの障壁 (`stepBarrier`、`GraphOptions.Barrier`)
  はそれぞれのグラフが自分のものを動かすので効く
- Ryzen AI 9 365 (2026-11-23、作業ツリーの `hwnpud serve` を別のソケットで、キャッシュの書き出し): hwnpud を通して `hwnpu-chat`
  (LFM3-1.7B は 1.9 秒で読み、自分のプロセスと同じ 16.9 tok/s。Qwen3.5-0.8B と 4B)、`hwnpu-audio tts` と `transcribe`
  (読み上げた文がそのまま書き起こされる。Whisper は 0.5 秒で読む)、`hwnpu-image generate` (FLUX.2 klein、28 秒)、`depth`、
  `segment`、`detect`、`label`、`face-detect`、`hwnpu-chat search` (PLaMo-Embedding-1B)、`hwnpu cache info`、`hwnpu status` が動いた。
  LFM3-1.7B の生成中に 2.6 秒の音声を 3 回 `transcribe` すると 1.0 秒が 2.2 秒ずつになり、チャットの 16.6 tok/s は変わらない。
  チャットのモデル 2 つ (LFM3-1.7B と Qwen3.5-4B、固定 5.35 GiB) を同時に生成させると、単独の 16.6 と 7.7 tok/s が 6.1 と 5.7 tok/s に
  なり、どちらも 3.5 分で終わった (順に動かすのとほぼ同じ)。カーネルのログには何も出ない。大きいモデル (Qwen3.6-35B-A3B) を
  並べるのは試していない
- 途中で見つかり (プログラム自身のプロセス、`HWNPU_DAEMON=off` でも起きる)、直したもの: SigLIP 2 の文のグラフが 2 回目の呼び出しで
  止まった (ERT_CMD_STATE_TIMEOUT。`label` は 444 のラベルを 8 つずつ埋め込む)。画像のモデルと同じく `GraphOptions.Barrier` で直る。
  ONNX Runtime があるときに CPU 向けに準備した YuNet は、それがなくなると読み込めなかった。`airole.Role.Usable` でストアが
  準備されていないと数え、準備し直すと hwnpu の実行器 (`milcpu`。そのために何倍も速くした: [vision-models](vision-models.ja.md)) で動く

## なぜ

### Mac

今プロセスごとに増えるもの (M3、macOS 27。[Core AI](apple-coreai.ja.md)、[Apple の Neural Engine](apple-neural-engine.ja.md)):

| 写し | 鍵 | 増え方 |
|---|---|---|
| Core AI の項目 (`~/Library/Caches/coreai-cache/<OS のビルド>/<実行ファイル名>/`) | 実行ファイル名 | 名前ごとに重みとほぼ同じ大きさをもう 1 回。開発の機械で hwnpu-chat 15 GB、hwnpu-image 15 GB、parallax 8.3 GB、layers 6.5 GB、interpreter 5.2 GB、計 50 GB 前後。layers と parallax は hwnpu-image と同じ深度と画像のモデルを持つ |
| aned のコンパイル済みの ANE のコード (`/Library/Caches/com.apple.aned`) | 署名の識別子 | Go のビルドはどれも ad-hoc の `a.out` なので今は共有している。自分の識別子で署名したアプリは 1 本ごとに別になる |
| ANE の wired メモリ (`mem.NPUWired`) | プログラムを読んだプロセス | `neural-engine.lock` のため一度に 1 プロセス分 |

ロックがメモリの足し算を防いでいるが、プロセスは終わるまでロックを持つ。アプリ A が開いている間、アプリ B は
`waiting for the Neural Engine` と出して止まる。ロックを外せば重みを 2 回分読み、2 つのプロセスが ANE を使うと M1 が固まったことがある。

### Linux (AMD XDNA)

`entries/amd-xdna2` の書き出しはもうディスクに 1 つ。足し算になるのはメモリと、複数のプロセスが NPU を使うときの脆さ ([AMD](amd.ja.md)):

- **固定されるメモリがプロセスごと**。NPU のバッファ (`BO_SHARE`) は全ページが固定され、どのプロセスの RSS にも出ない:
  Gemma 4 E2B で `Unevictable` が 9.3 GB 増える (Ryzen AI 9 365)。同じモデルを開く 2 つのアプリは 2 回分固定する
- **複数のプロセスは脆い**。別のプロセスのコンテキストが終わった直後の `CREATE_HWCTX` はときどき `EINVAL` (5 秒まで再試行)。
  モデルがコンテキストを作る間にファームウェアへ問い合わせると SMU が止まり、再起動しか直らなかった。NPU1 のハードウェア
  コンテキストは 6 個で、1 つのモデルが使い切る
- **優先度**はもうコンテキストごとに当てていて (QoS の LOW / NORMAL / HIGH)、ファームウェアがコマンドの順を決める。hwnpu の
  full ELF はプリエンプションの節を持たない (あればエラーにする) ので、切り替わるのはたぶんコマンド (グラフの 1 回の呼び出し) の
  境目だけで、下の門と同じ粒度。QoS がプロセスをまたいで効くかは測っていない

### どちらも

NPU を使うプロセスを 1 つにすれば、これが全部なくなる:

- ディスクのキャッシュは 1 つ (Mac では `hwnpud` の名前の Core AI のキャッシュ)
- 2 つのアプリが使うモデルも NPU に載せる (固定する) のは 1 回分
- アプリが待つのはプログラムの呼び出し 1 回で、プロセス全体ではない
- 2 つのプロセスが同時に NPU を使うことがない
- モデルを落とし、持ち、消すところが 1 つになり、どの NPU でも同じ API になる

## モデルとキャッシュ

これからは機械でモデルをビルドしない。取得元 ([配布](distribution.ja.md): Hugging Face、NAS、S3) からほぼできあがったものを
落として使う。それをするのが hwnpud で、hwnpu のキャッシュに書くのは hwnpud だけになる。

- **クライアントは描き、hwnpud が決める**。各コマンドと `ai` の cli の `list`、`status`、`models prepare` は要るもの
  (用途、機械のメモリ、アプリの名前) を送り、答えを描く: `chat.ListHave` の行、`ai.Plan`、`ai.Progress` の流れ。`hwnpu` の
  モデルのコマンド (`hwnpu cache`、`hwnpu memory`) も同じ。これらの型がそのまま通信の型になる
- **ダウンロードはみんなで 1 本**。同じモデルを用意する 2 つのクライアントは 1 本のダウンロードを分け合う。クライアントが
  去っても、ほかに待つクライアントがいれば続け、誰もいなくなれば止める
- **プロセスの間のロックがなくなる**。ダウンロード、変換、書き出しのロック (`cache.Lock`) は hwnpud の中の排他になる
- **消すときに使用中を知っている**。`hwnpu cache` の gc と空きの下限までの切り詰めは、載っているものを飛ばす
- **取得元はファイルから読む**。モデルの置き場と取得元のほかの設定と同じ ([設定](#設定))
- **ライセンス**: hwnpud には端末がない。`accept: true` のモデルには条件を添えて「同意が要る」と答え、クライアントが利用者に
  聞いて答えを送る。答えは今のとおり `licenses.json` に。`chat.Policy` (`HWNPU_MODEL_ALLOWLIST`、`policy:`) は hwnpud で当て、
  クライアントのものはそれを狭める ([設定](#設定))
- **まだビルドするもの**: `hwnpu-dev convert` と取得元への書き出しの書き戻しは開発者のプロセスのまま。Mac では prebuilt が
  あっても端末が ANE のコードを仕上げ、項目が消えればコンパイルし直す。これは hwnpud で動かす。でないと別の名前のキャッシュがまたできる
- **コードは 1 つ**。取得とキャッシュはライブラリのままで hwnpud がそれを載せる。プロセスの中で動かす逃げ道 (下) も同じコードを動かす

## 設定

hwnpu の設定は今は環境変数で、プロセスがそれぞれシェルと `.env` (`envfile`: 今のフォルダ、`~/.config/hwnpu/.env`、
機械のもの) から読む。hwnpud ができると 2 つに分かれる。

| 誰のもの | 変数 | hwnpud への届き方 |
|---|---|---|
| hwnpud | モデルの置き場と取得元、機械での動かし方: `HWNPU_CACHE`、`HWNPU_CACHE_MIN_FREE`、`HWNPU_CACHE_MAX_SIZE`、`HWNPU_MODEL_SOURCE*`、`HWNPU_MODEL_HUGGING_FACE`、`HWNPU_MODELS`、`HWNPU_BACKEND`、`HWNPU_DEVICE`、`HWNPU_MEMORY`、`HWNPU_ACCEPT_LICENSE`、`HWNPU_COREAI_*`、認証 (`HWNPU_WEBDAV_*`、`HF_TOKEN`、`HF_ENDPOINT`、`AWS_*`) | `.env` と、hwnpud が起こされたときの環境から。リクエストからは取らない |
| クライアント | その実行に要るもの: モデル、メモリ (`--memory`、タグ)、モデルの方針 (`HWNPU_MODEL_ALLOWLIST`) | リクエストごとに |

- **シェルのものは届かない**。launchd も systemd もログインシェルのプロファイルを読まないので、`~/.zprofile` の
  `HWNPU_MODEL_SOURCE` は hwnpud に届かない。`~/.config/hwnpu/.env` に書く
- **置き場がクライアントのものでない理由**。hwnpud があるのは、機械にモデルの写しが 1 つ、ダウンロードが 1 本、排他が 1 つで
  済むようにするため。自分の `HWNPU_CACHE` を持つクライアントは、ロックと使用中の数えの外に 2 つ目の写しを作る。自分の取得元を
  持つクライアントがいると、アプリが得るものがそのシェル次第になる。認証をソケットに流す理由もない
- **正はファイルで、hwnpud のメモリを書き換えるコマンドではない**。hwnpud は手が空くと終わり、launchd がまた起こすので、
  中だけで変えた値は消える。hwnpud を通らないプロセス (`HWNPU_DAEMON=off`、`hwnpu-dev convert`) も同じファイルを読むので、
  どの道でも同じ設定で動く
- **ファイルを変えれば hwnpud を起こし直さずに効く**。監督役は起こされたときの環境から、ファイルが設定したものを除いて持ち
  (`envfile.Original`)、エンジンを起こすたびにファイルを読み直す (`envfile.Environ`)。5 秒ごとと
  `POST /hwnpud/v1/reload` (`hwnpu config set`) で、hwnpu と Hugging Face と AWS の変数をすべて (起こされたときに読む
  `HWNPU_DAEMON*` は除く) エンジンを起こしたときのものと比べ、変わっていてリクエストがなければエンジンを止める。次のリクエストが
  新しい設定で起こす (ほかにリクエストのないリクエストも同じ)。古い置き場から読んだモデルは持ち越さない。`hello` は次のエンジンが
  受け取る設定 (認証は設定済みかだけ、URL のパスワードは除く、モデルの方針も)、読んだファイル、`pending` (動いているエンジンが
  違う値で持つ名前) を返す
- **hwnpud ではなく hwnpu service**。hwnpud はサービスの管理役が動かすもの (`serve` と、監督役が使う `engine`) だけを持ち、
  利用者は `hwnpu service install|uninstall|status|start|stop|restart` (`internal/servicecmd`) で扱う (`tailscale` と
  `tailscaled` のように)。インストールしてあれば、`start` は監督役 (つなぐとサービスの管理役が起こす) とエンジン
  (`POST /hwnpud/v1/start`) を起こし、`stop` は終わらせ (次の接続でまた起きる)、`restart` は終わるよう頼み
  (`POST /hwnpud/v1/restart`)、サービスの管理役が新しいもの (新しい実行ファイルでも) を起こすのを待って、エンジンを起こす。
  インストールしていなければ、`start` は自分の代わりに `hwnpud serve` をフォアグラウンドで動かす (`exec`。起こされたときの環境から
  ファイルが設定したものを除いて渡し、ファイルは hwnpud が自分で読む)。別のターミナルからの `stop` と `restart` も同じに効き、
  手で動かした監督役の `restart` は、同じプロセスで実行ファイルを動かし直す (`Run` が `ErrRestart` を返し、`serve` が自分を
  exec する): ログはそのターミナルに出続ける。pid が同じでも、`hello` の `started` で新しいものと古いものを見分ける
- **設定の違うクライアントには知らせる**。クライアントは自分の設定 (シェルと、プロジェクトのものも含む `.env`) を `hello`
  のものと比べる。`HWNPU_DAEMON=on` なら通り、hwnpud の設定を使うこととそれを読んだファイルをログに出す。`auto` なら自分の
  プロセスで自分の設定で動く。`hwnpu service status` は設定と、実行したシェルとの違いを出す。`hwnpu service install` はシェルだけが
  設定している変数を警告する
- **方針は狭めるだけで、広げない**。クライアントは `HWNPU_MODEL_ALLOWLIST` を送る (`LoadRequest.allowlist`、
  `X-Hwnpu-Model-Allowlist`)。hwnpud は、自分の方針が許し、定義のファイルの規則とクライアントの項目も許すモデルを動かす
  (`chat.AllowedAlso`)。クライアントはもっと断れるが、その `+nsfw` や `+license` は hwnpud の `-` を覆さない。管理者の規則は
  機械の `models.yaml` (`policy:`) に書けば、どのプロセスも読む。ファミリーからの選びは今は hwnpud の方針だけで決まり、選んだものを
  クライアントが断るとエラーになる (次のモデルにはならない)。下の Accept のような選びまで
- **変えるコマンド**は git config のようにキーで (`envfile.Keys`): 利用者は環境変数の名前ではなくキーで設定を指す。値が 1 つの
  もの (`cache`、`memory` など) は `hwnpu config set|get|remove KEY`、リストは `hwnpu config add|get|remove KEY` で 1 項目ずつ
  変える: `get` が項目に番号を付け、`remove N` がその番号 (か項目そのもの) で消す。`model-source` は探す順に並べた取得元で、
  `HWNPU_MODEL_SOURCE`、`_1`、`_2` … に書き、消すと番号を詰めて書き直す (`add --at 1` で先頭に入れる)。`model-allowlist` と `model-denylist` は
  `HWNPU_MODEL_ALLOWLIST` の `+` と `-` の項目で、符号を付けずに書く (`hwnpu config add model-allowlist nsfw`、
  `hwnpu config add model-denylist license:non-commercial`)。`hwnpu config` は効いている設定とその出どころを、`hwnpu config keys`
  はキーを並べる。値は `~/.config/hwnpu/.env` (`--file` で別のファイル) に、コメントとほかの行を残して書き、どのプロセスも読む。
  変えると hwnpud がすぐ効かせ (`reload`)、ファイルより勝つもの (シェル、先に読むファイル) があれば知らせる。認証を OS の保管庫に
  置く `--secret` (下) はまだ作っていない

決めたこと (2026-11-22。まだ作っていない):

- **認証は OS の保管庫に**。NAS のパスワードとトークンは `.env` から出す。`hwnpu config set --secret NAME` は、Mac では
  Keychain (サービス `hwnpu`、アカウント `NAME` の汎用パスワード) に `/usr/bin/security` で入れる。こうすると項目の
  アクセスリストに載るのは Apple の署名つきの道具になる。hwnpud の ad-hoc の署名は `go install` のたびに変わるので、直接だと
  そのたびに聞かれる。Linux では Secret Service (`secret-tool` か D-Bus、属性 `service=hwnpu name=NAME`)。環境が設定した
  変数は今までどおり勝つ (CI のため)。鍵束が開いていない Linux (画面なし、ログイン前の systemd のユーザーの unit) は
  `~/.config/hwnpu/secrets.env` (モード 0600) に落ちる。ほかの人が読めるモードなら hwnpud は読まない
- **置き場の引っ越しはクリア**。新しい `HWNPU_CACHE` は空から始まり、取得元 (NAS のビルド済みの書き出し、Hugging Face) から
  埋まる。古い根は写さずに消す (`hwnpu cache clean --root OLD`、または hwnpud が使わなくなってから手で)
- **ファミリーからの選びは HTTP の Accept のように**。クライアントは名前 1 つではなく、受け入れるものを好みつきで送る:
  `accept: ["plamo3translate;q=1", "lfm3;q=0.5", "+license:open"]`。方針の項目は絞り込み、名前、ファミリー、機能は q つき。
  hwnpud は機械のメモリに合うファミリーのモデルから、自分の方針とクライアントの accept が許す一番よいものを選び、どれを
  なぜ選んだかを `Content-Type` と `Vary` のように返す。何も残らなければ `406` のように、候補ごとに何が断ったかを返す。
  `HWNPU_MODEL_ALLOWLIST` はその絞り込みの部分になる
- **開発用の変数は整理した** ([変数](variables.ja.md)、#294): スイッチは `HWNPU_DEV` の鍵で、hwnpud はクライアントのものを
  取らない (`Settings` に入れない)。テストは自分のプロセスで、Neural Engine のロックを取って動く

後で:

- **クライアントの定義**。実験のモデルや、hwnpud が動かせるものと互換な別のアーキテクチャの蒸留モデルは、クライアントの
  `models.yaml` に定義する。クライアントはそのファイルをリクエストに添え、hwnpud はそのクライアントの分だけ自分のものの上に
  読む。作ってキャッシュするものは定義で鍵をつけ、同じ名前の 2 つのクライアントのモデルがぶつからないようにする。
  アーキテクチャは hwnpud のビルドが知っているものに限る。新しいものは hwnpud がそれでビルドされるまで `HWNPU_DAEMON=off` で
- **クライアントのビルドの定義**。hwnpud より新しいクライアントは、hwnpud の組み込みの `models.yaml` にないモデルを知って
  いることがある。名前を解くのは hwnpud なので、コマンドの `list` も hwnpud から取る (段取りの 2)。古い hwnpud は次に手が
  空いて終わるときに入れ替わる
- **1 台の Mac の複数の利用者**はそれぞれ自分の hwnpud とキャッシュを持つ。Neural Engine のロックは機械のものなので、2 人目の
  エンジンは 1 人目のものが終わるのを待つ

## 境界: グラフではなくモデル

RPC は `ai` のパッケージの用途の粒度のモデルの呼び出し: `Transcribe(pcm)`、トークンのストリームを返す
`Generate(prompt)`、画像の生成など。

- グラフの 1 回の呼び出しごとの RPC にはしない。chain と runner はキャッシュと状態をプログラムの配列 (Core AI の
  `cai_input_array`、IRON のバンク) に直接書き、同じ配列を複数のプログラムに束ねるので、KV キャッシュ、リング、窓はデーモンの
  中に置くしかない。呼び出しは 1 回 160 µs (M2) か 0.06 ms (XDNA) 以上で、1 トークンに何回もある
- `ai` のパッケージはもう宣言、`Plan`、`Open`、`Get` を分けているので、`ai.Runtime` の裏にリモートのものを同じ API で置く。
  アプリのコードは変わらない
- データは最初はソケットで写す: 1024² の画像で数 MB、音声はもっと小さい。`SCM_RIGHTS` で渡す共有メモリは、大きなテンソルが要るときに

## プロセス

| | Mac | Linux |
|---|---|---|
| サービス | ユーザーごとの LaunchAgent (`~/Library/LaunchAgents/<label>.plist`) | systemd のユーザーの unit (`~/.config/systemd/user/hwnpud.socket` と `.service`) |
| ソケットで起こす | plist の `Sockets`。Go は purego の `launch_activate_socket` で記述子を受け取る | `hwnpud.socket`。Go は `LISTEN_FDS` を読む |
| ソケット | ユーザーのキャッシュの下 (`sun_path` は 104 バイトまで) | `$XDG_RUNTIME_DIR/hwnpud.sock` |
| 相手の確認 | `LOCAL_PEERCRED` | `SO_PEERCRED` |

- **ユーザーごとで、root にしない**: キャッシュ、ライセンスの答え、`models.yaml` はどれもユーザーごと。どちらの NPU も動かすのに root は要らない
- **使われなければ終わる**: 最初の接続で起き、しばらく使われなければ終わるので、何も使っていないときは NPU のメモリ
  (ANE の wired、XDNA の固定) も常駐のメモリも持たない。`hwnpu-chat -idle` (5 分で放す) を全部のモデルに広げたもの
- **監督役と 1 つのエンジン**。Core AI は自分の変換に失敗するとプロセスごと落ち、止まった NPU のコマンドがソケットまで
  道連れにしてはならない。監督役はソケット、列、モデルの管理を持ち、NPU に触らない。エンジンは同じ実行ファイルを起こし直した
  もの (`hwnpud engine`) なので、Mac では名前が同じで同じ Core AI のキャッシュに当たる。エンジンはいつも 1 つで、落ちたら
  監督役が新しく起こし、動いていたリクエストは失敗にする。NPU のメモリはプロセスとともに消える
- **エンジンは NPU にプログラムがある間だけ NPU のロックを持つ**: Mac は今のとおり `neural-engine.lock`、Linux にも同じロック
  (今はチャット、埋め込み、判定のモデルの IRON の書き出しが取る)。使われずに降ろしたら放すので、自分のプロセスで NPU を開く道具 (`hwnpu-dev probe`、`aicheck`、IRON の道具) も
  順番を待って動く

## アプリをまたぐ割り振り

1 つのプロセスにすると、いくつかの上限が全アプリで共有になる:

- **プログラム**: Mac は 1 プロセス 112 本 (M2): 113 本目の読み込みは失敗し、Core AI は黙って GPU で動かす。XDNA は
  ハードウェアコンテキスト (NPU1 は 6 個)。エンジンはこれを数え (Mac は `aneLive`) 予算として扱い、届く前に使っていない順にモデルを降ろす
- **メモリ**: `App.InTurns` を機械全体の 1 つの計画に広げる: 予算 (models.yaml の `memory`)、全クライアントで 1 つに
  共有するモデル、降ろす順
- **優先度はプログラムの呼び出しごと**: エンジンは呼び出し (`cai_run`、IRON のコマンド) の周りの優先度つきの門で NPU を
  次の呼び出しへ渡す。リクエストの周りではない。対話のリクエスト (チャット、音声) が待つのは今動いている呼び出し 1 回で、
  画像 1 枚ではない: Mac の Anima の DiT は 1 ステップが 4 ブロックずつの呼び出し 7 回 (M1 で 1 回約 0.8 秒、ステップ 5.6 秒)、
  VAE は 10 段なので、音声はその間に入る。Anima が ANE に持つのは 0.57〜0.84 GiB だけ (`runner.Ring`) なので、音声のモデルは
  横に載せたままで読み込みを払わない。費用: 大きな DiT の組は別のプログラムの後で切り替えを払い (+50〜70 ms、`aicheck cycle`。
  XDNA のコンテキストの切り替えは 0.46 ms)、画像は間に入った呼び出しの分だけ遅れる。CPU の部分 (サンプラ、トークナイザー) は
  門の外で並んで動く。優先度を守るのもこの門: なければ呼び出しは来た順に並ぶ

## クライアントと版

- `HWNPU_DAEMON=on|off|auto`。設定しなければ `on` (2026-11-23 に決めた): `ai` のパッケージのアプリも hwnpu の
  コマンドも、すべてのプログラムが hwnpud を通り、答えなければ失敗する。`auto` は同じプロトコルで答えればそれを、
  なければ自分のプロセスで。`off` は自分のプロセスで (テストと同じ)。worktree の `go run ./cmd/hwnpu-chat` で
  その worktree のコードを動かすには `HWNPU_DAEMON=off` (`base/hwnpu-chat` / `new/hwnpu-chat` の比べ方も
  そうする) か、worktree の hwnpud を別のソケットで (`HWNPU_DAEMON_SOCKET`)
- `HWNPU_DAEMON_SOCKET` は別のソケットを指す。worktree でビルドした hwnpud を、入れてあるものの横で試すため
- 最初のメッセージ (`hello`) で、プロトコルの版、ビルド、組み込んだモデルの定義とレシピのハッシュ
  (`definitions`: models.yaml と model-recipes.yaml。書き出しとモデルの動かし方を決める)、役割ごとに受け取る設定
  (`roles`: `ai.LoadSettings` の名前、`airole.Role.LoadSettings`) を伝える。2026-11-27 に足した: 11-24 の
  hwnpud が入ったままで、新しいクライアントのモデルを自分の定義で動かし、知らない設定 (`reserve_moe_memory`) を
  黙って捨てていた
  - プロトコルが違う: これまでどおりクライアントが断る
  - `definitions` がない (前の hwnpud) か、クライアントのと違う: クライアントが 1 行で伝え
    (`go install ./cmd/hwnpud && hwnpu service restart`)、`hwnpu service status` も伝える。そのまま hwnpud を通る。
    別の版のモジュールでビルドしたアプリも同じ 1 行を見る。hwnpud は機械のものなので、エラーではなく注意
  - hwnpud の役割が受け取らない設定: クライアントが開く前に止める (役割が受け取る名前と、違えば版のことも)。
    engine も断り (`Open`、`CheckSettings`)、自分のプロセス (`HWNPU_DAEMON=off`) も断る: 名前の書き違いはどこでも
    エラーになる
- supervisor は自分の実行ファイルを見張る: 入れ替わり (`go install`)、engine が動いていなければ終わり、次の接続で
  サービスの管理役が新しいものを起こす。ファイルが入れ替わらなければ古いまま動き続ける: それを伝えるのが上の 1 行
- ソケットはモード 0600、デーモンは相手の uid を確かめる

## 予定: エンジンを持たないクライアント (まだ作っていない)

2026-11-27 に書いた。hwnpu を使うプログラムは、どれもエンジンを丸ごとリンクしている: `hwnpu-chat` は hwnpu の
パッケージ 88 個、hwnpud は 123 個、`ai` のパッケージのアプリも同じくらいに依存する。`HWNPU_DAEMON=off` では
クライアントが自分でモデルを動かす。hwnpud の前はそれがやり方で、テストは今もそうだが、共有を切るだけでない
費用がある:

- 道が 2 つ。同じ機械でも、hwnpud を通すのと自分のプロセスとでモデルの動き方が違いうる。`off` で調べたこと
  (M2 の Anima の VAE、2026-11-27) は hwnpud の道について何も言わない
- 版が見えない。書き出しとモデルの動かし方を決めるのはエンジンのコードと定義で、クライアントは hwnpud のとは
  別に自分のを持つ (上の `definitions` で見分けられるようにした)
- 設定が 2 か所: 自分のプロセスのクライアントはシェルを、hwnpud は .env のファイルを読む
- aned のコンパイルのキャッシュは署名の ID ごとで、Go のビルドはどれも `a.out`: Neural Engine を使うバイナリが
  多いほど、ほかのプログラムを使い回すことが増える (誤った prebuilt の VAE が新しいビルドでも戻った、2026-11-27)
- どのアプリもエンジンと同じ大きさで、shim (Core AI の dylib) を埋め込む

2026-11-27 に決めた: クライアントはモデルの定義を自分では持たず、表、ポリシーの判断、モデルを動かすメモリを
hwnpud に聞く。アプリが自分のプロセスでモデルを動かすのは、エンジンを読むとき (`import _
"github.com/shibukawa/hwnpu/ai/engine"`、hwnpud、hwnpu-dev、テストと同じ) だけで、`HWNPU_DAEMON=off` はそのときの
意味になる。

段はそれぞれ 1 つのプルリクエスト。`internal/boundary` のテストが、エンジンに届いてはいけないパッケージを並べ、
段ごとに足す:

1. 済み: `internal/daemon` はクライアントと supervisor。engine は `internal/daemon/engine` で、cmd/hwnpud が
   つなぐ。読み込みの流れ (`LoadStream`) と答えの手伝いはプロトコルのもので `internal/daemon` に
2. 済み: 定義のデータ (`Model`、`Kind`、ライセンス、ベンダーとチップごとのメモリ、variant) は
   `internal/catalog` にあり、hwnpu のほかのパッケージを読まない。`internal/chat` はその型を別名で使う。
   `internal/airole` は `internal/chat` の代わりに `internal/catalog` を読み、この機械の NPU を探すこと
   (`airole.Here`) はエンジンが入れるフック (`internal/chat`)。クライアントにはない
3. 済み: 定義は hwnpud のもの。hwnpud を通すクライアントが聞くのは 2 つだけなので、ファイル (その読み込みは
   取得の処理と絡んでいる) ではなく、エンジンの `ai.Host` の 2 つのメソッドにした: `Catalog` (`Store.Catalog`:
   モデル、使い方とその選び方、別名、機械のメモリと NPU のベンダー) と `Find` (`Catalog.Find`、役割の確かめも。
   ない名前は `*NoModelError` で返る)。それを持たない hwnpud には新しいものを入れるよう伝える。`ai` のパッケージは
   自分のプロセスの道のために、段 4 まで `internal/chat` を読んだまま
4a. 済み: `ai` のパッケージと `modelstore` はエンジンを読まない。`internal/chat` にしていたこと (表、ポリシー、
   ライセンス、メモリ、取得) は `internal/aieng.Engine` で、`ai/engine` が入れる (`import _
   "github.com/shibukawa/hwnpu/ai/engine"`)。それがないと、自分のプロセスの道はそう伝える (`aieng.ErrNone`)。
   役割のパッケージを分けるまでは、それぞれと `cmd/hwnpu`、`internal/aihost` が `ai/engine` を読むので、
   どのプログラムも変わらない
4. 役割のパッケージ: `Prepare` と `Load` (と、それが返すエンジンの型) を `ai/engine/...` に移して
   `airole.Register` で登録する。クライアントの型は実装を呼ぶ: hwnpud のもの (`Remote`) か、読んであれば
   エンジンのもの
5. コマンド: `hwnpu-chat`、`hwnpu-image`、`hwnpu-audio` がエンジンを読まなくなる。自分のプロセスで
   していること (変換の引数、`--cpu`) は hwnpud を通すか hwnpu-dev に移す
6. エンジンを読まずに `HWNPU_DAEMON=off` はエラーにして、worktree の hwnpud の動かし方を伝える (hwnpu-dev の
   コマンドが別のソケットでビルドして起こす)

最初に書いた段取り:

1. クライアントの側だけにする。役割のパッケージ (`ai/text`、`ai/imagegen`、...) は型、`Remote`、使い方を残し、
   `Prepare` と `Load` (とそれが読むもの) はエンジンのパッケージに移して登録する (`airole.Register`)。それを
   読むのは hwnpud (`internal/aihost`)、hwnpu-dev、テスト。`internal/daemon` のクライアント (`Client`、`Use`、
   `Hello`) を、今はバックエンドを読む supervisor と engine から分ける。`cmd/hwnpu-chat` と `ai` のアプリが
   `internal/llm`、`internal/backend`、`internal/coreai` を読まないことをテストで確かめる
2. コマンドは hwnpud だけを通る。`HWNPU_DAEMON=off` はエンジンをリンクするところ (hwnpud 自身、`hwnpu-dev`、
   `go test`) に残し、ほかではエラーにして worktree の hwnpud の動かし方を伝える
3. 開発: worktree のコードは worktree の hwnpud を別のソケットで (`HWNPU_DAEMON_SOCKET`)。`hwnpu-dev` の
   コマンドがビルドして起こす。試す道と出す道を同じにする

先に確かめること:

- hwnpud は Mac (launchd) と Linux (systemd) で動き、Windows にはまだサービスがない (`install_other.go`):
  Windows の Intel の NPU は段取り 2 の前にそれが要る
- コマンドのどの道も、`off` と同じにエンジンを通して動くか: 特に CPU での確かめ (`--cpu`、hwnpu の CPU の
  インタプリタ) と最初の準備 (`CompileOnLoad`)
- `hwnpu-dev` (convert、prebuild、models package) は自分のプロセスのまま: プログラムを作るもので、配るものではない

## 入れ方と配り方

### 今: Go でビルドする

- `go install ./cmd/hwnpud` (または `go build`) で実行ファイル名は `hwnpud`。Mac ではこれが Core AI の項目の唯一の鍵。
  `go run ./cmd/hwnpud` も一時フォルダから同じ名前で動き、同じキャッシュに当たる
- Mac では Go のリンカーはどの開発ビルドとも同じく `a.out` で ad-hoc に署名するので、aned のコードは共有のまま。`codesign` の手順はない
- `hwnpu service install` は自分のパス (`os.Executable`、たいてい `~/go/bin/hwnpud`) を指すサービスのファイルを書き、起こす
  (`launchctl bootstrap gui/$UID`、`systemctl --user enable --now hwnpud.socket`)。`hwnpu service uninstall` で戻す。
  `go install` でバイナリを入れ替えたら、次に起きたときから新しいもの

### 後で: Go のない人のためのバイナリ

- 実行ファイル名はずっと `hwnpud` のまま。新しい版も Core AI のキャッシュを引き継ぐ
- Mac では ad-hoc の署名でも、有償の Apple Developer の契約なしに識別子を決められる (`codesign -f -s - -i <id>` で
  `Identifier=<id>`、`Signature=adhoc`)。リリースのビルドは、後で Developer ID の署名が使うのと同じ逆ドメインの識別子にする
- Gatekeeper が見るのは quarantine の属性があるファイルだけ:

| 経路 | quarantine | 公証なしで |
|---|---|---|
| `go install` | なし | 動く |
| Homebrew の formula (ソースからか bottle) | なし | 動く |
| `hwnpu service install` や `ai` の cli の `daemon install` (Go で落とし、sha256 を確かめる) | なし | 動く |
| GitHub Releases をブラウザで落とす | あり | 止まる。システム設定の「このまま開く」か `xattr -d com.apple.quarantine` |

- 有償の契約 (年 99 USD、日本では 12,800 円) が要るのは、端末を使わない人に `.app` や `.pkg` を落としてもらうとき、
  アプリからログイン項目を登録するとき (SMAppService)。キャッシュの共有には要らない
- Linux も同じ `daemon install` とリリースのバイナリ。ディストリビューションのパッケージは後で
- アプリは自分の hwnpud を同梱しない。入れてあるものを探し、なければ入れるかを聞く (`ai` の cli の
  `daemon install|status|uninstall`、`models prepare` と同じ見た目)。なければ自分のプロセスで動く

## 先に測ること

どれも小さい。NPU を使うものは裏でロックを待つ。

1. Mac: 同じ実行ファイルから exec した子 (名前が同じで引数が違う) が Core AI のキャッシュに当たるか。測った (M3): 当たる。
   前のエンジンが作った項目から、新しいエンジンが PLaMo 3 Translate 2B を 0.3 秒で読む
2. Mac: aned の鍵が識別子だけか: 同じ識別子を決めた ad-hoc のビルド 2 本が ANE のコードを共有するか。識別子が変わった後の
   最初の起動にどれだけかかるか (ad-hoc の `a.out` → 決めた識別子 → チームつきの Developer ID)
3. Mac: 2 つのアプリのモデルを 1 プロセスに (interpreter と parallax): `mem.NPUWired` と 112 本の上限
4. Mac: 1 つのプロセスで Anima の DiT の呼び出しの間に音声の呼び出しを入れる: 音声の待ちと時間、画像の増える時間、
   Core AI が 2 つのプログラムの `fn.run` を同時に受けられるか。スレッドは問題ではない: `cai_run` のスレッドは
   セマフォで待つだけで、呼び出しは `Task.detached` が Swift のプールで動かす。goroutine のスレッドは関係なく、
   `runtime.LockOSThread` も要らない
5. Linux: 1 つのプロセスに 2 つのモデルを並べる (チャットと音声): 固定されるメモリ、ハードウェアコンテキスト、門で 2 つの
   モデルのコマンドを交互に流しても障壁 (`stepBarrier` / `chunkBarrier`) が効くか。門なしで測った (Ryzen AI 9 365、上の段取りの 3):
   チャットと Whisper、チャットのモデル 2 つがエンジンに並んで動き、呼び出しが交互に流れて結果は正しい。チャットのモデル 2 つは
   NPU の時間をほぼ半分ずつ使った。まだ: 別のモデルの横で、NPU の空きより多くのコンテキストを取るモデル

## 段取り

1. 監督役、エンジン、ソケットで起きる、使われなければ終わる、`install` / `uninstall` の hwnpud を Mac と Linux で。
   チャットのモデルだけで、`hwnpu-chat` をクライアントに (`HWNPU_DAEMON=on`)
2. モデルの管理を hwnpud に: 一覧、状態、計画、進み具合つきの準備、ライセンス、gc。コマンドの `list` / `status` と
   `hwnpu cache` をクライアントに。`hwnpu config`。クライアントの方針をファミリーの選びに渡す
3. `ai.Runtime` のリモートの側と、既定の `HWNPU_DAEMON=on`。samples の interpreter、layers、parallax をデーモン
   経由にし、Core AI のキャッシュが `coreai-cache/<OS>/hwnpud/` だけになること。実装した (上)
4. 画像と音声。機械全体の割り振り (プログラムの予算、降ろす、優先度)。クライアントがいくつでもモデルを 1 つに
   することは実装した (上)
5. リリースのビルド (Mac は識別子を決める)、`daemon install`、Homebrew の tap。有償の契約ができたら公証
6. Windows は実機で調べてから (サービス、ソケット、XRT)

## 採らなかったもの

- **グラフの呼び出しごとの RPC**: KV キャッシュと複数のプログラムに束ねた配列は動かすプロセスの中にあり、1 トークンごとに何回もソケットをまたぐ
- **root のサービス** (LaunchDaemon、システムの unit): キャッシュ、ライセンス、定義はユーザーごと。NPU に root は要らない
- **ロックを外してアプリが同時に ANE を使う**: 重みを 2 回分読み、2 つのプロセスで M1 が固まったことがある ([Apple の Neural Engine](apple-neural-engine.ja.md))
- **XDNA はファームウェアの QoS だけに任せる**: コマンドの順は決まるかもしれないが、プロセスごとにモデルを固定し、複数のプロセスで SMU が止まった
- **取得元をクライアントの環境から渡す**: アプリごとに取得元が変わり、キャッシュの埋まり方も変わる
- **アプリごとに hwnpud を同梱 (SMAppService)**: 署名したアプリが要り、ラベルと版の違う写しがキャッシュをまた割る
- **Homebrew の cask**: cask には quarantine の属性が付き、Homebrew は Gatekeeper を通らない cask をやめていく
