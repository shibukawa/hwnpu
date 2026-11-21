# hwnpud: NPU とモデルを 1 つのプロセスに

[English](hwnpud.md)

状態: Mac で段取りの 1 を実装 (下の「実装したもの」)。残りは設計。hwnpud は macOS と Linux のユーザーごとのデーモンで、NPU とモデルを持つ: 変換済みのモデルを落として持ち、
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
- `hwnpud install` は `~/Library/LaunchAgents/com.github.shibukawa.hwnpud.plist` (ログは `~/Library/Logs/hwnpud.log`)
  を書いて bootstrap する。シェルだけが設定している hwnpu の変数を警告する。`uninstall`、`status`、`stop`
- `hwnpu-chat [chat]` は `HWNPU_DAEMON=on` (か `auto`) でこれを通る: 読み込みのログ、会話、同じ出力と統計の行。
  モデルの変換と動かし方のフラグはそこでは断る
- M3: PLaMo 3 Translate 2B のプログラムは `hwnpud` の名前で初回 22 秒 (Core AI の項目 2.5 GB)。次の実行は
  モデルがエンジンに残っていて、始めから終わりまで 0.6 秒

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
- **取得元はファイルから読む**。launchd も systemd もログインシェルのプロファイルを読まないので、`~/.zprofile` の
  `HWNPU_MODEL_SOURCE` は hwnpud に届かない。取得元は `~/.config/hwnpu/.env` か `models.yaml` に書く。クライアントの環境は
  取得元に使わない: アプリごとに取得元が変わってしまう
- **ライセンス**: hwnpud には端末がない。`accept: true` のモデルには条件を添えて「同意が要る」と答え、クライアントが利用者に
  聞いて答えを送る。答えは今のとおり `licenses.json` に。`chat.Policy` (`HWNPU_MODEL_ALLOWLIST`、`policy:`) は hwnpud で当てる
- **まだビルドするもの**: `hwnpu-dev convert` と取得元への書き出しの書き戻しは開発者のプロセスのまま。Mac では prebuilt が
  あっても端末が ANE のコードを仕上げ、項目が消えればコンパイルし直す。これは hwnpud で動かす。でないと別の名前のキャッシュがまたできる
- **コードは 1 つ**。取得とキャッシュはライブラリのままで hwnpud がそれを載せる。プロセスの中で動かす逃げ道 (下) も同じコードを動かす

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
- **エンジンは NPU にプログラムがある間だけ NPU のロックを持つ**: Mac は今のとおり `neural-engine.lock`、Linux にも同じロックを
  足す (今はない)。使われずに降ろしたら放すので、自分のプロセスで NPU を開く道具 (`hwnpu-dev probe`、`aicheck`、IRON の道具) も
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

- `HWNPU_DAEMON=on|off|auto`。`ai` のパッケージのアプリは `auto`: デーモンが同じプロトコルで答えればそれを、なければ
  今のとおり自分のプロセスで。hwnpu のコマンドは指定がなければ `off`。worktree の `go run ./cmd/hwnpu-chat` が
  その worktree のコードで動き、`base/hwnpu-chat` / `new/hwnpu-chat` の比べ方も崩れない
- `HWNPU_DAEMON_SOCKET` は別のソケットを指す。worktree でビルドした hwnpud を、入れてあるものの横で試すため
- 最初のメッセージでプロトコルの版とビルドを伝える。クライアントより古いデーモンには手が空いたら終わってもらい、
  次の接続でサービスの管理役が新しいバイナリを起こす
- ソケットはモード 0600、デーモンは相手の uid を確かめる

## 入れ方と配り方

### 今: Go でビルドする

- `go install ./cmd/hwnpud` (または `go build`) で実行ファイル名は `hwnpud`。Mac ではこれが Core AI の項目の唯一の鍵。
  `go run ./cmd/hwnpud` も一時フォルダから同じ名前で動き、同じキャッシュに当たる
- Mac では Go のリンカーはどの開発ビルドとも同じく `a.out` で ad-hoc に署名するので、aned のコードは共有のまま。`codesign` の手順はない
- `hwnpud install` は自分のパス (`os.Executable`、たいてい `~/go/bin/hwnpud`) を指すサービスのファイルを書き、起こす
  (`launchctl bootstrap gui/$UID`、`systemctl --user enable --now hwnpud.socket`)。`hwnpud uninstall` で戻す。
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
| `hwnpud install` や `ai` の cli の `daemon install` (Go で落とし、sha256 を確かめる) | なし | 動く |
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
   モデルのコマンドを交互に流しても障壁 (`stepBarrier` / `chunkBarrier`) が効くか

## 段取り

1. 監督役、エンジン、ソケットで起きる、使われなければ終わる、`install` / `uninstall` の hwnpud を Mac と Linux で。
   チャットのモデルだけで、`hwnpu-chat` をクライアントに (`HWNPU_DAEMON=on`)
2. モデルの管理を hwnpud に: 一覧、状態、計画、進み具合つきの準備、ライセンス、gc。コマンドの `list` / `status` と
   `hwnpu cache` をクライアントに。取得元はファイルから
3. `ai.Runtime` のリモートの側と `HWNPU_DAEMON=auto`。samples の interpreter、layers、parallax をデーモン経由にし、
   Core AI のキャッシュが `coreai-cache/<OS>/hwnpud/` だけになること
4. 画像と音声。機械全体の割り振り (プログラムの予算、降ろす、優先度)
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
