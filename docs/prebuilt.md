# 変換済みのモデルを Hugging Face で配る (計画)

作成: 2026-11-07。状態: 計画 (未実装)。

利用者の手元では、ダウンロードして読み込むだけにする。変換と校正と照合は aix1 (Ryzen AI 9 365、Linux) を GitHub Actions の self-hosted runner にして、そこで回す。design.md 16 章の当初の方針 (「配布前に一度だけ各 NPU の形式へ変換し、S3/NFS などで配る」) の置き場所を、Hugging Face に具体化したもの。

## 1. 目的と範囲

- 利用者: `hwnpu-chat -model LiquidAI/LFM3-1.7B` は、まず変換済みのリポジトリを見る。合う組があれば、その NPU のサブフォルダだけを落として読み込む。なければ今までどおり手元で変換する (後退はさせない)
- 一番効くのは AMD。今の利用者は IRON、mlir-aie、Peano、Python の環境を入れて書き出している。次が GPTQ の符号 (校正は CPU で数時間、符号は 1 モデル 1.7 GB 級) と埋め込みの表
- 手元に残る処理 (配れないもの):
  - Apple: ANE のコンパイル (e5 バンドルのキャッシュ) は機械と OS ごと。初回の起動の重さ (Gemma 4 は 100 秒級) は残る
  - Intel: NPU の blob はドライバが読み込むときにコンパイルする (ドライバの版に依る)
  - AMD: なし (xclbin、ELF、バンクの初期像をそのまま読む)

## 2. 前提になる作業 (配る前に済ませる)

| 作業 | 理由 | 参照 |
|---|---|---|
| 埋め込みの表を変換時に書き出す (T5) **済み** (chat、埋め込み、hwnpu-transcribe、hwnpu-tts、hwnpu-systemone) | 今は実行時もチェックポイントを開くので、配っても元の重み (LFM3 3.5 GB、Gemma 4 10.2 GB) を落とすことになる | model-fit.md T5、design.md 16.x |
| hwnpu-chat が GPTQ の符号を使う **済み** (`--codes`、#77) | 配る形式は、校正済みのもの (lut6 の KL が ANE で −23〜35%) にしたい | handoff-m3-gptq.md、improvements.ja.md 11.5 |
| 重みを 1 つにして実行時に組み立てる | Q=1 と Q=128 の `weight.bin`、AMD の版どうしのバンクの先頭は同じバイト。そのまま置くと 2 回落とす。配るのは 1 つにし、ローダーが clone / hardlink で組み立てる (`cache.ShareFiles` の逆向き) | design.md 16.64、16.79 |
| 書き出し形式の版を manifest に | 今は「既定が変わったら変換し直す」ために、形式を名前に入れて区別している。配るものは版で互換性を判断する | 4 節 |

## 3. 何を配るか (組み合わせを絞る)

今のプログラム名は、形式 × 文脈 × Q × last × 塊 × 段階 (`lfm3-lut6-grouped-p8192-c262144-q128-last.mlpackage`) でできていて、全部は配れない。**NPU ごとに既定の 1 組だけ**を配り、それ以外は手元で変換する。

| 対象 | 配るもの | 既定の例 | 照合 |
|---|---|---|---|
| `apple` | `.mlpackage` (分けない形。部分の数は実行時に決める (16.12)) | GPTQ lut6 か lut8 (LFM3 系は lut8)、最大の文脈、`-paged` の既定 | aix1: milcpu の KL。公開前に Mac の ANE で `hwnpu-dev check` (7 節) |
| `amd-npu2` | IRON の書き出し (ELF、xclbin、バンク、manifest) | q4 (GPTQ)、Anima の DiT は q8 | aix1 の NPU で `ironcheck` と KL |
| `amd-npu1` | `--per-step` の書き出し | 同上 | 実機がないので「未確認」として staging まで |
| `intel` | OpenVINO IR (`.xml` / `.bin`) | int8 | OpenVINO の CPU。実機の NPU で確かめるまで staging まで |

## 4. Hugging Face のリポジトリの形

- 1 モデル 1 リポジトリ (例: `hwnpu/LFM3-1.7B`)。NPU ごとのサブフォルダに分け、利用者は `allow_patterns` で自分の分だけ落とす

```
README.md                 モデルカード (元のモデル、ライセンス、検証の結果)
hwnpu.json                全体の目録: 版、元の HF リポジトリとリビジョン、組の一覧
tokenizer.json ...        トークナイザー (元のリポジトリから写す)
embed/                    埋め込みの表 (T5、どの NPU でも共通)
apple/lut8-p8192-c262144/ manifest.json、weights/ (1 つ)、各プログラム (重みなし)
amd-npu2/q4/              manifest.json、ELF、xclbin、bank
intel/int8/
```

- `hwnpu.json` と各 `manifest.json` に書くもの:
  - `format_version` (書き出し形式の版。読める hwnpu の範囲はコードの側で持つ)
  - 作った hwnpu のコミット、IRON / mlir-aie の版 (今は v1.4.4)、Peano の版
  - 元の HF リポジトリとリビジョン (コミット)
  - 各ファイルの sha256 と大きさ
  - 検証の結果: KL、最上位の一致、配置 (`all on the NPU`)、測った機械
- リビジョン: `staging` ブランチに上げ、検証が揃ったら `main` に進める (HF のブランチ)。hwnpu は既定で `main` を読み、`-revision` で固定できる
- 重複除去: HF のストレージは Xet のチャンク単位で重複を除く。ダウンロードの量にも効くかは確かめる。効かなければ 2 節の「重みを 1 つに」で対処する

## 5. ビルドの仕組み: aix1 を self-hosted runner に

GitHub のホストランナーは 14 GB のディスクと 16 GB 以下のメモリしかなく、NPU もない。チェックポイント、校正、書き出しのどれも収まらないので、aix1 で回す。

### 5.1 公開リポジトリでの安全

**shibukawa/hwnpu は PUBLIC。** self-hosted runner を公開リポジトリに付けると、フォークからの PR のワークフローが aix1 で任意のコードを動かせる。次の全部で防ぐ。

- ビルドのワークフローは `pull_request` / `pull_request_target` では**決して起動しない**。起動は `main` への push で `prebuilt/models.yaml` が変わったとき (5.4) と、owner の `workflow_dispatch` だけ
- ジョブの頭で `if: github.repository == 'shibukawa/hwnpu' && github.actor == 'shibukawa'`
- リポジトリの設定: 「Require approval for all external contributors」。runner は専用の runner group に入れ、使えるワークフローを `.github/workflows/prebuilt.yml` に限る (Selected workflows)
- 既存の CI (もし足すなら) は GitHub のホストランナーのままにし、self-hosted のラベルを使わない
- aix1 の runner は専用のユーザー `hwnpu-ci` で動かす (sudo なし、ホームの他のデータは読めない)。`~/models` の作業中のデータは開発用のユーザーのものなので、CI は自分のキャッシュ (`/srv/hwnpu-ci/cache`) を持つ

### 5.2 runner の設置

- 登録: `gh api -X POST repos/shibukawa/hwnpu/actions/runners/registration-token` で得たトークンで `./config.sh --url https://github.com/shibukawa/hwnpu --labels aix1,amd-npu2,linux-x64 --runnergroup prebuilt`
- サービス: `sudo ./svc.sh install hwnpu-ci`。**systemd のサービスには `/etc/security/limits.d` が効かない**ので、unit に次を足す (INSTALL_AMD.ja.md の memlock と render の手順に当たるもの)

```ini
[Service]
LimitMEMLOCK=infinity
SupplementaryGroups=render
```

- 道具: Go、uv と Python 3.12、mlir-aie v1.4.4 の `ironenv`、Peano、XRT、`hf` (huggingface_hub)、OpenVINO (CPU の照合用)。版は 4 節の manifest に書くので、`tools/ci/aix1-setup.sh` に固定して置く

### 5.3 更新番号のファイル (`prebuilt/models.yaml`)

配るモデルと、そのモデルの**更新番号**を持つ設定ファイル。モデルのビルドは、このファイルの番号が上がったときにだけ走る。コードを変えただけではモデルは作り直さない。

```yaml
format_version: 1            # 書き出し形式の版 (コードの側の読める範囲と合わせる)
models:
  - name: LFM3-1.7B
    source: LiquidAI/LFM3-1.7B
    revision: 0123abcd...    # 元の HF リポジトリのコミット (固定)
    build: 3                 # 更新番号。作り直したいときに 1 つ上げる
    targets:
      apple:    {format: lut8, codes: gptq}
      amd-npu2: {format: q4, codes: gptq}
      intel:    {format: int8, publish: staging}   # 実機の確認までは main に進めない
```

- 番号を上げる理由の例: 変換器の修正で出力が変わった、校正の設定を変えた、元のリビジョンを上げた
- 同じ番号で作り直すことはない。HF の側にも番号を残す (`hwnpu.json` の `build`、HF のタグ `b3`)。利用者の hwnpu は `build` を見て、手元のものより新しければ取り直す
- `format_version` を上げたときは、全モデルの `build` も上げる (どのモデルを作り直すかは、いつも番号の差分だけで決まるようにする)

### 5.4 ワークフロー (`.github/workflows/prebuilt.yml`)

- **PR ではモデルをビルドしない。** PR で走るのは GitHub のホストランナーでの軽い検査だけ: `prebuilt/models.yaml` の形、`build` が下がっていないこと、`format_version` を上げたのに `build` を上げていないモデルがないこと。self-hosted のラベルを使うジョブは PR では起動しない (5.1)
- **ビルドはデプロイの段階だけ**: `main` への push のうち、`paths: [prebuilt/models.yaml]` に当たるとき。前のコミットのファイルと比べ、`build` が上がった (または新しく足した) モデルと対象の組だけを作る
- 手で走らせる `workflow_dispatch` は、失敗したビルドのやり直し用に残す (入力は `model`。番号は変えずに同じ番号で作り直し、HF の `staging` を上書きする。`main` に進んだ番号は上書きしない)
- ジョブの流れ (1 組ごと):
  1. 元のチェックポイントを取得する (`/srv/hwnpu-ci/cache/hf`、リビジョンを固定)。Gemma など gated のモデルは、トークンのアカウントで規約に同意しておく
  2. 校正 (GPTQ の符号)。元のリビジョンと校正の設定のハッシュでキャッシュし、変わらなければ飛ばす (数時間かかるので)
  3. 変換、書き出し (`hwnpu-dev convert`、`iron_export_lm.py`)
  4. 照合: `hwnpu-dev check` (milcpu)、`ironcheck` と NPU での KL、OpenVINO の CPU。決めた上限 (例: 既存の値の 1.2 倍) を超えたら失敗にする
  5. 目録を作る (sha256、版、検証の結果)
  6. `hf upload --revision staging` (コミットの説明に `build` の番号と hwnpu のコミット)
- `concurrency: group: aix1` で、同時に 1 ジョブだけ。NPU は 1 つで、メモリも校正と書き出しが重い (`CREATE_HWCTX` の `EINVAL` (improvements.ja.md 11.3) も避けられる)
- `timeout-minutes`: 校正を含むジョブは長くする (self-hosted は最大 5 日)
- 後片付け: ジョブの作業ディレクトリは消し、チェックポイントと符号のキャッシュは残す。ディスクの残りが一定を下回ったら、古いものから消す手順を最初に走らせる
- 秘密: `HF_TOKEN` は配布先の名前空間だけに書ける fine-grained トークン。GitHub の environment `huggingface` に置き、`main` からのジョブにだけ渡す (フォークの PR には秘密が渡らないが、5.1 と合わせて二重にする)

## 6. 利用者の側 (`chat.Prepare`)

1. NPU と形式から組の名前を決める (今の `kind` と同じ規則)
2. `hwnpu.json` を読み、`format_version` が読める範囲で、組があればそのサブフォルダを落とす (`chat.Fetch` に `allow_patterns` 相当を足す)
3. sha256 を確かめ、重みを組み立てる (clone / hardlink)
4. なければ、または `-convert` を付けたときは、今までどおり元のチェックポイントから変換する
- 置き場所は今の `~/.cache/hwnpu/models/<name>/`。変換済みと手元の変換は名前で分ける (配ったものに `-hf` を付けるなど)

## 7. Mac での確認 (後で)

ANE の配置と KL は Mac でしか測れない。順番:

1. 当面: aix1 が `staging` に上げたものを M3 で手で `hwnpu-dev check` し、結果をモデルカードに書いてから `main` に進める
2. 後で: M3 を 2 台目の self-hosted runner (ラベル `macos-ane`) にし、`staging` に上がったら ANE の照合を回す。M3 は 16 GB でディスクが厳しく、e5 バンドルのキャッシュが溜まる (ジョブの終わりに `~/Library/Caches/<process>/com.apple.e5rt.e5bundlecache` を消す)

## 8. ライセンス

公開して再配布するので、モデルごとに確かめる。モデルカードの元のライセンス、表示の義務 (例: PLaMo の表示)、商用の条件、gated かどうか。gated の元モデル (Gemma) の変換物は、同じ条件で gated にすることも考える。

| モデル | ライセンス | 確認 |
|---|---|---|
| LFM3 系 | LFM Open License | 未 |
| Gemma 4 | Gemma の利用規約 | 未 |
| PLaMo 3 / PLaMo-Embedding | PLaMo コミュニティライセンス | 未 |
| Qwen3.5 | Apache 2.0 | 未 |
| Anima | | 未 |

## 9. 進める順

1. T5 (埋め込みの表の書き出し) と、hwnpu-chat で GPTQ の符号を使うこと
2. manifest の `format_version`、重みの組み立て、`chat.Prepare` で変換済みを読む形 (ローカルのディレクトリで試す)
3. aix1 の runner の設置 (5.1、5.2)、`prebuilt/models.yaml` (5.3) と `prebuilt.yml` (5.4。最初はアップロードせずにビルドと照合だけ)
4. LFM3-1.7B の `amd-npu2` と `apple` を `staging` に上げ、M3 で確かめて `main` へ
5. ほかのモデル。Intel は実機で確かめてから
