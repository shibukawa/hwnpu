# npuinfo

[English](README.md)

OS が NPU について教えてくれることを読む。デバイス、どのプロセスが使っているか、どれだけ忙しいか、ドライバとファームウェアが報告すること。macOS のアクティビティモニタや Linux のいつもの道具には出てこない (Windows のタスクマネージャーには一部出る)。

今は、OS とドライバが返すものをそのまま OS ごとの構造体 (`Info.Linux`、`Info.Windows`、`Info.Darwin`) に入れるだけ。共通の API は、各 OS で何が取れるかわかってから考える。import するのは標準ライブラリだけで、hwnpu の他のパッケージも使わない。独立したモジュールに切り出せて、他のモジュール (gopsutil など) が依存を増やさずに使えるようにするため。型はどの OS でもビルドされるので、別の機械が書いた JSON を見る側で読める。

```bash
go run ./npuinfo/cmd/npuinfo              # 1 回、字下げした JSON
go run ./npuinfo/cmd/npuinfo -n 10 -i 1s  # 1 秒おきに 10 回、1 行 1 JSON
go run ./npuinfo/cmd/npuinfo -wake        # 寝ている NPU にも問い合わせる (Linux。起こす)
```

```go
info, err := npuinfo.GetWithContext(ctx, npuinfo.Options{})
```

読むことで読む相手を変えないこと。amdxdna の問い合わせは寝ている NPU を起こし、ドライバの自動サスペンドの待ち (5 秒) の間起こしたままにする。毎秒読む監視ツールだと NPU が寝られなくなる。そこで問い合わせは NPU がもう動いているとき (`power/runtime_status` が active) だけにし、`Options.Wake` で変えられる。`/dev/accel/accel0` を開くことと sysfs を読むことでは起きない (寝ている NPU で確かめた)。

macOS の IOReport と `proc_*` は C の関数。cgo なしで、golang.org/x/sys/unix と同じやり方で呼ぶ。libSystem から import するのは `dlopen` と `dlsym` だけ (`cgo_import_dynamic` とアセンブリのトランポリン) で、残りは実行時に探す。非公開のライブラリが無い OS でも、バイナリが起動しなくなるのではなく `Info.Errors` に出るだけ。IOReport を読んでも ANE は起きない (休みの間に 40 回読んで電源の状態は 0 のまま)。`Options.SampleInterval` (500 ms) の間待つ。負にすると読まない。

## OS ごとに取れるもの

| | Linux (amdxdna、Ryzen AI 9 365 で確認) | Linux (intel_vpu、未確認) | Windows (未確認) | macOS (M1、macOS 27.2.1 で確認) |
|---|---|---|---|---|
| デバイス | sysfs: PCI の ID、`fw_version`、`vbnv` (RyzenAI-npu4)、リンク、`power_state`。ioctl: AIE の版、配列の形 (8 列 × 6 行)、ファームウェア、クロック (MP-NPU 1267 MHz、H 1800 MHz)、電力モード、TOPS とタスクの上限 | sysfs: `npu_busy_time_us`、`npu_current_frequency_mhz`、`npu_max_frequency_mhz`、`sched_mode` (sysfs の属性をまとめて読む中に入る) | PnP: ComputeAccelerator のデバイス、ドライバの版と日付。D3DKMT: 計算専用 (MCDM) を含むアダプター、種類、PCI の ID、ノード (エンジン) | I/O Registry: ANE のクラス (`H11ANEIn` など) とその属性全部。型付きで: アーキテクチャ (`h13g`)、コア数 (16)、版、`ane,t8020`、ファームウェアを読み込んだか、電源の状態 (休みで 0、動かしている間 1)。sysctl: 機種、CPU |
| 使っているプロセス | ioctl `GET_ARRAY(HW_CONTEXT_ALL)`: ハードウェアの文脈ごとに PID、列、送ったコマンドと終わったコマンドの数、状態 (root 不要)。`/proc/*/fd`: `/dev/accel/*` を開いているプロセス | `/proc/*/fd` と fdinfo (ドライバが持てば `drm-*`) | `pid_N` のインスタンスを持つパフォーマンスカウンター (`GPU Engine`、`GPU Process Memory`、NPU 用のセットがあるビルドではそれも) | ANE の下のユーザークライアント (`IOUserClientCreator`: "pid N, 名前")。プロセスごと (`proc_pid_rusage`): ANE のメモリ。`aned` 経由で使うプロセス (`mediaanalysisd`) もこれで見つかる。パス、クライアントの数 |
| 忙しさ | gpu_metrics の `average_ipu_activity` (%)、コマンド数の増え方 | `npu_busy_time_us` の増え方 | エンジンとプロセスごとの `Utilization Percentage`、`Running Time` | IOReport (root 不要): ANE の割り込み (プログラムの呼び出し 1 回で約 3)、電源の状態 |
| 電力 | gpu_metrics の `average_ipu_power` (mW)、ソケット全体。amdxdna のセンサーはこの BIOS では ENODEV | — | ドライバが入れていれば D3DKMT のアダプターの性能データ | IOReport (root 不要): `PMP`/`Energy Counters`/`ANE` (mJ) から W (0.23 GiB の lut4 のプログラムを回して 2.0 W)。root なら powermetrics の `ane_power` |
| メモリ | `/proc/PID/maps` の `/dev/accel/accel0` の行: RSS に出ない NPU のバッファ (LFM3 q4 で 1635 MiB、前に測った 1.64 GiB と合う)。gpu_metrics の NPU と DRAM の読み書き (MB/s) | fdinfo の `drm-total-*`、`drm-resident-*` | `GPU Process Memory`、`GPU Adapter Memory`、セグメントの大きさ | プロセスごとの `ri_neural_footprint` (今と最大。0.23 GiB のプログラムで 234 MiB。アクティビティモニタのメモリには出ない)。IOReport: ANE の DRAM の読み書き (`AMC Stats`、そのプログラムで 44 GB/s = 重み × 毎秒 175 回) |
| NPU の中 | 列ごとの AIE の状態レジスタ (生のまま、504 バイト × 8)、ファームウェアのテレメトリ (生のまま)、最後の非同期エラー | — | — | — |
| キャッシュ | — | — | — | `~/Library/Caches` と `DARWIN_USER_CACHE_DIR` の下の `com.apple.e5rt.e5bundlecache`: 大きさ、ファイル数、OS のビルド |

取れなかったもの: Linux 7.0 の amdxdna は fdinfo に `drm-*` を足さない。文脈の `gops`、`fps`、`heap_usage` はこのファームウェア (1.1.2.64) では 0 のまま。Windows のタスクマネージャーは 26100.8524 (24H2)、26200.8524 (25H2) からプロセスごとの NPU の列を持つが、どのカウンターセットを読んでいるかは書かれていない。そこで名前に GPU、NPU、Neural を含むセットを全部読む。向こうで `npuinfo` を動かせば、どれがあるかわかる。

## 次

- 生のままの部分を読み解く: タイルごとの AIE の状態 (コアの状態、PC、ロック、DMA)、テレメトリ
- Windows (Intel と AMD) で動かして、「未確認」の列を埋める。M2 以降で IOReport のチャンネル名を確かめる (`Energy Model`/`ANE…` も読むようにはしてある)
- OS をまたぐ共通の API、そのあとブラウザで見る道具
