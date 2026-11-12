<#
.SYNOPSIS
  Runs hwnpu's checks on an Intel Core Ultra NPU (Windows 11) in one go and packs the logs into a zip.

.DESCRIPTION
  Each test runs one of hwnpu's commands with a time limit, keeps its standard output and error,
  its time and its memory, and goes on to the next one when it fails. At the end everything is in
  hwnpu-intel-<machine>-<time>.zip under %USERPROFILE%\hwnpu-intel-logs: send that file back.

  Every command runs as "go run ./cmd/NAME [SUBCOMMAND] ..." in the clone this script is in (-Repo to name
  another), so Go has to be on PATH. The commands are built once first (go build), so that the
  times of the tests are not the compiler's. The commit is written to system.txt.

  Levels (-Level):
    quick     system information, hwnpu-dev info, hwnpu-dev report, a small LFM3-1.7B chat (about 30 min)
    standard  + LFM3-1.7B with its defaults, the same again from the compile cache, hwnpu-chat agent,
              hwnpu-dev formats, PLaMo 3 Translate 0.8B, PLaMo-Embedding-1B, Gemma 4 E2B, what the NPU holds at once
              (hwnpu-dev probe resident), Anima 512 x 512 in fp16 and int8
              (default; a few hours, about 25 GB of downloads the first time)
    full      + report with power policies low and performance, LFM3-VL-2B with an image,
              LFM3-Audio-2B with a clip, Qwen3.5-0.8B, Qwen3-0.6B, PLaMo 3 Translate 2B, Anima 1024 x 1024,
              resident in int8, hwnpu-dev probe, and the models first converted to OpenVINO IR in
              2026-11 (checked only on OpenVINO's CPU before): OCR, RTMDet, RTMO, YuNet, DA3, MoGe-2,
              SwiftF0, Basic Pitch, Whisper, LFM3-Audio TTS, Magenta RealTime 2, Sana 600M and
              MOSS-SoundEffect (about 25 GB more of downloads; -Skip sana,sfx leaves out 17 GB of them)

  A test that finished (exit code 0) is skipped when the script runs again with the same -Out,
  so it can be stopped with Ctrl-C and run again. -Rerun runs them all again.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\hwnpu-intel-check.ps1

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\hwnpu-intel-check.ps1 -Level quick

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\hwnpu-intel-check.ps1 -Only chat-small,agent
#>
[CmdletBinding()]
param(
  [ValidateSet("quick", "standard", "full")]
  [string]$Level = "standard",
  # tests to run, comma separated names (-List shows them); overrides -Level
  [string[]]$Only = @(),
  # tests to leave out
  [string[]]$Skip = @(),
  # print the tests and exit
  [switch]$List,
  # the clone of hwnpu to run (default: the one this script is in)
  [string]$Repo = "",
  # where the models are kept (hwnpu's -dir; default: %LOCALAPPDATA%\hwnpu\models)
  [string]$ModelsDir = "",
  # directory of the logs (default: a new %USERPROFILE%\hwnpu-intel-logs\hwnpu-intel-<machine>-<time>)
  [string]$Out = "",
  # multiply every time limit by this
  [double]$TimeoutScale = 1.0,
  # run the tests that already passed in -Out again
  [switch]$Rerun,
  # do not make the zip at the end
  [switch]$NoZip
)

$ErrorActionPreference = "Continue"
# native commands' output (git's commit subjects, python's) is read as UTF-8, not the ANSI code page
try { [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false } catch { }
Set-StrictMode -Version 2.0

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($Repo -eq "") { $Repo = (Resolve-Path (Join-Path $Here "..\..")).Path }
$Go = Get-Command go -ErrorAction SilentlyContinue
if (-not $Go) { Write-Host "go is not on PATH"; exit 2 }
if (-not (Test-Path (Join-Path $Repo "go.mod"))) { Write-Host "$Repo is not a clone of hwnpu (-Repo)"; exit 2 }
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
if ($Out -eq "") { $Out = Join-Path $env:USERPROFILE ("hwnpu-intel-logs\hwnpu-intel-" + $env:COMPUTERNAME + "-" + $Stamp) }
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$Out = (Resolve-Path $Out).Path
$Inputs = Join-Path $Out "inputs"
New-Item -ItemType Directory -Force -Path $Inputs | Out-Null
$Utf8 = New-Object System.Text.UTF8Encoding $false
$RunLog = Join-Path $Out "run.log"
$AnsiRe = [string][char]27 + "\[[0-9;]*[A-Za-z]"

function Log([string]$msg) {
  $line = (Get-Date -Format "HH:mm:ss") + " " + $msg
  Write-Host $line
  [System.IO.File]::AppendAllText($RunLog, $line + "`r`n", $Utf8)
}

function Write-Utf8([string]$path, [string]$text) {
  [System.IO.File]::WriteAllText($path, $text, $Utf8)
}

# Quote-Arg quotes one argument the way Go's and the C runtime's command line parsers read it.
function Quote-Arg([string]$a) {
  if ($a -ne "" -and $a -notmatch '[\s"]') { return $a }
  $sb = New-Object System.Text.StringBuilder
  [void]$sb.Append('"')
  $bs = 0
  foreach ($ch in $a.ToCharArray()) {
    if ($ch -eq '\') { $bs++; continue }
    if ($ch -eq '"') { [void]$sb.Append('\' * (2 * $bs + 1)); [void]$sb.Append('"'); $bs = 0; continue }
    if ($bs -gt 0) { [void]$sb.Append('\' * $bs); $bs = 0 }
    [void]$sb.Append($ch)
  }
  if ($bs -gt 0) { [void]$sb.Append('\' * (2 * $bs)) }
  [void]$sb.Append('"')
  return $sb.ToString()
}

# Tail returns the last non-empty line of a file another process is writing.
function Tail([string]$path) {
  try {
    $fs = [System.IO.File]::Open($path, 'Open', 'Read', 'ReadWrite')
    try {
      $n = [Math]::Min($fs.Length, 4096)
      if ($n -le 0) { return "" }
      [void]$fs.Seek(-$n, 'End')
      $buf = New-Object byte[] $n
      [void]$fs.Read($buf, 0, $n)
      $text = $Utf8.GetString($buf) -replace $AnsiRe, ""
      $lines = $text -split "[`r`n]+" | Where-Object { $_.Trim() -ne "" }
      if ($lines) { return @($lines)[-1] }
    } finally { $fs.Close() }
  } catch { }
  return ""
}

function FreeMB {
  try { return [int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024) } catch { return -1 }
}

function DirMB([string]$path) {
  if (-not (Test-Path $path)) { return 0 }
  $s = (Get-ChildItem -LiteralPath $path -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
  if ($s -eq $null) { return 0 }
  return [int]($s / 1MB)
}

# ---------------------------------------------------------------------------------------------
# executables

$Cmds = @("hwnpu-dev", "hwnpu-chat", "hwnpu-image", "hwnpu-audio")

# Warm-Build builds every command once, so that go run only links them; a command that does not
# build fails its tests with the compiler's errors in their logs.
function Warm-Build {
  Push-Location $Repo
  try {
    Log ("go build in $Repo with " + (& go version))
    $pkgs = @($Cmds | ForEach-Object { "./cmd/" + $_ })
    $o = @(& go build @pkgs 2>&1)
    if ($LASTEXITCODE -ne 0) { Log "go build failed:"; foreach ($l in $o) { Log ("  " + $l) } } else { Log "go build ok" }
  } finally { Pop-Location }
}

# ChildMem returns the peak working set and private bytes of the command (NAME.exe) among the
# processes under id: go run starts it from go.exe, which may itself be started by a shim (scoop).
# When no NAME.exe is found yet, the largest of the descendants.
function ChildMem([int]$id, [string]$name) {
  $ws = 0; $priv = 0
  try {
    $all = @(Get-CimInstance Win32_Process -Property ProcessId, ParentProcessId, Name, PeakWorkingSetSize, PrivatePageCount -ErrorAction Stop)
    $kids = @{}
    foreach ($c in $all) { $k = [int]$c.ParentProcessId; if (-not $kids.ContainsKey($k)) { $kids[$k] = @() }; $kids[$k] += $c }
    $under = @(); $todo = @($id)
    while ($todo.Count -gt 0) {
      $next = @()
      foreach ($q in $todo) { if ($kids.ContainsKey($q)) { foreach ($c in $kids[$q]) { if ([int]$c.ProcessId -ne $q) { $under += $c; $next += [int]$c.ProcessId } } } }
      $todo = $next
      if ($under.Count -gt 200) { break }
    }
    $mine = @($under | Where-Object { $_.Name -ieq ($name + ".exe") })
    if ($mine.Count -eq 0) { $mine = $under }
    foreach ($c in $mine) {
      $w = [int64]$c.PeakWorkingSetSize * 1024
      if ($w -gt $ws) { $ws = $w }
      if ([int64]$c.PrivatePageCount -gt $priv) { $priv = [int64]$c.PrivatePageCount }
    }
  } catch { }
  return @($ws, $priv)
}

# ---------------------------------------------------------------------------------------------
# inputs made here: prompts, an image, a clip, a working directory for the agent

function Make-Image([string]$path) {
  try {
    Add-Type -AssemblyName System.Drawing
    $bmp = New-Object System.Drawing.Bitmap 512, 512
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = 'AntiAlias'
    $g.Clear([System.Drawing.Color]::White)
    $g.FillEllipse([System.Drawing.Brushes]::Red, 40, 60, 180, 180)
    $g.FillRectangle([System.Drawing.Brushes]::Blue, 290, 60, 180, 180)
    $font = New-Object System.Drawing.Font "Arial", 56, ([System.Drawing.FontStyle]::Bold)
    $g.DrawString("NPU 42", $font, [System.Drawing.Brushes]::Black, 110, 330)
    $g.Dispose()
    $bmp.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
    $bmp.Dispose()
    return $true
  } catch { Log ("could not draw the test image: " + $_); return $false }
}

$Script:ClipText = ""
function Make-Clip([string]$path) {
  try {
    Add-Type -AssemblyName System.Speech
    $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $voices = @($s.GetInstalledVoices() | Where-Object { $_.Enabled } | ForEach-Object { $_.VoiceInfo })
    $en = @($voices | Where-Object { $_.Culture.Name -like "en*" })
    if ($en.Count -gt 0) {
      $s.SelectVoice($en[0].Name)
      $Script:ClipText = "The meeting moves to Thursday at three in the afternoon. Please bring the budget report."
    } else {
      if ($voices.Count -gt 0) { $s.SelectVoice($voices[0].Name) }
      $Script:ClipText = "会議は木曜日の午後三時に変更になりました。予算の報告書を持ってきてください。"
    }
    $fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo 16000, ([System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen), ([System.Speech.AudioFormat.AudioChannel]::Mono)
    $s.SetOutputToWaveFile($path, $fmt)
    $s.Speak($Script:ClipText)
    $s.SetOutputToNull()
    $s.Dispose()
    Write-Utf8 (Join-Path $Inputs "clip.txt") $Script:ClipText
    return $true
  } catch { Log ("could not synthesize the test clip: " + $_); return $false }
}

$ImagePng = Join-Path $Inputs "shapes.png"
$ClipWav = Join-Path $Inputs "clip.wav"
$AgentDir = Join-Path $Out "agent-work"

# Questions with answers that can be checked by eye in the logs.
$ChatPrompts = @(
  "日本の首都はどこですか？一言で答えてください。",
  "What is 17 times 23? Answer with the number only.",
  "Write a Python function that returns the n-th Fibonacci number. Code only."
) -join "`n"

# ---------------------------------------------------------------------------------------------
# tests

$Common = @()
if ($ModelsDir -ne "") { $Common = @("--dir", $ModelsDir) }

function T($h) { return New-Object PSObject -Property $h }

$Lfm = @("--backend", "intel", "-m", "LFM3-1.7B")
$Short = @("--temp", "0", "-n", "256", "--no-think")
$Translate = "今日は雨が降っているので、駅まで車で送ってもらえますか。`nThe quarterly report shows a 12 percent increase in revenue."
$Anima = "masterpiece, best quality, score_7, safe, 1girl, cherry blossoms"

$Tests = @(
  (T @{ Name = "hwnpuinfo"; Level = 0; Exe = "hwnpu-dev"; Sub = "info"; Args = @(); Min = 5
        Note = "devices, driver, Level Zero version, graph extension" }),
  (T @{ Name = "hwnpuinfo-trace"; Level = 0; Exe = "hwnpu-dev"; Sub = "info"; Args = @(); Min = 5; Env = @{ ZE_ENABLE_LOADER_DEBUG_TRACE = "1" }
        Note = "the same with the Level Zero loader's trace" }),
  (T @{ Name = "report"; Level = 0; Exe = "hwnpu-dev"; Sub = "report"; Min = 20
        Args = @("-o", "{OUT}\report.md", "-json", "{OUT}\report.json")
        Note = "two OpenVINO IR models compiled strictly, checked against fp32, timed" }),
  (T @{ Name = "chat-small"; Level = 0; Exe = "hwnpu-chat"; Sub = "chat"; Min = 90; Stdin = $ChatPrompts
        Args = $Lfm + @("--paged=-1", "--tier", "none", "--context", "4096") + $Short
        Note = "LFM3-1.7B, int8, one program, 4096 tokens (INSTALL_INTEL 6, check 1 and 2)" }),

  (T @{ Name = "chat-default"; Level = 1; Exe = "hwnpu-chat"; Sub = "chat"; Min = 120; Stdin = $ChatPrompts
        Args = $Lfm + $Short
        Note = "LFM3-1.7B with its defaults: split at attention (--paged), 262144 tokens (check 3)" }),
  (T @{ Name = "chat-default-cached"; Level = 1; Exe = "hwnpu-chat"; Sub = "chat"; Min = 60; Stdin = $ChatPrompts
        Args = $Lfm + $Short
        Note = "the same again: loading from the compile cache" }),
  (T @{ Name = "chat-think"; Level = 1; Exe = "hwnpu-chat"; Sub = "chat"; Min = 60
        Args = $Lfm + @("--temp", "0", "-n", "1024", "A farmer has 17 sheep. All but 9 run away. How many are left?")
        Note = "LFM3-1.7B thinking before it answers, the prompt given on the command line" }),
  (T @{ Name = "agent"; Level = 1; Exe = "hwnpu-chat"; Sub = "agent"; Min = 60
        Args = $Lfm + @("-C", "{AGENT}", "--tools", "read,write", "--yes", "--steps", "8", "--temp", "0",
                 "Read notes.txt and write the secret number it contains, and nothing else, to answer.txt.")
        Note = "hwnpu-chat agent calling its read and write tools (check 4); answer.txt should hold 7351" }),
  (T @{ Name = "formats"; Level = 1; Exe = "hwnpu-dev"; Sub = "formats"; Min = 180
        Args = @("-backend", "intel", "-json", "{OUT}\formats.json")
        Note = "int8, int4-g128, int4 against fp16: time per call, next token, perplexity (check 5)" }),
  (T @{ Name = "translate-0.8b"; Level = 1; Exe = "hwnpu-chat"; Sub = "chat"; Min = 90; Stdin = $Translate
        Args = @("--backend", "intel", "-m", "plamo-3-translate-0.8b", "--temp", "0")
        Note = "PLaMo 3 Translate 0.8B, int8: ja -> en and en -> ja" }),
  (T @{ Name = "embedding"; Level = 1; Exe = "hwnpu-chat"; Sub = "search"; Min = 60
        Stdin = "富士山は日本で最も高い山で、標高は3776メートルです。`nPythonはプログラミング言語の一つで、読みやすさを重視しています。`n寿司は酢飯と魚介類を組み合わせた日本料理です。`n?日本一高い山は？`n?コードを書くための言語`n?魚を使った料理"
        Args = @("--backend", "intel", "-m", "plamo-embedding-1b")
        Note = "PLaMo-Embedding-1B, int8: each ?query should find its own document first" }),
  (T @{ Name = "gemma-e2b"; Level = 1; Exe = "hwnpu-chat"; Sub = "chat"; Min = 150; Stdin = $ChatPrompts
        Args = @("--backend", "intel", "-m", "gemma-4-E2B-it") + $Short
        Note = "Gemma 4 E2B with its defaults" }),
  (T @{ Name = "resident"; Level = 1; Exe = "hwnpu-dev"; Sub = "probe"; Min = 60
        Args = @("resident", "-resident-max-gb", "8", "-o", "{OUT}\resident.md")
        Note = "how much the NPU holds at once: 0.25 GiB fp16 programs kept open until a load fails (Anima ran out at about 4.6 GB), whether closing one gives its memory back, and loading 8 GiB in turn" }),
  (T @{ Name = "resident-load-only"; Level = 1; Exe = "hwnpu-dev"; Sub = "probe"; Min = 45
        Args = @("resident", "-resident-run=false", "-resident-hold", "open", "-resident-max-gb", "8", "-o", "{OUT}\resident-load-only.md")
        Note = "the same held open without running any: tells a crash (0xc0000005) in running from one in loading" }),
  (T @{ Name = "image-512"; Level = 1; Exe = "hwnpu-image"; Sub = "generate"; Min = 180
        Args = @("--backend", "intel", "-m", "anima-turbo", "--size", "512", "--seed", "1", "-o", "{OUT}\anima-512.png", $Anima)
        Note = "Anima-Turbo 512 x 512, 8 steps: downloads, converts to OpenVINO IR (fp16), draws" }),
  (T @{ Name = "image-512-int8"; Level = 1; Exe = "hwnpu-image"; Sub = "generate"; Min = 120
        Args = @("--backend", "intel", "-m", "anima-turbo", "--size", "512", "--seed", "1", "--dit-format", "int8", "--text-format", "int8", "-o", "{OUT}\anima-512-int8.png", $Anima)
        Note = "the same in int8 (half the memory of fp16): compare the image and the step's time" }),

  (T @{ Name = "report-low"; Level = 2; Exe = "hwnpu-dev"; Sub = "report"; Min = 20
        Args = @("-power", "low", "-o", "{OUT}\report-low.md", "-json", "{OUT}\report-low.json")
        Note = "report with the power policy low" }),
  (T @{ Name = "report-performance"; Level = 2; Exe = "hwnpu-dev"; Sub = "report"; Min = 20
        Args = @("-power", "performance", "-o", "{OUT}\report-performance.md", "-json", "{OUT}\report-performance.json")
        Note = "report with the power policy performance" }),
  (T @{ Name = "chat-int4-g128"; Level = 2; Exe = "hwnpu-chat"; Sub = "chat"; Min = 90; Stdin = $ChatPrompts
        Args = $Lfm + @("--format", "int4-g128") + $Short
        Note = "LFM3-1.7B in int4-g128 (half the weights of int8)" }),
  (T @{ Name = "vision"; Level = 2; Exe = "hwnpu-chat"; Sub = "chat"; Min = 120; Needs = "image"
        Args = @("--backend", "intel", "-m", "LFM3-VL-2B", "--image", "{INPUTS}\shapes.png") + $Short + @("What shapes and colors are in this image, and what text is written in it?")
        Note = "LFM3-VL-2B: a red circle, a blue square and the text NPU 42" }),
  (T @{ Name = "audio"; Level = 2; Exe = "hwnpu-chat"; Sub = "chat"; Min = 120; Needs = "clip"
        Args = @("--backend", "intel", "-m", "LFM3-Audio-2B", "--audio", "{INPUTS}\clip.wav") + $Short + @("Write down what is said in this clip.")
        Note = "LFM3-Audio-2B: the clip's text is in inputs\clip.txt" }),
  (T @{ Name = "qwen35-0.8b"; Level = 2; Exe = "hwnpu-chat"; Sub = "chat"; Min = 90; Stdin = $ChatPrompts
        Args = @("--backend", "intel", "-m", "Qwen3.5-0.8B", "--temp", "0", "-n", "256")
        Note = "Qwen3.5-0.8B (gated delta layers; checked on OpenVINO's CPU only)" }),
  (T @{ Name = "translate-2b"; Level = 2; Exe = "hwnpu-chat"; Sub = "chat"; Min = 120; Stdin = $Translate
        Args = @("--backend", "intel", "-m", "plamo-3-translate-2b", "--temp", "0")
        Note = "PLaMo 3 Translate 2B, int8" }),
  (T @{ Name = "image-1024"; Level = 2; Exe = "hwnpu-image"; Sub = "generate"; Min = 240
        Args = @("--backend", "intel", "-m", "anima-turbo", "--size", "1024", "--seed", "1", "-o", "{OUT}\anima-1024.png", $Anima)
        Note = "Anima-Turbo 1024 x 1024" }),
  (T @{ Name = "resident-int8"; Level = 2; Exe = "hwnpu-dev"; Sub = "probe"; Min = 60
        Args = @("resident", "-resident-formats", "int8", "-resident-hold", "open", "-resident-max-gb", "8", "-o", "{OUT}\resident-int8.md")
        Note = "the same in int8: does the limit count the weights' bytes (twice as many programs) or the programs" }),
  (T @{ Name = "npuprobe"; Level = 2; Exe = "hwnpu-dev"; Sub = "probe"; Min = 120; Args = @("-o", "{OUT}\npuprobe.md")
        Note = "speed of generated LLM shaped models, and the largest model the compiler takes (PROBE.md)" }),
  # converted to OpenVINO IR since 2026-11-12, checked on OpenVINO's CPU only (HWNPU_BACKEND=openvino)
  (T @{ Name = "qwen3-0.6b"; Level = 2; Exe = "hwnpu-chat"; Sub = "chat"; Min = 90; Stdin = $ChatPrompts
        Args = @("--backend", "intel", "-m", "Qwen3-0.6B") + $Short
        Note = "Qwen3-0.6B, int8, split at its attention" }),
  (T @{ Name = "ocr"; Level = 2; Exe = "hwnpu-image"; Sub = "ocr"; Min = 30; Needs = "image"
        Args = @("--device", "npu", "{INPUTS}\shapes.png")
        Note = "PP-OCRv6 small on the NPU (max_pool, clip lowered): the line NPU 42" }),
  (T @{ Name = "detect"; Level = 2; Exe = "hwnpu-image"; Sub = "detect"; Min = 30; Needs = "image"
        Args = @("--device", "npu", "-m", "RTMDet-m", "{INPUTS}\shapes.png")
        Note = "RTMDet-m on the NPU: converts and runs (the test image holds no COCO objects)" }),
  (T @{ Name = "pose"; Level = 2; Exe = "hwnpu-image"; Sub = "pose"; Min = 30; Needs = "image"
        Args = @("--device", "npu", "{INPUTS}\shapes.png")
        Note = "RTMO-s on the NPU: converts and runs (no people in the test image)" }),
  (T @{ Name = "face"; Level = 2; Exe = "hwnpu-image"; Sub = "face-detect"; Min = 30; Needs = "image"; NoDir = $true
        Args = @("--device", "npu", "{INPUTS}\shapes.png")
        Note = "YuNet on the NPU: converts and runs (no faces in the test image)" }),
  (T @{ Name = "depth-da3"; Level = 2; Exe = "hwnpu-image"; Sub = "depth"; Min = 45; Needs = "image"; NoDir = $true
        Args = @("-m", "DA3-SMALL", "--raw", "-o", "{OUT}\depth-da3", "{INPUTS}\shapes.png")
        Note = "Depth Anything 3 small (pixel_shuffle lowered): depth-da3\shapes.depth.png" }),
  (T @{ Name = "depth-moge"; Level = 2; Exe = "hwnpu-image"; Sub = "depth"; Min = 45; Needs = "image"; NoDir = $true
        Args = @("-m", "moge-2-vits", "-o", "{OUT}\depth-moge", "{INPUTS}\shapes.png")
        Note = "MoGe-2 ViT-S: depth-moge\shapes.depth.png" }),
  (T @{ Name = "pitch"; Level = 2; Exe = "hwnpu-audio"; Sub = "pitch"; Min = 20; Needs = "clip"
        Args = @("--device", "npu", "{INPUTS}\clip.wav")
        Note = "SwiftF0 on the NPU: the pitch of the clip every 16 ms" }),
  (T @{ Name = "notes"; Level = 2; Exe = "hwnpu-audio"; Sub = "notes"; Min = 20; Needs = "clip"
        Args = @("--device", "npu", "{INPUTS}\clip.wav")
        Note = "Basic Pitch on the NPU: the clip's notes" }),
  (T @{ Name = "transcribe"; Level = 2; Exe = "hwnpu-audio"; Sub = "transcribe"; Min = 60; Needs = "clip"
        Args = @("{INPUTS}\clip.wav")
        Note = "Whisper large-v3-turbo, int8 (1.6 GB download): the text of inputs\clip.txt" }),
  (T @{ Name = "tts"; Level = 2; Exe = "hwnpu-audio"; Sub = "tts"; Min = 60
        Args = @("-o", "{OUT}\tts.wav", "--seed", "1", "The meeting moves to Thursday at three in the afternoon.")
        Note = "LFM3-Audio-2B speaking (int8, shares the audio test's download): listen to tts.wav" }),
  (T @{ Name = "music"; Level = 2; Exe = "hwnpu-audio"; Sub = "generate"; Min = 60
        Args = @("--duration", "5s", "--seed", "1", "-o", "{OUT}\music.wav", "disco funk")
        Note = "Magenta RealTime 2 small (conv_transpose, elu lowered): 5 s of music.wav" }),
  (T @{ Name = "sana"; Level = 2; Exe = "hwnpu-image"; Sub = "generate"; Min = 240
        Args = @("--backend", "intel", "-m", "Sana-600M", "--size", "512", "--seed", "1", "-o", "{OUT}\sana-512.png", "a red apple on a wooden table, photo")
        Note = "Sana 600M 512 x 512 (about 6 GB download): Gemma 2 and the DiT in int8" }),
  (T @{ Name = "sfx"; Level = 2; Exe = "hwnpu-audio"; Sub = "generate"; Min = 240
        Args = @("-m", "sfx", "--duration", "5s", "--steps", "20", "--seed", "1", "-o", "{OUT}\sfx.wav", "A dog barking loudly in a park.")
        Note = "MOSS-SoundEffect v2.0 (11.5 GB download): the text encoder and the DiT in int8, 20 steps" })
)

$LevelNum = @{ quick = 0; standard = 1; full = 2 }[$Level]

if ($List) {
  foreach ($t in $Tests) { "{0,-22} {1,-9} {2}" -f $t.Name, @("quick", "standard", "full")[$t.Level], $t.Note }
  exit 0
}

$Only = @($Only | ForEach-Object { $_ -split "," } | Where-Object { $_ -ne "" })
$Skip = @($Skip | ForEach-Object { $_ -split "," } | Where-Object { $_ -ne "" })
foreach ($n in ($Only + $Skip)) {
  if (-not ($Tests | Where-Object { $_.Name -eq $n })) { Write-Host "unknown test $n (-List shows them)"; exit 2 }
}
$Selected = @($Tests | Where-Object {
  if ($Only.Count -gt 0) { $Only -contains $_.Name } else { $_.Level -le $LevelNum }
} | Where-Object { $Skip -notcontains $_.Name })

# ---------------------------------------------------------------------------------------------
# system information

function Collect-System {
  $f = Join-Path $Out "system.txt"
  $sb = New-Object System.Text.StringBuilder
  function Sec([string]$title, $value) {
    [void]$sb.AppendLine("## " + $title)
    try {
      $text = if ($value -is [scriptblock]) { (& $value 2>&1 | Out-String -Width 200) } else { ($value | Out-String -Width 200) }
    } catch { $text = "error: " + $_ }
    [void]$sb.AppendLine($text.TrimEnd())
    [void]$sb.AppendLine("")
  }
  Sec "script" ("level " + $Level + ", tests " + (($Selected | ForEach-Object { $_.Name }) -join ",") + "`nPowerShell " + $PSVersionTable.PSVersion + ", started " + (Get-Date -Format o))
  Sec "computer" { Get-CimInstance Win32_ComputerSystem | Format-List Manufacturer, Model, SystemFamily, TotalPhysicalMemory }
  Sec "processor" { Get-CimInstance Win32_Processor | Format-List Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed }
  Sec "os" { Get-CimInstance Win32_OperatingSystem | Format-List Caption, Version, BuildNumber, OSArchitecture, FreePhysicalMemory }
  Sec "os build" { (Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion" | Format-List DisplayVersion, CurrentBuild, UBR) }
  Sec "bios" { Get-CimInstance Win32_BIOS | Format-List Manufacturer, SMBIOSBIOSVersion, ReleaseDate }
  Sec "memory modules" { Get-CimInstance Win32_PhysicalMemory | Format-Table Capacity, Speed, ConfiguredClockSpeed, PartNumber -AutoSize }
  Sec "npu devices" { Get-PnpDevice -PresentOnly -ErrorAction SilentlyContinue | Where-Object { $_.FriendlyName -match "\bNPU\b|AI Boost|\bVPU\b" -or $_.Class -eq "ComputeAccelerator" } | Format-List FriendlyName, Class, Status, InstanceId }
  Sec "npu drivers" { Get-CimInstance Win32_PnPSignedDriver | Where-Object { $_.DeviceName -match "\bNPU\b|AI Boost|\bVPU\b" -or $_.DeviceClass -eq "COMPUTEACCELERATOR" } | Format-List DeviceName, DriverVersion, DriverDate, DriverProviderName, InfName, HardWareID }
  Sec "graphics drivers" { Get-CimInstance Win32_VideoController | Format-List Name, DriverVersion, DriverDate }
  Sec "ze_loader.dll on PATH" { where.exe ze_loader.dll }
  Sec "ze_loader.dll and the NPU driver's DLLs" {
    $dirs = @("$env:SystemRoot\System32", "$env:SystemRoot\System32\DriverStore\FileRepository")
    Get-ChildItem $dirs -Recurse -Include ze_loader.dll, ze_intel_npu*.dll, npu_driver_compiler*.dll, ze_tracing_layer.dll, ze_validation_layer.dll -ErrorAction SilentlyContinue |
      ForEach-Object { "{0}  {1}  {2}" -f $_.FullName, $_.VersionInfo.FileVersion, $_.LastWriteTime }
  }
  Sec "HWNPU and ZE environment" { Get-ChildItem env: | Where-Object { $_.Name -match "^(HWNPU|ZE_|HF_|OV_)" -and $_.Name -notmatch "TOKEN" } | Format-Table -AutoSize }
  Sec "power plan" { powercfg /getactivescheme }
  Sec "power source" { Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue | Format-List BatteryStatus, EstimatedChargeRemaining }
  Sec "disks" { Get-PSDrive -PSProvider FileSystem | Format-Table Name, @{n = "UsedGB"; e = { [int]($_.Used / 1GB) } }, @{n = "FreeGB"; e = { [int]($_.Free / 1GB) } }, Root -AutoSize }
  Sec "source" {
    Push-Location $Repo
    try { & git log -1 --format="%H %cd %s" 2>&1; & git status --short 2>&1 } finally { Pop-Location }
  }
  Sec "go" { $g = Get-Command go -ErrorAction SilentlyContinue; if ($g) { & go version } else { "not installed" } }
  Sec "python and openvino" {
    $py = Get-Command python -ErrorAction SilentlyContinue
    if ($py -and $py.Source -notmatch "WindowsApps") {
      $f = Join-Path $Inputs "ov.py"
      Write-Utf8 $f ("import sys`nprint(sys.version)`ntry:`n    import openvino as ov`n    c = ov.Core()`n    print('openvino', ov.__version__, c.available_devices)`n" +
        "    if 'NPU' in c.available_devices: print('NPU', c.get_property('NPU', 'FULL_DEVICE_NAME'))`nexcept Exception as e:`n    print('openvino:', e)`n")
      & $py.Source $f
    } else { "python not on PATH" }
  }
  $hw = Join-Path $env:LOCALAPPDATA "hwnpu"
  Sec "hwnpu cache sizes (MB)" { "models {0}`ncache total {1}" -f (DirMB $(if ($ModelsDir -ne "") { $ModelsDir } else { Join-Path $hw "models" })), (DirMB $hw) }
  Write-Utf8 $f $sb.ToString()
  Log "system information in system.txt"
}

# ---------------------------------------------------------------------------------------------
# running a test

$Results = New-Object System.Collections.ArrayList
$ResultsJson = Join-Path $Out "results.json"
$Previous = @{}
if ((Test-Path $ResultsJson) -and -not $Rerun) {
  try { foreach ($r in (Get-Content $ResultsJson -Raw -Encoding UTF8 | ConvertFrom-Json)) { $Previous[$r.name] = $r } } catch { }
}

function Expand([string]$s) {
  return $s.Replace("{OUT}", $Out).Replace("{INPUTS}", $Inputs).Replace("{AGENT}", $AgentDir)
}

function Run-Test($t) {
  $name = $t.Name
  if ($Previous.ContainsKey($name) -and $Previous[$name].exit -eq 0) {
    Log "[$name] passed before, skipped (-Rerun runs it again)"
    [void]$Results.Add($Previous[$name])
    return
  }
  $r = [ordered]@{ name = $name; note = $t.Note; exit = $null; status = ""; seconds = 0; timeout_min = 0; peak_ws_mb = 0; peak_private_mb = 0; min_free_mb = 0; started = (Get-Date -Format o); command = "" }
  $needs = $null
  if ($t.PSObject.Properties["Needs"]) { $needs = $t.Needs }
  if ($needs -eq "image" -and -not (Test-Path $ImagePng)) { if (-not (Make-Image $ImagePng)) { $r.status = "no test image"; [void]$Results.Add($r); return } }
  if ($needs -eq "clip" -and -not (Test-Path $ClipWav)) { if (-not (Make-Clip $ClipWav)) { $r.status = "no test clip"; [void]$Results.Add($r); return } }
  if ($name -eq "agent") {
    New-Item -ItemType Directory -Force -Path $AgentDir | Out-Null
    Write-Utf8 (Join-Path $AgentDir "notes.txt") "Shopping: eggs, milk.`r`nThe secret number is 7351.`r`nCall the dentist on Monday.`r`n"
    Remove-Item (Join-Path $AgentDir "answer.txt") -ErrorAction SilentlyContinue
  }

  # go run ./cmd/EXE [SUB] [--dir DIR] ARGS
  $argv = @("run", ("./cmd/" + $t.Exe))
  if ($t.PSObject.Properties["Sub"]) { $argv += $t.Sub }
  # the commands that take the models directory (hwnpu-dev formats keeps the std flag package, which reads --dir too)
  # (hwnpu-image depth and face-detect keep theirs in hwnpu's own models directory: NoDir)
  $noDir = $t.PSObject.Properties["NoDir"] -and $t.NoDir
  if (-not $noDir -and ($t.Exe -in @("hwnpu-chat", "hwnpu-image", "hwnpu-audio") -or ($t.Exe -eq "hwnpu-dev" -and $t.Sub -eq "formats"))) { $argv += $Common }
  foreach ($a in $t.Args) { $argv += (Expand $a) }
  $cmdline = ($argv | ForEach-Object { Quote-Arg $_ }) -join " "
  $r.command = "go " + $cmdline
  $stdout = Join-Path $Out ($name + ".out.txt")
  $stderr = Join-Path $Out ($name + ".err.txt")
  $stdin = Join-Path $Inputs ($name + ".stdin.txt")
  $stdinText = ""
  if ($t.PSObject.Properties["Stdin"]) { $stdinText = (Expand $t.Stdin) + "`n" }
  Write-Utf8 $stdin $stdinText

  $saved = @{}
  if ($t.PSObject.Properties["Env"]) {
    foreach ($k in $t.Env.Keys) { $saved[$k] = [Environment]::GetEnvironmentVariable($k); [Environment]::SetEnvironmentVariable($k, $t.Env[$k]) }
  }
  $cwd = $Repo
  $limit = [int]($t.Min * $TimeoutScale)
  $r.timeout_min = $limit
  Log "[$name] $($t.Note)"
  Log "[$name] $($r.command)  (limit $limit min)"

  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $minFree = FreeMB
  $peakWS = 0; $peakPriv = 0
  try {
    $sp = @{ FilePath = $Go.Source; WorkingDirectory = $cwd; NoNewWindow = $true; PassThru = $true
             RedirectStandardInput = $stdin; RedirectStandardOutput = $stdout; RedirectStandardError = $stderr }
    $sp.ArgumentList = $cmdline
    $p = Start-Process @sp
  } catch {
    $r.status = "could not start: " + $_
    Log "[$name] $($r.status)"
    foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
    [void]$Results.Add($r); return
  }
  $null = $p.Handle  # keeps the exit code readable after it exits (PowerShell 5.1)
  $lastShown = 0
  $timedOut = $false
  while (-not $p.WaitForExit(2000)) {
    try {
      $m = ChildMem $p.Id $t.Exe
      if ($m[0] -gt $peakWS) { $peakWS = $m[0] }
      if ($m[1] -gt $peakPriv) { $peakPriv = $m[1] }
    } catch { }
    if ($sw.Elapsed.TotalSeconds - $lastShown -ge 30) {
      $lastShown = $sw.Elapsed.TotalSeconds
      $fm = FreeMB
      if ($fm -ge 0 -and $fm -lt $minFree) { $minFree = $fm }
      $line = Tail $stderr
      if ($line -eq "") { $line = Tail $stdout }
      if ($line.Length -gt 140) { $line = $line.Substring(0, 140) }
      Write-Host ("  {0:mm\:ss} ws {1} MB, free {2} MB | {3}" -f $sw.Elapsed, [int]($peakWS / 1MB), $fm, $line)
    }
    if ($sw.Elapsed.TotalMinutes -ge $limit) {
      $timedOut = $true
      Log "[$name] over $limit min, stopping it"
      try { & taskkill.exe /T /F /PID $p.Id 2>&1 | Out-Null } catch { }
      if (-not $p.HasExited) { try { $p.Kill() } catch { } }
      [void]$p.WaitForExit(30000)
      break
    }
  }
  $sw.Stop()
  foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
  $r.seconds = [Math]::Round($sw.Elapsed.TotalSeconds, 1)
  $r.peak_ws_mb = [int]($peakWS / 1MB)
  $r.peak_private_mb = [int]($peakPriv / 1MB)
  $r.min_free_mb = $minFree
  try { $r.exit = $p.ExitCode } catch { }
  if ($timedOut) { $r.status = "timeout" }
  elseif ($r.exit -eq 0) { $r.status = "ok" }
  else { $r.status = "failed" }

  # lines worth reading at a glance
  $err = ""
  if (Test-Path $stderr) { $err = [System.IO.File]::ReadAllText($stderr, $Utf8) -replace $AnsiRe, "" }
  # hwnpu-dev probe writes what it measured to standard output
  if ($t.Exe -eq "hwnpu-dev" -and $t.Sub -eq "probe" -and (Test-Path $stdout)) { $err += "`n" + ([System.IO.File]::ReadAllText($stdout, $Utf8) -replace $AnsiRe, "") }
  $pick = @($err -split "`r?`n" | Where-Object { $_ -match "ready:|tokens/s|error|panic|fail|not on the NPU|ErrNotOnNPU|compil|held at once|loaded in turn|oldest closed" } | Select-Object -Last 12)
  $r.highlights = $pick
  if ($name -eq "agent") {
    $ans = Join-Path $AgentDir "answer.txt"
    if (Test-Path $ans) { $r.answer = (Get-Content $ans -Raw).Trim() } else { $r.answer = "(no answer.txt)" }
    Log "[$name] answer.txt: $($r.answer)"
  }
  Log ("[{0}] {1} (exit {2}) in {3:N0} s, peak working set {4} MB, private {5} MB, least free memory {6} MB" -f $name, $r.status, $r.exit, $r.seconds, $r.peak_ws_mb, $r.peak_private_mb, $r.min_free_mb)
  foreach ($l in $pick) { Log ("    " + $l) }
  [void]$Results.Add($r)
}

function Write-Summary {
  # this run's results, then those of earlier runs into -Out that this one did not run again
  $all = @($Results)
  $names = @($all | ForEach-Object { $_.name })
  foreach ($k in $Previous.Keys) { if ($names -notcontains $k) { $all += $Previous[$k] } }
  ConvertTo-Json -InputObject $all -Depth 5 | ForEach-Object { Write-Utf8 $ResultsJson $_ }
  $sb = New-Object System.Text.StringBuilder
  [void]$sb.AppendLine("# hwnpu on Intel's NPU: $env:COMPUTERNAME, $Stamp")
  [void]$sb.AppendLine("")
  [void]$sb.AppendLine("| test | status | exit | seconds | peak WS MB | private MB | least free MB |")
  [void]$sb.AppendLine("|---|---|---|---|---|---|---|")
  foreach ($r in $all) {
    [void]$sb.AppendLine(("| {0} | {1} | {2} | {3} | {4} | {5} | {6} |" -f $r.name, $r.status, $r.exit, $r.seconds, $r.peak_ws_mb, $r.peak_private_mb, $r.min_free_mb))
  }
  foreach ($r in $all) {
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine("## " + $r.name)
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine($r.note)
    [void]$sb.AppendLine("")
    [void]$sb.AppendLine('`' + $r.command + '`')
    $hl = $null
    if ($r -is [System.Collections.IDictionary]) { if ($r.Contains("highlights")) { $hl = $r.highlights } } elseif ($r.PSObject.Properties["highlights"]) { $hl = $r.highlights }
    if ($hl) {
      [void]$sb.AppendLine("")
      [void]$sb.AppendLine('```')
      foreach ($l in $hl) { [void]$sb.AppendLine($l) }
      [void]$sb.AppendLine('```')
    }
  }
  Write-Utf8 (Join-Path $Out "summary.md") $sb.ToString()
}

# ---------------------------------------------------------------------------------------------

Log "hwnpu Intel check: level $Level, $($Selected.Count) tests, logs in $Out"
$env:CGO_ENABLED = "0"
Warm-Build
Collect-System

try {
  foreach ($t in $Selected) {
    Run-Test $t
    Write-Summary
  }
} finally {
  Write-Summary
  Log "summary in summary.md"
  if (-not $NoZip) {
    $zip = $Out + ".zip"
    Remove-Item $zip -ErrorAction SilentlyContinue
    try {
      Compress-Archive -Path (Join-Path $Out "*") -DestinationPath $zip -CompressionLevel Optimal
      Write-Host ""
      Write-Host "Send this file back: $zip"
    } catch { Write-Host "could not make the zip ($_); send the directory $Out" }
  }
}
