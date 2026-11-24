# Audio models

[日本語](audio-models.ja.md)

All audio models run through one command, `bonsaiai-audio` (`generate`, `tts`, `transcribe`, `pitch`, `notes`, `list`). Its flags and subcommands are in [cli.md](cli.md). Audio input to chat models (LFM3-Audio, d1-Omni) is in [llm-models.md](llm-models.md); the encoder is summarized here.

## Overview

| Model (models.yaml) | Use | Apple (Core AI) | Intel | AMD (IRON) | Memory (models.yaml) |
|---|---|---|---|---|---|
| `whisper-large-v3-turbo` (MIT) | transcription | lut8, all on the NE | IR (int8), checked on OpenVINO CPU only | encoder + decoder graphs, decoder q8 | apple 1.9 / amd 2.2 GB |
| `LFM3-Audio-2B-TTS` (LFM Open 1.0) | speech | text model lut8, depthformer and codec fp16 | IR (int8), CPU only | text model q8, depthformer and codec graphs | apple 2.4 / amd 2.9 GB |
| `magenta-realtime-2-small` (CC-BY 4.0) | music | all on the NE | IR, CPU only | SpectroStream decoder graph; LM, MusicCoCa, encoder on the CPU | apple 1.0 / amd 2.7 GB |
| `MOSS-SoundEffect-v2.0` / `-lut8` (Apache-2.0) | sound effects | all on the NE (M1 and M3) | IR, CPU only | text encoder, DiT and DAC graphs | apple 2.3 (2.2) / amd 5.0 GB |
| `swift-f0` (MIT) | pitch (F0) | all on the NE | IR, CPU only | CPU (ONNX Runtime, or milcpu) | 0.06 GB |
| `basic-pitch` (Apache-2.0) | notes, MIDI | partly off the NE | IR, CPU only | CPU (ONNX Runtime, or milcpu) | 0.07 GB |

- "IR, CPU only": Intel programs are written (lut8 read as int8 by `llm.IntelFormat`) and the whole path was checked on OpenVINO's CPU (`BONSAIAI_BACKEND=openvino`), never on an Intel NPU. See [intel.md](intel.md).
- Magenta RT2 weights are CC-BY 4.0: credit "Magenta RealTime 2 by Google DeepMind" where the music is used.
- SwiftF0 and Basic Pitch are published as ONNX only; they come from Hugging Face mirrors byte-identical to the official files, pinned by revision and sha256 (`internal/pitch` checks the sha256 too).

## Backends

### Apple: Core AI only

- On a Mac `bonsaiai-audio` uses Core AI alone (macOS 27). Without Core AI it fails with `errNoCoreAI` ("needs macOS 27"). `BONSAIAI_BACKEND` empty, `coreml` or `coreai` all mean Core AI (`audioDevice`, `cmd/bonsaiai-audio/backend.go`). See [apple-coreai.md](apple-coreai.md).
- Every model writes Core AI programs directly from the checkpoint (no `.mlpackage` step). Exports are `NAME.coreai` beside the checkpoint (`llm.CoreAIExport`); Intel IR goes in `Dir/intel`.
- Core ML is kept only for the CPU interpreter: `--cpu`, and pitch/notes on a machine without ONNX Runtime.
- The TTS text model's 1-token and 64-token programs are two functions of one `.aimodel` (`chat.ConvertShared`): weights once on disk and in the cache (0.81 + 0.82 GiB in 2 parts; Core ML needed 3.4 GB in 4).
- Core AI cannot choose compute units, so there is no `--decoder-on`.
- Core AI was closer than Core ML's NE to the CPU reference on Magenta (LM KL 0.0000 vs 0.0003 nats; top token 297 vs 295 of 300).

### AMD: IRON graphs per model

- The converter does not write AMD programs; each model has an IRON export in `tools/amd` (`iron_whisper.py`, `iron_lfm3speech.py`, `iron_moss_sfx.py` + `iron_moss_dac.py`, `iron_magenta.py`, `iron_supertonic.py`). Details in `../amd-npu-models.md`; shared IRON pieces in [amd.md](amd.md).
- `bonsaiai-audio` uses AMD when an AMD NPU is present and `BONSAIAI_BACKEND` is empty or `amd`/`xdna`/`iron` (`onAMD`). `PrepareAMD` (per package) takes the export from here, else from the sources, else exports with IRON and writes back. The key needs pinned sha256 in models.yaml (`chat.PinnedRecipe`); `like` does not inherit sha256, so `MOSS-SoundEffect-v2.0-lut8` lists them again.
- `iron.GraphProgram` turns a graph export into a `runner.Program`: meta `program` gives the program's shape; padded rows/columns are filled with 0 on input and cut on output. So the host code (sampling, MaskGIT, CFG, chunking) is shared with Apple/Intel.
- Models without an AMD export (pitch, notes) fall to the CPU (`chat.ConvertedDevice`).
- Exports in the sources: whisper 1.81 GB, LFM3-Audio TTS 3.13 GB, MOSS 6.3 GB (fp16) / 5.3 GB (lut8), Magenta decoder 0.09 GB, Supertonic 3 0.27 GB. See [distribution.md](distribution.md).

## Whisper (`internal/llm/whisper`, `internal/transcribe`)

- Default `openai/whisper-large-v3-turbo` (809M). `kotoba-tech/kotoba-whisper-v2.2` (Japanese distil, 2 decoder layers) runs with the same code when named by repo; not in models.yaml. It breaks on long audio (also in HF's long-form), so turbo is the one for long audio ([not-adopted/kotoba-whisper.md](../not-adopted/kotoba-whisper.md)).

### Programs (Apple / Intel)

| Arch | What |
|---|---|
| `whisper-encoder` | mel (1, 3000, 128) → 2 convs (kernel 3, the 2nd stride 2) as 3 stacked frames × one Linear (stride by `reshape`) → 32 pre-norm layers → (1, 1500, 1280) |
| `whisper-cross` | all decoder layers' cross K/V at once, (1, 160, 1500, 64); host splits into `xk_N`/`xv_N` |
| `whisper` | 4-layer decoder, own KV cache (448), cross K/V as inputs, learned positions as one-hot `pos` (1, q, 448) × table (no gather). Vocab 51866 padded to 51872 for lut (`blocks.LinearPadded`) |

- `decode.Session.SetInputF16`: an input written once and kept across steps, marks and rewinds (the cross K/V).
- `qchunk` (query split of encoder attention) defaults to 0: 0/300/500/750 rows gave 520/518/537/555 ms, same output; the 90 MB score tensor is not slower.

### AMD (`iron_whisper.py`)

- Two graphs: `encoder` (log-mel, conv1/conv2, 32 layers, and the cross K/V of the 4 decoder layers) and `decoder` (1 token; self KV cache and cross `xk{i}`/`xv{i}` as graph states; token and position embeddings added by the host).
- conv2 stride 2: host lays out conv1's im2col with even frames first, odd after; frame t reads even t, odd t, odd t−1 (the last written to a state shifted by one row).
- Decoder: q8 weights (`--decoder bf16` for bf16; q8 GEMV widened to K multiple of 256), MHA `num_pipelines` 4 (cross MHA 3.9 → 2.0 ms for 4 layers), fused QKV GEMV, `linear_add` for residuals. Logits table padded to 52224.
- Encoder matmuls MEDIUM (bfp16): 0.84 s vs 1.71 s HIGHEST a window, same text.
- One decoder token is bound by op count, not weight reads: ~0.13–0.15 ms fixed cost per op.

### Host

- Features (`whisper.Features`, `FeaturesLong`) match WhisperFeatureExtractor within 2.7e-6; long audio computes the mel and its floor over the whole clip (as HF).
- Tokenizer: GPT-2 ByteLevel pre-tokenizer; 0 differences from HF on 40 fixed + 3040 random texts.
- ≤ 30 s: one pass with `<|notimestamps|>`. Longer: timestamped windows as transformers' long-form generation (`timestampRules` = `WhisperTimeStampLogitsProcessor`, `segments` = `_retrieve_segment`); the next window starts at the last closed segment's end. Cutting at fixed 30 s lost text.
- No language given: picked from the logits after `<|startoftranscript|>`.
- `transcribe.PrepareAMD` on AMD; `programs` interface hides converted vs IRON programs.

### Accuracy

- Encoder fp16 vs fp32 cosine 0.9978. Layers 20–21 make attention-sink values (~300) at silent positions; tiny input errors move which positions become sinks. HF in fp16 shows the same (0.9978), so it is fp16 itself, not the NE. Speech positions are right and transcripts unchanged.
- To check NE numerics: `TestEncoderOnDevice` (`internal/transcribe/device_test.go`, `BONSAIAI_TEST_WHISPER_REF`, `BONSAIAI_TEST_WHISPER_ENCODERS`, `BONSAIAI_TEST_WHISPER_LAYER` with `-layers N`), NE vs Core ML CPU vs HF.
- Formats (turbo, M3): LibriSpeech dummy (73 clips, 481 s) WER: HF fp32 4.43 %, fp16 4.43 %, lut8 4.43 %, lut6 4.87 %. Japanese (48 `say` clips) CER: 3.60 / 3.60 / 3.74 / 3.94 %. **Default lut8.**
- AMD: random-input encoder vs IRON reference cosine 0.9999 (8 layers) → 0.99 (32 layers), but real speech transcribes right; 88.3 s English in 3 windows, same text and cuts.

### Numbers

| Machine | Result |
|---|---|
| M3 16 GB, Core AI | first run 1 m 44 s (download 33 s, convert and compile 1 m 06 s), then load 0.1 s; 30 s window (encoder + cross) 0.53–0.56 s; decoding 244–258 tok/s; English `say` clips: 5.6 s in 0.70 s (8× real time), 71.8 s in 2.95 s (24×); NE 0.78 GiB |
| Ryzen AI 9 365 | decoder 13 ms/token (78 tok/s); encoder 0.84 s a window; 88.3 s English in 6.8 s; peak 2.0 GiB; IRON export 19 s |

## LFM3-Audio

### Audio input (encoder `lfm3-audio-encoder`, `internal/llm/lfm3audio`, `internal/audio`)

- Reads LFM3-Audio and LFM3-Omni layouts. Three stride-2 convs over (time, mel) cut time to 1/8, a Linear, causal LFM2 layers (short conv, sliding-window attention, QK norm, RoPE θ 1e6), RMSNorm, projector (LayerNorm, Linear, GELU, Linear).
- Fixed length per program (`-frames`, default 3000 = 30 s); `internal/audio` picks the shortest program that fits, else `ErrTooLong`. Outputs after the clip are zeroed by a halved mask at each stage, so a short clip in a long program gives the same embeddings.
- Preprocessing (`lfm3audio.Features`) = llama.cpp's: 16 kHz, 400-sample periodic Hann every 160, 128 Slaney mels, log10, (v + 4) / 4; within 1e-4 of librosa. WAV readers (8/16/24/32-bit PCM, float32, channels averaged); other rates resampled with a Hann-windowed sinc.
- Raw live ASR ("Perform live ASR.") is not handled.

### TTS (`internal/speech`, `internal/llm/lfm3gen`)

- Flow: the text model reads the TTS prompt (`Perform TTS in {language} using a {female|male} voice.` … `<|audio_start|>`; Omni adds `<think></think>`) and gives h (after the final norm, `-hidden-out 1`, `decode.Session.Hidden`). Per 80 ms frame the depthformer picks 12 codebooks by MaskGIT in random order (4 at a time, 3 passes), sampling with temperature, top-k (ties kept) and a repetition penalty. An end code stops. The frame's feedback embedding (sum of `codebook_embedding` rows) is the text model's next input.
- `lfm3-depthformer`: one MaskGIT pass; h projection + code/mask embeddings (host table lookup), bidirectional LFM3 attention layers, per-codebook norm and code table matmul.
- `lfm3-codec`: codec decoder (`audio_tokenizer/`, liquid_codec): FSQ values, each frame copied to 6 positions, 8 causal LFM3 layers (window-30 attention, short conv), log-magnitude and phase. Layer kinds and widths are read from weight shapes.
- Codec runs in chunks of `-codec-frames` (default 64 = 5.1 s) after all frames are generated, overlapped by the decoder's look-back; the head of each chunk is dropped. Same as streaming (causal, relative RoPE).
- ISTFT (`speech.ISTFT`) on the host = llama.cpp's; 24 kHz output.
- Sampling defaults: LFM3-Audio temperature 1.0, top-k 128, penalty 1.1, window 4096; Omni 0.4, no top-k, 1.2, 64; `generation_config.json` `audio_*` overrides. Go PCG RNG: same seed does not give llama.cpp's sound.
- Text model format lut8: at lut4 the speech transcribed back broken (M3); lut8 matched word for word.
- `bonsaiai-dev convert` uses `-codec-frames` (not `-frames`) because flags are shared across archs (`TestParamNames`).

AMD (`iron_lfm3speech.py`, export `lfm3-audio-2b-tts.amd-npu2`):
- Text model from `iron_export_lm.py lfm3tts` via `iron_lfm3tts.py` (MLP and head q8, `--q4`), writing h to state `bonsaiai_hidden`; host embedding rows (`bonsaiai_embed_rows`) let the host write the feedback row (`iron.Model.StepRow`). `iron_export_lm.py` itself is left unchanged so `amdtools.LMHash` and all chat exports keep their keys.
- Depthformer: 12 codebook rows packed into MHA's 64 rows; heads of 32 laid out in 64 as [x0..x15, 0×16, x16..x31, 0×16] so rotate_half pairs stay right; RMSNorm over 64 with weights / √2 and eps / 2.
- Codec: 64 frames a call; 6× copy by the host; short conv as per-tap products written to a state shifted by k rows.
- Host tables in `tables.safetensors` (bf16).
- Pitfall: opening the checkpoint's 4 GB tables before loading the text export made `CREATE_HWCTX` fail with `EINVAL` about every other run (64 MiB device heap registration). The cause is the heap's address alignment, now fixed (see [AMD](amd.md)); `Load` still reads the text model first.

Numbers:
| Machine | Result |
|---|---|
| M3 16 GB, Core AI | lut8: 1.6–1.7× real time (8.3–9.0 s of speech in 5.1–5.3 s: text model 3.2 s, depthformer 1.5 s, codec 0.1 s); first run 1 m 32 s (convert and compile 1 m 23 s), then load 0.2 s; English and Japanese voices transcribed back to the text by Whisper |
| Ryzen AI 9 365 | 0.43× real time: per frame text model 107 ms, depthformer 3 × 24 ms; codec 72 ms per 64 frames; peak 2.6 GiB; IRON export 52 s. All CPU: 4.16 s of speech in 42.5 s |

## Magenta RealTime 2 small (`internal/music`, `internal/tflite`)

- 230M. Style (MusicCoCa's 12 codes from text and/or audio) → SpectroStream 12-level RVQ codes at 25 frames/s → 48 kHz stereo. Pinned `google/magenta-realtime-2` revision (1.7 GB).

| Part | Where | Notes |
|---|---|---|
| Temporal LM (12 layers, 1024, 8 heads, one sink K/V per attention, window 41 + current) | NPU, 1 program | K/V ring outside; returns the new frame's K/V |
| Depth decoder (2 layers, 768, logits 12294, softcap 30) | NPU, 1 program | 12 calls a frame |
| Sampling (top-k 40, temperature 1.3, Gumbel), conditions | host | CFG strength passed as binned condition tokens, so no doubled batch |
| SpectroStream decoder (causal 2-D conv / transposed conv, ELU) | NPU, 25 frames + 16 back | iSTFT (960 / 480) on the host |
| SpectroStream encoder (for `--continue`) | NPU, 41 frames | host STFT and RVQ |
| MusicCoCa text encoder | NPU (`textEncoderCut`) | SentencePiece Unigram from `spm.model` (`tokenizer.LoadSentencePieceUnigram`); embedding lookup on the host (fp16 table 24 MB) |
| MusicCoCa music encoder | NPU | 10 s chunks of 16 kHz mono, host preprocessing (TFLite PAD, RFFT2D, COMPLEX_ABS) |

- TFLite is read without dependencies (`internal/tflite`, FlatBuffers → `internal/onnx` graph). `onnx.Graph.Unroll` unrolls a WHILE whose count is constant (music encoder's 12 layers), so per-layer weight slices fold to constants. The 6-D patch reshape (Core ML max 5-D) is a host index table.
- Projection noise reproduces numpy `RandomState(0).randn` (`LegacyNormal`).
- Resampling uses resampy `kaiser_best` ported exactly (`Resample`, 3e-10 from resampy). A Hann sinc moved the MusicCoCa embedding 42 % and changed codes.
- Style mixing (`Mix`, "jazz@2" song.wav): weighted mean of directions, length = weighted mean of lengths. Audio embeddings have length ~28, text 1, and quantization depends on length, so a plain mean would be audio only.
- Continuation (`Session.Prefill`): forced codes, style masked, up to the last 500 frames (20 s). Notes/drums (`PianoRoll`, `notes.ReadMIDI`) as in the C++ runner's `populate_condition_tokens`; channel 10 dropped. CFG as condition tokens (as Python's generate), not the C++ runner's 3× batch.
- Streaming: decode every 25 frames; `-o -` streams WAV to stdout; the file is also written every second. Generate one extra frame for the decoder's look-ahead. The first 4 STFT frames are dropped before the iSTFT (as the reference).

### fp16 numerics (NE, Intel)

- SpectroStream decoder values grow per block up to 1.4e7 (past fp16's 65504). Within a block max/RMS stays in hundreds, so each block computes u = value / s (`dgraph.scale`, `decoderScales`): conv bias / s, ELU as relu(u) + (exp(clip(s·u, −30, 0)) − 1)/s, power-of-two scale changes, output = STFT / 64 (host multiplies back), block sums saturated at ±60000.
- Scales too large make bias and ELU's negative side subnormal; the NE flushes fp16 subnormals (< 6.1e-5) to 0. Reproduce with milcpu `FlushSubnormals`. Scales in use: 1, 1, 8, 16, 64, 1024, 1024, output 64 (max 1.8e4 on the reference song).
- The encoder's values stay under 1e4, scale 1.
- Load runs each SpectroStream program once on silence so failures show at start.
- `Where` lowered as cond·a + (1 − cond)·b: b + cond·(a − b) with b = −3e4 rounds a − b to steps of 16 (NE rel. error 0.32). Constants clamped to fp16 range (−2.38e38 became −inf, × 0 = NaN).
- Core AI: `elu` lowered as max(x, 0) + α(exp(min(x, 0)) − 1).
- Embedding lookup stays on the host: a one-hot (128 × 16000) × table version made every NE prediction on the machine fail after ~100 s for ~20 min (M1). Test new program shapes on the NE alone and check a small known program still runs.
- MusicCoCa RVQ: later levels flip on tiny differences; Core AI vs Core ML differ from the 3rd code on "disco funk" with embedding cosine 0.999997+. Compare embeddings, not codes.

### AMD (`iron_magenta.py`)

- Only the SpectroStream decoder is on the NPU; LM, MusicCoCa and the encoder run Core ML exports on the CPU (`dev` nil, skipped by `checkNPU`). Needs the checkpoint for the CPU parts.
- Layout: rows (time, frequency), columns channels. Convs: padded grid, im2col by IRON Copy taps, one GEMM. DMA's innermost dim ≤ 1023 words, so 1024-channel 3-tap goes per tap to column blocks. Transposed conv: per output phase a normal conv on the input (Flax weights flipped) scattered by taps.
- No mask: the host puts the clip's first frame at row 0; the grid's zero rows give the same zeros. Window 48 frames.
- ELU: float32 kernel `kernels/elu_f32.cc` (exp cut at −30). bf16 needs no block scales.
- Output 7×7 conv (64 → 2): im2col of the 7 time taps only (K 3136 → 448, copy 578 → 82 MB), frequency taps as GEMM columns summed by shifted copies (`out_sum`).

### Numbers

| Machine | Result |
|---|---|
| M3 16 GB, Core AI | 2.1× real time, all on the NE; first run 1 m 12 s (load 58 s), then style 0.3 s, load 0.1 s; two runs byte-identical |
| M1 16 GB (Core ML era) | LM 22 ms/frame lut8 (fp16 28 ms; real time needs 40 ms); NE decoder rel. error 6.8e-3 vs reference; note following 14/18 notes (5 without `--notes`) |
| Ryzen AI 9 365 | decoder 0.86 s per 1 s of audio (CPU 5.7 s); 4 s of music in 15.4 s (~0.3× real time; all CPU 0.14×); peak 2.5 GiB (8 s); IRON export 13 s |

- LM format: lut8 default (NE KL 0.0064 at temperature 1.3; fp16 0.0002). `--format` fp16 / lut6.
- Footprint (M1): codes given 0.42 GB, audio style 0.55 GB, text style 0.97 GB, continuation 0.96 GB.

## MOSS-SoundEffect v2.0 (`internal/sfx`, `bonsaiai-audio generate -m sfx`)

- Structure (OpenMOSS MOSS-TTS `moss_soundeffect_v2`): Qwen3-1.7B last hidden state (after final norm) as text encoder, Wan 2.1 1.3B DiT (30 blocks, width 1536, 12 heads, cross attention to 512 text rows), DAC VAE (128 latent channels, 50/s, 48 kHz mono, rates 8 5 4 3 2). Flow matching Euler, shift 5, card defaults 100 steps, CFG 4.
- Negative prompt is empty → context all zeros, no text encoder pass. Always denoise 30 s (1500 frames) and cut the asked length; prompt gets " duration: 10.0s".
- `.pth` read in Go (`internal/torchzip`: torch.save zip + pickle, only the opcodes torch.save writes). DAC weight norm folded to g·v/|v| at conversion.
- Two models.yaml entries: `MOSS-SoundEffect-v2.0` (machine 16, text fp16) and `-lut8` (machine 8, `moss-soundeffect:8g`). `sfx.Formats`: ≥ 16 GB text fp16 / DiT lut8; else both lut8, with layers 0, 1, 2, 26, 27 of the text encoder in fp16 (`TextFP16`). Intel: int8 with the same 5 layers fp16.
- CPU tables (token embeddings, final norm, DiT time embedding etc., 39 tensors) are cut to `sfx-tables.safetensors` (0.69 GB, `chat.WrittenTables`) and shared via the sources, so a run needs no checkpoint (11.5 GB). Checkpoints are fetched only to convert or to write the tables.
- WAV: 48 kHz stereo (mono duplicated), gain 1; fade only when asked.
- Text encoder runs first and is released before the DiT loads.

### Text encoder (Qwen3-1.7B, `internal/llm/qwen3` NoHead, 256 rows)

- Massive activation up to 1.2e4 on the first token from layer 3. int8 gave inf at layer 2 down_proj on the NE (scale applied after the sum); lut8 dequantizes first and is fine.

| Format | Rel. error (RMS) | Size |
|---|---|---|
| fp16 | 0.66 % | 2.8 GB, 4 parts |
| lut8 | 10 % (first token cosine 0.943) | |
| lut8 + `TextFP16` | 4.9 % (cosine ≥ 0.989) | 1.7 GB |
| lut8 kmeans-minmax | 19 % | |
| lut4 | 127 % | |

### DiT (6 programs of 5 blocks)

- Time embedding, its projection and each block's scale_shift_table on the host, passed as input `mod`.
- Wan RoPE rotates pairs (2j, 2j+1); q/k projection rows (and bias, RMSNorm weight) are permuted to `blocks.ApplyRope`'s (j, j+64). Scores unchanged.
- Residual max 1.4e4, so it is kept at 1/4 (`DitScale`). Pitfall: an untyped integer `DitScale` made `1/DitScale` 0 and erased patch embedding and cross output; use 4.0.
- Formats vs reference velocity: fp16 2.1 % (1.65 s a call, 2.7 GB), **lut8 2.7 %** (1.28 s, 1.4 GB), lut6 12.5 %.

### DAC decoder on Core AI (`internal/sfx/dac.go`, `DACRevision` 3)

- Neural Engine dims are ≤ 16384, so each stage runs in time windows with halos wider than the receptive field; masks on every conv input recreate zero padding so windows equal one long pass.
- Snake sin: u = y / π, n = (u + mg) − mg2, w = 2(u − n), sin(y) = sin(π/2 · w) by a degree-9 series. Two rounding inputs `mg`, `mg2` (both 1536): with one, Core AI folds (u + mg) − mg to u. Clip ±1500 kept. Channels with α < 1e-2 use αx²(1 − y²/3 + 2y⁴/45) (1/α beyond fp16).
- All convs along the height axis (`conv1d`, x as (1, C, T, 1)). Core AI's width-axis conv with kernel > 1, ≥ 512 channels and a computed input is wrong (1.26 rel. RMS).
- Dilation as rows (`phaseConv`): T frames as (1, C, T/d, d), an undilated k × 1 height conv. Core AI's dilated convs and explicit `pad` before a conv were wrong. Frames per call chosen so outputs are multiples of 9: 1026, 3195, 3996, 4500, 7992.
- Transposed convs of stride ≠ 2 as polyphase (kernel-2 conv to s·Cout channels, reshape/transpose to time). Core ML put stride 5 and 7 on the CPU and stride 8/4/3 were 22/45/4 % off on the NE (M1).
- M1 (h13) fixes: `sin` goes to the GPU; clip fused into elementwise ("NEFUSED_EW: EW: MAX") and the subtraction of constant-scaled inputs ("KernelPreScales") are rejected; stage 3 at 4998 frames "unexpected failure", 4500 works. M3 and later (h15g/c, h16g/s/c, h17g/s/c/d) take `bonsaiai-dac-r3-fp16-sin-native.coreai` instead: sin itself, unclipped (model-recipes.yaml rewrite; offline one region a stage, vs 8–9 on h13g/h14g/h14c).
- Check placement offline: `xcrun coreai-build compile dac-N.aimodel --platform macOS --architecture h13g --preferred-compute neural-engine`, then `manifest.plist` `mps.fullyPlacedOnANE`; reasons in `/usr/bin/log show --predicate 'process == "coreai-build"'` (zsh's `log` is a builtin).
- Core ML export (`--cpu` only) uses plain `sin`; Intel keeps the older fp16 series and `(v + 1536) − 1536` rounding with an input constant.
- Reference pitfall: the `runner` CPU path computes in fp32 unless `BONSAIAI_DEV=cpu.f16=1`, so fp16 rounding tricks do nothing there. Decide the reference's precision first (aicheck runs milcpu with `RoundF16`).

### AMD (`iron_moss_sfx.py`, `iron_moss_dac.py`, `sfx.AMDExportVersion` 2)

- Text encoder: 28 layers, 256 rows, 4 layers a command; `--text bf16` (fp16 model) or `q8` (lut8 model; `TextFP16` layers bf16). Final norm and embeddings on the host.
- DiT: residual / `DitScale` as on Apple; q/k RMSNorm over all 1536; Wan RoPE `two_halves=False`; self attention `iron_anima.dit_attention` with 1500 keys; cross K/V in the first 512 of 1536 state rows with `nt` = 512 (MHA keys must be ≥ query pad). Exact GELU. MEDIUM matmuls. One block a command.
- DAC: Snake in float32 (`kernels/snake_f32.cc`): reduction (αx, v/π, nπ) in float32 limbs, series to r⁷ in bf16 products, sin and sin² in two limbs; error vs exact 0.0013 mean (bf16 rounding alone 0.0012). No small-α branch needed. 0.59 ms per million values.
- DAC convs: Copy-tap im2col (dilation in the stride) + GEMM; 2048 channels split in 1024 column blocks; transposed conv (stride s, kernel 2s) as a 3-tap conv to s × Cout. No masks: the host places windows at the clip start or a halo before. For < 512 output channels, weights are A and im2col column-major B (stage 4 46 → 33 ms).
- DAC GEMMs stay HIGHEST: MEDIUM drops SNR vs CPU to 25.8 dB (34.6 at HIGHEST), f32 accumulation does not help, it saves only 20 %. Per-channel power-of-two scaling cut the error only 5 %; split bfp16 is slower than HIGHEST.
- Window frames per stage: 768, 2048, 2048, 5120, 8192 (row multiples of 256).

### Numbers

| Machine | Result |
|---|---|
| M3 16 GB, Core AI, `:8gb`, 5 s, 30 steps | all on the NE; text encoder 1.6 s, load 0.1 s, 1.8 s a step, 55 s total (first run: text encoder 15 s, load 3 m 20 s); runs byte-identical |
| M1, Core AI, 5 s, 30 steps, from tables | all on the NE; 1 m 31 s (fp16), 1 m 43 s (lut8); DAC 5 s latent in 4.1 s (rel. RMS 0.0085 vs CPU) |
| Ryzen AI 9 365 | 1.24 s a velocity (41 ms a block), 2.8 s a CFG step; DAC 1.5 s for 5 s (CPU 5.7 s, SNR 34.5 dB); 5 s at 25 steps 1 m 06 s; peak 4.71 (fp16) / 4.75 (lut8) GiB at 10 s; IRON export 28 s |

- Accuracy: 4 steps from the reference noise, waveform cosine 0.992 (16 GB setup) / 0.990 (8 GB). AMD vs Core ML fp16 on CPU: text output cosine 0.99963, first velocity 0.99978.

## Supertonic 3 (`internal/supertonic`)

### AMD (`iron_supertonic.py`, `supertonic.AMDExportVersion` 1)

- The flow (`estimator-tT-lL`, one step), the vocoder (`vocoder-lL`), the text encoder (`encoder-tT`) and the duration predictor's sentence encoder (`duration-tT`) are IRON graphs of each of `Sizes`. No ONNX Runtime. The text graphs' windowed relative attention (VITS's) runs on the host between commands (`iron.on_host`, Go's `Graph.After`, `relAttention` on sgemm): IRON's MHA has no relative-position band, and a bf16 diagonal Copy breaks the DMA's 4-byte unit; the GEMMs (most of the energy) stay on the NPU. The sizes share one image of their weights (596 → 255 MB).
- Both branches of the guidance run in one call: the stream's rows are the conditioned branch's frames and the unconditioned's, each between 16 zero rows. The unconditioned text attention is a constant row (its text is one token at every position). The style keys are constants; the host computes the time rows, RoPE tables and the update x + (4 v_c − 3 v_u) / total.
- pwconv1's bias and GELU are one operator (`iron_biasgelu.py`); pwconv2's bias is a weight column fed by a padded GELU column of exactly 16 (10 → 8 operators a ConvNeXt block, 5–8 % faster). MEDIUM GEMMs with GELU fused into the epilogue were 9–12 % faster still but dropped a step's accuracy from 25–30 to 18–21 dB, so not taken.
- Depthwise convolutions are `iron_dwconv1d.py` (K taps up to 7, dilation, bias in one operator, tiles of 16 rows × 128 channels; a row-window FIFO ran out of buffer descriptors). Left edge padding copies each branch's first frame; the right is masked zeros, as the Mac's fixed-length programs.
- bf16 throughout (HIGHEST): one flow step's update is 21–27 dB from padded ORT; the vocoder 37 dB once the latent's unnormalization is folded into the embedding (29 dB when the host unnormalized it in bf16). MEDIUM did not make the graph faster: operator reconfiguration dominates (a ConvNeXt block's GEMM pair takes about 1 ms at any row count).
- The CPU interpreter's `conv1d` now goes through `conv2d` (sgemm, im2col): the text encoder took 1.5 s in a one-thread loop, then 57 ms (before it moved to the NPU). sgemm takes at least 8 Mflop a goroutine (each one packs all of A).

### Numbers

| Machine | Result |
|---|---|
| Ryzen AI 9 365 | flow step 70 / 77 / 91 ms (64 × 32 / 128 × 64 / 256 × 128), vocoder 21 / 27 / 40 ms; 4.25 s of English in 0.68–0.70 s (6.2x; text 0.04 s); +7.4 W, 1.15 J a second of speech (CPU ONNX Runtime, 4 threads: 7.6x, +41.9 W, 5.5 J) |

## SwiftF0 (`internal/pitch`)

- lars76/swift-f0 0.3.0, 14,386 params: pitch (95 bins, 46.875–2093.75 Hz) and voicing every 16 ms at 16 kHz. ONNX initializers read with protowire (`onnx.go`); layers written in Go.
- Host: 3 STFTs (1024/2048/512, hop 256) → 128 bands, log(+1e-8), in float64 (sparse filterbank, precomputed twiddles); final decode (softmax of best bin and neighbours in log frequency). NPU: the CNN, one program of 1896 frames (1875 + 11 + 10), windows of 1875 as `SwiftF0.detect`.
- `mask` multiplies every ReLU output and input so a short clip in a fixed program equals its own-length program.
- Large dilations (17, 27, 40, 48) are rewritten as one 1×1 conv over k band-shifted copies stacked on channels (`shifted`); Core ML put them on the CPU otherwise. All 115 ops on the NE (`Revision` 3).
- Accuracy: vs ONNX reference max 0.11 cent (CPU fp16), NE 0.23 cent over 2581 frames, voicing all equal. Core AI vs Core ML: same pitch for all frames.
- M1 (Core ML): 41.3 s in 0.24 s, footprint 58 MB. M3 Core AI: first run 7.8 s, then load 0.1 s.

## Basic Pitch (`internal/notes`, via `internal/onnx`)

- `nmp.onnx` (36K). `internal/onnx` cuts the graph (`Graph.Cut`): static part lowered to MIL, dynamic tail evaluated on the host (`EvalShapes`). CQT (22050 Hz, 2 s windows; sample axis 43844 > 16384) and log normalization on the host; harmonic stacking and the CNN on the NPU.
- Outputs concatenated flat (`Flat`): last-axis concat broke note activations (59 % off).
- Windowing (30-frame overlap), note making (`output_to_notes_polyphonic`, inferred onsets, melodia trick, pitch bends) and MIDI writing ported from basic-pitch; notes from reference activations match exactly.
- The (3, 39) conv + ReLU after harmonic stacking stays off the NE (also split to width ≤ 16), on both Core ML and Core AI.
- M1 (Core ML): 5 s in 0.14 s, NE activation rel. error 2 %, same 10 notes. M3 Core AI: 0.2 s after the first run; same 8 notes as the CPU reference.

## Not adopted

- SpectroStream on the GPU in fp32: the goal is the NPU; block scales made the NE decoder right (rel. 6.8e-3) and faster (10 s decoded in 2.0 s vs 4.5 s GPU). `--decoder-on` removed.
- Core ML on the Neural Engine for `bonsaiai-audio`: Core AI only on Macs.
- Whisper lut6: WER 4.87 % vs 4.43 %, the encoder loses most. Whisper fp16 as default: 2 parts and about half the decode speed, same transcripts.
- Whisper encoder `qchunk` 500: no speed gain. `LayerNormMax`: no change (not an fp16 overflow).
- MOSS text encoder plain int8 (inf on the NE), lut4 (127 %), kmeans-minmax lut8 (19 %). DiT lut6 (12.5 %).
- Magenta one-hot embedding lookup on the NE: stalled the NE for the whole machine.
- Magenta LM on AMD's NPU (`iron_magenta_lm.py`): temporal 34 ms and depth 58 ms a frame, same as the CPU, and it would compete with the decoder for the NPU. Export kept for measuring.
- Magenta output layer with FLUX VAE's `GridGEMM`: failed in IRON ("aiex.npu.rtp_write must be a Value"); cause not studied.
- Fusing Snake into the following GEMM on AMD: IRON's GEMM has no post-op hook, and before a 7-tap im2col it would cost 7×. Faster kernel instead.
- MOSS DAC GEMMs at MEDIUM on AMD (see above).
- MMAudio for sound effects: its text features use Apple DFN5B-CLIP under a research-only license.
- Separate commands `bonsaiai-transcribe`, `bonsaiai-tts`, `bonsaiai-pitch`, `bonsaiai-notes`, `bonsaiai-music`, `bonsaiai-sfx`: merged into `bonsaiai-audio`.
- Not handled: chat replies as speech (`<|mixed_start|>`), streaming TTS output, LFM3-Audio raw ASR, LFM3-Omni TTS (`model.depth_decoder`), ftfy `fix_text` for MOSS prompts, Magenta's true CFG (3× batch).

## Open

- Listening tests: none of TTS, Magenta, MOSS was judged by ear (numbers and re-transcription only). TTS vs llama.cpp from the same codes.
- Intel NPU hardware: all audio models only checked on OpenVINO's CPU (MOSS: 20 steps, 18 s a step; whether the NPU compiler takes ConvolutionBackpropData, Elu, etc.).
- M2: Core ML NE decoding of SpectroStream once hung the machine; not re-checked with block scales or Core AI.
- MOSS DAC r3 speed and values on an M3 device (the offline compile places it).
- macOS 26 error path of `bonsaiai-audio` not checked on a real machine.
- Whisper: real recordings (noise, overlap), Japanese on AMD, temperature fallback, `condition_on_prev_tokens`; `-v` reports `ja` for English windows and long English has many capitals (also on Core ML). No `/v1/audio/transcriptions` API.
- Whisper on AMD: LayerNorm not fused into the GEMV (~1.9 ms; `QGEMVQKV` normalizes by RMS only); IRON's per-op fixed cost (~0.13–0.15 ms) bounds small decoders.
- LFM3-Audio TTS on AMD: skip the unused head (0.27 GB), run `depth_linear` once a frame (4 ms), depthformer weights q8.
- Magenta: louder than the reference song saturates (±60000); 44.1 kHz continuation not compared; MusicCoCa audio side and encoder on AMD are CPU.
- Basic Pitch: why the (3, 39) conv stays off the NE.
- SwiftF0: real voices and instruments, streaming (`PitchStream`), `segment_notes`.
