# Variables: hwnpu's environment

[日本語](variables.ja.md)

State: sorted out on 2026-11-22. hwnpu's processes read four kinds of variables; the developers'
switches and the tests' are each one family with one way of naming, so that what a variable is
for can be told from its name.

| Kind | Names | Read by |
|---|---|---|
| Settings | `HWNPU_CACHE*`, `HWNPU_MODEL_SOURCE*`, `HWNPU_MODEL_HUGGING_FACE`, `HWNPU_MODELS`, `HWNPU_MODEL_ALLOWLIST`, `HWNPU_ACCEPT_LICENSE`, `HWNPU_MEMORY`, `HWNPU_BACKEND`, `HWNPU_DEVICE`, `HWNPU_COREAI_UNITS`, `HWNPU_COREML_UNITS`, `HWNPU_COREAI_PREBUILT`, `HWNPU_MOE_WIRE`, `HWNPU_WEBDAV_*`, `HWNPU_HF_CLIENT_ID`, `HWNPU_DAEMON*`, `HWNPU_JOBS`, `HWNPU_PROFILE`, `HWNPU_ENERGY`, `HWNPU_NO_RUN_LOG` | The programs, from the environment and the `.env` files (`internal/envfile`); with hwnpud, whose each one is:  [hwnpud](hwnpud.md#settings) |
| The machine's libraries | `HWNPU_ONNXRUNTIME`, `HWNPU_OV_PYTHON`, `HWNPU_XRTSHIM`, `HWNPU_XRT_CORE`, `HWNPU_ZE_LOADER`, `HWNPU_IRON_ENV` | The backends; hwnpud's |
| The developers' switches | `HWNPU_DEV=KEY[=VALUE],...` | `internal/dev`; a process's own, never sent to hwnpud |
| The tests' | `HWNPU_TEST_<GROUP>_<ROLE>`, `HWNPU_TEST_NPU` | `go test` only |

Not here: the Python tools' own switches (`tools/amd/iron_*.py`: `HWNPU_LFM3_FUSED_*`,
`HWNPU_Q_CORES`, ...), which an export reads, and `hwnpu-dev powerbench`'s `HWNPU_POWERMETRICS`
(a command line). Removed: `HWNPU_NPU_LOCK` (the Neural Engine's lock is never skipped: several
apps share the Neural Engine through hwnpud).

## The developers' switches: HWNPU_DEV

Experiments, measurements and traces: a conversion's other form, a policy off, a trace written.
None changes what a user gets by default, and none is a setting of the user's.

- Items separated by commas; an item is a key (on) or `KEY=VALUE`; a key's group goes before a
  dot (`moe.`, `iron.`, `coreai.`, `ailower.`, `mxfp4.`, `draft.`). `HWNPU_DEV=moe.time,moe.prefetch=0,ailower.trace`.
- `internal/dev.Keys` lists every key with what it does; `TestKeysUsed` checks that every key the
  code reads is there and every key there is read. An unknown key is told once on standard error.
- It is read at each call (`dev.Get`, `On`, `IsOff`, `Value`, `Int`), so that a test's
  `t.Setenv` takes.
- hwnpud does not take a client's: an experiment runs in the developer's own process
  (`HWNPU_DAEMON=off`; the default is on), with the Neural Engine's lock as before.

What were variables of their own:

| Before | `HWNPU_DEV`'s key |
|---|---|
| `HWNPU_ANIMA_ATTENTION` | `anima.attention` |
| `HWNPU_DRAFT_POLICY` | `draft.policy` |
| `HWNPU_DRAFT_RATIO` | `draft.ratio` |
| `HWNPU_DRAFT_TRACE` | `draft.trace` |
| `HWNPU_MTP_DRAFTS` | `mtp.drafts` |
| `HWNPU_IRON_DRAFTS` | `iron.drafts` |
| `HWNPU_ROTATE` | `rotate` |
| `HWNPU_LOAD_AHEAD` | `load-ahead` |
| `HWNPU_HOLD_DIT` | `hold-dit` |
| `HWNPU_CPU_F16` | `cpu.f16` |
| `HWNPU_KLEIN_DECODE` | `klein.decode` |
| `HWNPU_IRON_PREFETCH` | `iron.prefetch` |
| `HWNPU_IRON_SYNC` | `iron.sync` |
| `HWNPU_IRON_BARRIER` | `iron.barrier` |
| `HWNPU_IRON_TIMING` | `iron.timing` |
| `HWNPU_COREAI_DEBUG` | `coreai.debug` |
| `HWNPU_COREML_TRACE` | `coreml.trace` |
| `HWNPU_AILOWER_CUT` | `ailower.cut` |
| `HWNPU_AILOWER_DIVIDE` | `ailower.divide` |
| `HWNPU_AILOWER_NOTES` | `ailower.notes` |
| `HWNPU_AILOWER_TRACE` | `ailower.trace` |
| `HWNPU_OV_DEVICE` | `ov.device` |
| `HWNPU_OV_PRECISION` | `ov.precision` |
| `HWNPU_ATTENTION` | `attention` |
| `HWNPU_COREAI_COPY` | `coreai.copy` |
| `HWNPU_MOE_DEBUG=ITEM,...` | `moe.ITEM,...` (`1`: `moe.time`) |
| `HWNPU_MXFP4=ITEM,...` | `mxfp4.ITEM,...` |

## The tests' variables

`HWNPU_TEST_<GROUP>_<ROLE>`. The group is the model or the part (`LFM3`, `SFX`, `MRT`, `IRON`,
`OCR`, ...); the role says what the value is:

| Role | The value |
|---|---|
| `_DIR` | A model's directory of its checkpoint (Hugging Face's layout); `_CKPT_DIR` a derived checkpoint |
| `_EXPORT` | A converted export or package |
| `_FILE` | One input file: a model file (ONNX, TFLite, ELF), a tokenizer, a picture, a sound |
| `_REF` | A reference written by `tools/ref` (Python); `_TOKENS_REF` a tokenizer's, `_EMBED_REF` an embedding model's, `_ENCODER_REF` an encoder's |
| `_OUT` | Where a test writes what it made |
| anything else | A parameter of the test (`_FORMAT`, `_QUANT`, `_LAYERS`, `_PROMPT`, ...) |

- **`HWNPU_TEST_NPU`** (set: on) runs the tests that use the NPU, which the variables of their
  inputs no longer turn on alone. They wait for the Neural Engine's lock and may convert, so
  `go test ./...` without it never touches the NPU.
- **`-short`** skips the tests that are only slow on the CPU (`TestPagedCheckpoint`).
- A test that needs a model's files skips, saying which variables it needs, when they are not
  set.
- Tests run in their own process and never through hwnpud, with the Neural Engine's lock as
  before; `HWNPU_CACHE` is a temporary root unless set (`internal/cache`).

The names before (2026-11-22):

| Before | Now |
|---|---|
| `HWNPU_AIE_ELF` | `HWNPU_TEST_AIE_FILE` |
| `HWNPU_ANIMA_DIR` | `HWNPU_TEST_ANIMA_DIR` |
| `HWNPU_ANIMA_DIT` | `HWNPU_TEST_ANIMA_DIT` |
| `HWNPU_ANIMA_EMPTY_COND` | `HWNPU_TEST_ANIMA_NEGATIVE_REF` |
| `HWNPU_ANIMA_EXPORT` | `HWNPU_TEST_ANIMA_EXPORT` |
| `HWNPU_ANIMA_PROBE` | `HWNPU_TEST_ANIMA_PROBE` |
| `HWNPU_ANIMA_PROBE_COREML` | `HWNPU_TEST_ANIMA_PROBE_COREML` |
| `HWNPU_ANIMA_PROBE_FORMAT` | `HWNPU_TEST_ANIMA_PROBE_FORMAT` |
| `HWNPU_ANIMA_PROBE_OUT` | `HWNPU_TEST_ANIMA_PROBE_OUT` |
| `HWNPU_ANIMA_PROMPTS` | `HWNPU_TEST_ANIMA_PROMPTS` |
| `HWNPU_ANIMA_RANGES` | `HWNPU_TEST_ANIMA_RANGES_OUT` |
| `HWNPU_ANIMA_STEPS` | `HWNPU_TEST_ANIMA_STEPS` |
| `HWNPU_ATTN_PROBE` | `HWNPU_TEST_ATTN_PROBE` |
| `HWNPU_ATTN_PROBE_OUT` | `HWNPU_TEST_ATTN_PROBE_OUT` |
| `HWNPU_ATTN_PROBE_SHAPE` | `HWNPU_TEST_ATTN_PROBE_SHAPE` |
| `HWNPU_BASIC_PITCH` | `HWNPU_TEST_BASICPITCH_FILE` |
| `HWNPU_BASIC_PITCH_REF` | `HWNPU_TEST_BASICPITCH_REF` |
| `HWNPU_BASIC_PITCH_DEVICE` | `HWNPU_TEST_NPU` |
| `HWNPU_BERT_DIR` | `HWNPU_TEST_BERT_DIR` |
| `HWNPU_CALL_PKGS` | `HWNPU_TEST_CHAT_CALL_PKGS` |
| `HWNPU_COPYBENCH` | `HWNPU_TEST_COPYBENCH_FILE` |
| `HWNPU_COREAI_DEVICE` | `HWNPU_TEST_NPU` |
| `HWNPU_DETECT` | `HWNPU_TEST_DETECT_FILE` |
| `HWNPU_DETECT_AT` | `HWNPU_TEST_DETECT_AT` |
| `HWNPU_DETECT_EXP` | `HWNPU_TEST_DETECT_EXP` |
| `HWNPU_DETECT_NPU` | `HWNPU_TEST_NPU` |
| `HWNPU_DETECT_REF` | `HWNPU_TEST_DETECT_REF` |
| `HWNPU_GEMMA4_DIR` | `HWNPU_TEST_GEMMA4_DIR` |
| `HWNPU_GEMMA4_GOLDEN` | `HWNPU_TEST_GEMMA4_TOKENS_REF` |
| `HWNPU_GPTOSS_CKPT` | `HWNPU_TEST_GPTOSS_CKPT_DIR` |
| `HWNPU_GPTOSS_DIR` | `HWNPU_TEST_GPTOSS_DIR` |
| `HWNPU_GPTOSS_GAP` | `HWNPU_TEST_GPTOSS_GAP` |
| `HWNPU_GPTOSS_GOLDEN` | `HWNPU_TEST_GPTOSS_TOKENS_REF` |
| `HWNPU_GPTOSS_LAYERS` | `HWNPU_TEST_GPTOSS_LAYERS` |
| `HWNPU_GPTOSS_ROWS` | `HWNPU_TEST_GPTOSS_ROWS` |
| `HWNPU_INTERRUPT_HELPER` | `HWNPU_TEST_INTERRUPT_HELPER` |
| `HWNPU_IRON_AUDIO` | `HWNPU_TEST_IRON_AUDIO_EXPORT` |
| `HWNPU_IRON_AUDIO_CHECK` | `HWNPU_TEST_IRON_AUDIO_CHECK` |
| `HWNPU_IRON_EXPORT` | `HWNPU_TEST_IRON_EXPORT` |
| `HWNPU_IRON_KVRING` | `HWNPU_TEST_IRON_KVRING_EXPORT` |
| `HWNPU_IRON_RECURRENT` | `HWNPU_TEST_IRON_RECURRENT_EXPORT` |
| `HWNPU_IRON_VL` | `HWNPU_TEST_IRON_VL_EXPORT` |
| `HWNPU_IRON_VL_CHECK` | `HWNPU_TEST_IRON_VL_CHECK` |
| `HWNPU_IR_OUT` | `HWNPU_TEST_PROBE_IR_OUT` |
| `HWNPU_KLEIN_EXPORT` | `HWNPU_TEST_KLEIN_EXPORT` |
| `HWNPU_KLEIN_F32` | `HWNPU_TEST_KLEIN_F32` |
| `HWNPU_KLEIN_FORMAT` | `HWNPU_TEST_KLEIN_FORMAT` |
| `HWNPU_KLEIN_OUT` | `HWNPU_TEST_KLEIN_OUT` |
| `HWNPU_KLEIN_REPO` | `HWNPU_TEST_KLEIN_DIR` |
| `HWNPU_LFM3_COLBLOCK` | `HWNPU_TEST_LFM3_COLBLOCK` |
| `HWNPU_LFM3_CONV` | `HWNPU_TEST_LFM3_CONV` |
| `HWNPU_LFM3_DIR` | `HWNPU_TEST_LFM3_DIR` |
| `HWNPU_LFM3_FORMAT` | `HWNPU_TEST_LFM3_FORMAT` |
| `HWNPU_LFM3_FORMATS` | `HWNPU_TEST_LFM3_FORMATS` |
| `HWNPU_LFM3_GROUP` | `HWNPU_TEST_LFM3_GROUP` |
| `HWNPU_LFM3_KEEP` | `HWNPU_TEST_LFM3_KEEP` |
| `HWNPU_LFM3_LAYERS` | `HWNPU_TEST_LFM3_LAYERS` |
| `HWNPU_LFM3_PAGED_CPU` | not with `-short` |
| `HWNPU_LFM3_PERLAYER` | `HWNPU_TEST_LFM3_PERLAYER` |
| `HWNPU_LFM3_POSTSCALE` | `HWNPU_TEST_LFM3_POSTSCALE` |
| `HWNPU_LFM3_QUANT` | `HWNPU_TEST_LFM3_QUANT` |
| `HWNPU_LFM3_WEIGHTS` | `HWNPU_TEST_LFM3_WEIGHTS_FILE` |
| `HWNPU_MOSS_VAE` | `HWNPU_TEST_MOSS_VAE_FILE` |
| `HWNPU_MRT` | `HWNPU_TEST_MRT_EXPORT` |
| `HWNPU_MRT_COREAI` | `HWNPU_TEST_NPU` |
| `HWNPU_MRT_DEVICE` | `HWNPU_TEST_NPU` (the format: `HWNPU_TEST_MRT_FORMAT`) |
| `HWNPU_MRT_DIR` | `HWNPU_TEST_MRT_DIR` |
| `HWNPU_MRT_ENC_REF` | `HWNPU_TEST_MRT_ENCODER_REF` |
| `HWNPU_MRT_F16` | `HWNPU_TEST_MRT_F16` |
| `HWNPU_MRT_REF` | `HWNPU_TEST_MRT_REF` |
| `HWNPU_MRT_TRACE` | `HWNPU_TEST_MRT_TRACE` |
| `HWNPU_MRT_UNITS` | `HWNPU_TEST_MRT_UNITS` |
| `HWNPU_MUSICCOCA` | `HWNPU_TEST_MUSICCOCA_DIR` |
| `HWNPU_MUSICCOCA_AUDIO_REF` | `HWNPU_TEST_MUSICCOCA_AUDIO_REF` |
| `HWNPU_MUSICCOCA_REF` | `HWNPU_TEST_MUSICCOCA_REF` |
| `HWNPU_OCR_DET` | `HWNPU_TEST_OCR_DET_FILE` |
| `HWNPU_OCR_DET_REF` | `HWNPU_TEST_OCR_DET_REF` |
| `HWNPU_OCR_DIR` | `HWNPU_TEST_OCR_DIR` |
| `HWNPU_OCR_NPU` | `HWNPU_TEST_NPU` |
| `HWNPU_OCR_PICTURE` | `HWNPU_TEST_OCR_PICTURE_FILE` |
| `HWNPU_OCR_REC` | `HWNPU_TEST_OCR_REC_FILE` |
| `HWNPU_OCR_REC_REF` | `HWNPU_TEST_OCR_REC_REF` |
| `HWNPU_OCR_WANT` | `HWNPU_TEST_OCR_WANT` |
| `HWNPU_PLAMO3_DIR` | `HWNPU_TEST_PLAMO3_DIR` |
| `HWNPU_PLAMO3_GOLDEN` | `HWNPU_TEST_PLAMO3_TOKENS_REF` |
| `HWNPU_PLAMO_DIR` | `HWNPU_TEST_PLAMO_DIR` |
| `HWNPU_PLAMO_F16` | `HWNPU_TEST_PLAMO_F16` |
| `HWNPU_PLAMO_FORMAT` | `HWNPU_TEST_PLAMO_FORMAT` |
| `HWNPU_PLAMO_GOLDEN` | `HWNPU_TEST_PLAMO_EMBED_REF` |
| `HWNPU_PLAMO_HEADSCALE` | `HWNPU_TEST_PLAMO_HEADSCALE` |
| `HWNPU_QWEN35_DIR` | `HWNPU_TEST_QWEN35_DIR` |
| `HWNPU_QWEN36_CKPT` | `HWNPU_TEST_QWEN36_CKPT_DIR` |
| `HWNPU_QWEN36_DIR` | `HWNPU_TEST_QWEN36_DIR` |
| `HWNPU_QWEN36_GAP` | `HWNPU_TEST_QWEN36_GAP` |
| `HWNPU_QWEN36_OUT` | `HWNPU_TEST_QWEN36_OUT` |
| `HWNPU_QWEN36_SKEW` | `HWNPU_TEST_QWEN36_SKEW` |
| `HWNPU_QWEN36_SPIN` | `HWNPU_TEST_QWEN36_SPIN` |
| `HWNPU_RESAMPY` | `HWNPU_TEST_RESAMPY_REF` |
| `HWNPU_RTMDET_DIR` | `HWNPU_TEST_RTMDET_DIR` |
| `HWNPU_RTMO` | `HWNPU_TEST_RTMO_FILE` |
| `HWNPU_RTMO_REF` | `HWNPU_TEST_RTMO_REF` |
| `HWNPU_S3_TEST_BUCKET` | `HWNPU_TEST_S3_BUCKET` |
| `HWNPU_S3_TEST_ENDPOINT` | `HWNPU_TEST_S3_ENDPOINT` |
| `HWNPU_S3_TEST_REGION` | `HWNPU_TEST_S3_REGION` |
| `HWNPU_SFX_AMD` | `HWNPU_TEST_SFX_AMD_EXPORT` |
| `HWNPU_SFX_CONTEXT` | `HWNPU_TEST_SFX_CONTEXT` |
| `HWNPU_SFX_COREAI` | `HWNPU_TEST_NPU` |
| `HWNPU_SFX_CPU` | `HWNPU_TEST_SFX_CPU` |
| `HWNPU_SFX_CPU_DAC` | `HWNPU_TEST_SFX_CPU_DAC_DIR` |
| `HWNPU_SFX_DEVICE` | `HWNPU_TEST_NPU` |
| `HWNPU_SFX_DIR` | `HWNPU_TEST_SFX_DIR` |
| `HWNPU_SFX_DIT_FORMAT` | `HWNPU_TEST_SFX_DIT_FORMAT` |
| `HWNPU_SFX_FORMAT_FOR` | `HWNPU_TEST_SFX_FORMAT_FOR` |
| `HWNPU_SFX_HIDDEN` | `HWNPU_TEST_SFX_HIDDEN` |
| `HWNPU_SFX_LATENTS` | `HWNPU_TEST_SFX_LATENTS` |
| `HWNPU_SFX_ONLY` | `HWNPU_TEST_SFX_ONLY` |
| `HWNPU_SFX_PROMPT` | `HWNPU_TEST_SFX_PROMPT` |
| `HWNPU_SFX_QUANT` | `HWNPU_TEST_SFX_QUANT` |
| `HWNPU_SFX_REF` | `HWNPU_TEST_SFX_REF` |
| `HWNPU_SFX_STAGES` | `HWNPU_TEST_SFX_STAGES` |
| `HWNPU_SFX_TEXT_FORMAT` | `HWNPU_TEST_SFX_TEXT_FORMAT` |
| `HWNPU_SFX_WAV` | `HWNPU_TEST_SFX_WAV_OUT` |
| `HWNPU_SFX_WRITE` | `HWNPU_TEST_SFX_OUT` |
| `HWNPU_SFX_X` | `HWNPU_TEST_SFX_X` |
| `HWNPU_SIGLIP_EXPORT` | `HWNPU_TEST_SIGLIP_EXPORT` |
| `HWNPU_STYLE_WAV` | `HWNPU_TEST_MUSICCOCA_WAV_FILE` |
| `HWNPU_SWIFTF0` | `HWNPU_TEST_SWIFTF0_FILE` |
| `HWNPU_SWIFTF0_REF` | `HWNPU_TEST_SWIFTF0_REF` |
| `HWNPU_T5_TOKENIZER` | `HWNPU_TEST_T5_TOKENIZER_FILE` |
| `HWNPU_TOKENIZERS` | `HWNPU_TEST_TOKENIZERS` |
| `HWNPU_VARIANT` | `HWNPU_TEST_MLPACKAGE_VARIANT` |
| `HWNPU_WHISPER_DIR` | `HWNPU_TEST_WHISPER_DIR` |
| `HWNPU_WHISPER_ENCODERS` | `HWNPU_TEST_WHISPER_ENCODERS` |
| `HWNPU_WHISPER_GOLDEN` | `HWNPU_TEST_WHISPER_TOKENS_REF` |
| `HWNPU_WHISPER_LAYER` | `HWNPU_TEST_WHISPER_LAYER` |
| `HWNPU_WHISPER_REF` | `HWNPU_TEST_WHISPER_REF` |
| `HWNPU_WHISPER_UNITS` | `HWNPU_TEST_WHISPER_UNITS` |
