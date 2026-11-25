# Addition to FB25624129: on an M2, loading from prebuilt is 3 to 5 times slower than compiling on the device for language and audio models

Measured 2026-11-25 on a MacBook Air M2 16 GB (h14g), macOS 27.2.1 (26B131), coreai-build 3605.6.3. For each model, the first load with empty Core AI and aned caches (a copy of the binary with a new code-signing identifier, Core AI's entries removed), once compiling on the device and once from `.aimodelc` files coreai-build made for h14g for every program. Nothing else used the Neural Engine.

| model | programs | compile on the device | from prebuilt | disk peak (device / prebuilt) |
|---|---|---|---|---|
| plamo-3-translate-0.8b (language model) | 17 | 46 s | 209 s | 3.9 / 1.7 GB |
| whisper-large-v3-turbo (speech recognition) | 3 | 78 s | 253 s | 2.2 / 3.0 GB |
| LFM3-1.7B (language model) | 19 | 158 s | 453 s | 8.3 / 6.3 GB |
| plamo-3-translate-8b (language model) | 12 | 78 s | 393 s | 7.0 / 10.7 GB |
| Qwen3.5-0.8B (language model) | 25 | 541 s | not measured | 3.6 GB / — |

The answers and transcriptions were the same both ways.

The 10-second spawn interval of this report explains the difference for plamo-3-translate-0.8b (17 programs, 163 s more), but not for the others: whisper is 175 s slower with 3 programs, plamo-3-translate-8b 315 s with 12. Finishing each program itself takes longer than compiling it. For example, whisper's 3 programs loaded in 75.8 s compiled on the device and in 250.7 s from prebuilt. plamo-3-translate-0.8b's first program took 1 min 25 s from prebuilt.

For comparison, the image-model blocks of the same day finished in 41–61 s each from prebuilt against 21–23 min compiled on the device. So prebuilt wins only for programs that take minutes to compile, and loses for the many smaller programs of language and audio models. We keep shipping prebuilt programs only for image models.
