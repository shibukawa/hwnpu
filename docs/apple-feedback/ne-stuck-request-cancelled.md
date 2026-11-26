# One request stalls the Neural Engine for 100 s and ends in "Request cancelled"; after it, every request from every process does the same for half an hour or more

- **Area**: Core AI / Apple Neural Engine (aned) on M2
- **Type**: Hang / Incorrect behavior

## Description

Running one function of a program on the Neural Engine (Anima's VAE stage 0 at 752×1360: a 1×1 and a 3×3 convolution, then residual blocks of 3×3 convolutions, RMS norm and SiLU, and one attention over the pixels; compiled on the device, no prebuilt) stalls for about 100 s and fails:

```
InferenceError(location: unknown, underlyingError: MetalPerformanceShadersGraph.MPSGraphDelegateError.ndxRuntimeError("inferValue failed: processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:: ANEProgramProcessRequestDirect() Failed with status=0xf : statusType=0x9: Request cancelled"))
```

After that, **every** Neural Engine request fails the same way after about 100 s: other functions of the same program that ran a moment before (`s1024x1024`, 0.24 s), other programs, other processes, and programs from other binaries (prebuilt or compiled on the device). Nothing else used the Neural Engine. It comes back by itself only after a long idle: 5–10 minutes with no request did not bring it back; 35 minutes with no request (06:19 to 06:53) did, as did about two hours (04:11 to 06:14). Each failing request seems to restart the wait.

Reproduced after the Neural Engine had come back by itself (2026-11-27, MacBook Air M2, no restart in between):

| time | run | result |
|---|---|---|
| 06:14:27 | `vae_s0#s1024x1024` | right, 0.24 s (twice) |
| 06:16:14 | `vae_s0#s752x1360` | "Request cancelled" after 100 s |
| 06:17:54 | `vae_s0#s1024x1024` | "Request cancelled" after 100 s |
| 06:53:26 | `vae_s0#s1024x1024` (no request since 06:19) | right, 0.25 s |

The same stage's `s512x512`, `s1024x1024`, `s832x1248` and `s880x1168` functions run (0.2–0.5 s). `s1248x832`, `s1168x880` and `s1360x752` were only tried after the Neural Engine was stuck, so it is not known whether each also stalls it.

Seen before with other programs: a 128-row function of a text encoder stalled the Neural Engine for 1–2 minutes, failed with "Request cancelled" and left it failing for every process until it was idle; 64 and 320 rows of the same build worked. A prebuilt T5-XXL first block failed with "Request cancelled" four times running on an M1.

## Steps to reproduce

1. With the Neural Engine idle, run `vae_s0.aimodel` (attached), function `s1024x1024`, on the Neural Engine with a random fp16 input (1×16×128×128): it returns in about 0.25 s.
2. Run function `s752x1360` (input 1×16×170×94).
3. Run function `s1024x1024` again, from the same or another process.

## Expected results

Step 2 returns a result, or fails at once with an error that names the cause. Either way, step 3 works as in step 1: a failed request does not affect later requests of other programs or processes.

## Actual results

Step 2 fails with "Request cancelled" after about 100 s. Step 3, and every later Neural Engine request on the machine, fails the same way after about 100 s, until the Neural Engine has had no request for a long time (35 minutes was enough once, 5–10 minutes was not).

## Environment

MacBook Air M2 16 GB (h14g), macOS 27.2.1 (26B131), Xcode 27, Metal Toolchain 27.2.47.0.

## Attachments

- `vae_s0.aimodel` (18 MB, 8 size functions)
- a sysdiagnose taken while the Neural Engine is stuck (after step 3)
- `log show --info --predicate 'process == "aned" OR process == "ANECompilerService" OR process == "ANELargeModelCompilerService" OR sender == "H11ANEIn"'` from step 1 to step 3
