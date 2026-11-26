# coreai-build h13g: an activation kernel shared between tasks is sized for the last one, and tasks with more coefficient sets read past it; the M1 hangs the Neural Engine at the first call, or ANELargeModelCompilerService crashes

- **Area**: Xcode / coreai-build (h13g), Core AI (ANELargeModelCompilerService, Neural Engine firmware) on M1
- **Type**: Crash / Incorrect results

## Description

`coreai-build compile --architecture h13g --preferred-compute neural-engine` writes LLIR in which several activation tasks of a function load one kernel (the same `memref.view` into the weights): the two steps of every sigmoid and of `x / (1 + exp(-x))`, the tanh of every gelu. The view has the size the **last** of these tasks needs. A task over a tensor laid out with rows or heads as channels reads one set of coefficients a channel (8 or 16 sets of 128 bytes); a task over a tensor laid out as the output of a linear layer reads one set. When the last task is of the second kind the view is 128 bytes, and the earlier tasks' `anehlo.coeff_dma_offset_size` operations still name (0, 128), (128, 128), … (896, 128): they read 1024 or 2048 bytes of a kernel of 128.

The smallest program found (16 tasks, 620 KB, attached as `share.aimodel`):

```
h = linear(x)                               x (1, 8, 256), h (1, 8, 1024)
g = reshape(sigmoid(reshape(h, (1, 8, 16, 64))), (1, 8, 1024))
u = linear2(x);  s = u / (1 + exp(-u))
y = g * s
```

Its tasks 6 and 7 (the sigmoid's two steps, 8 coefficient sets each) load the kernels of the later `exp` and inverse of `s` (one set each), 128 bytes. With the sigmoid applied before the reshape (`share-flat.aimodel`, the same values) every task has one set and nothing reads past a kernel. On the M1, `share.aimodel` from its `.aimodelc` was finished and its first call was given up after 100 s (2026-11-27, `checkForTimedOutRequests: Request timeout detected … elapsed: 100005 ms, timeout: 100000 ms`); `share-flat.aimodel` from its `.aimodelc` ran (0.39 ms a call).

On an M1 (h13g) such a program is fully placed on the Neural Engine offline and then:

- **one function**: ANELargeModelCompilerService finishes the `.aimodelc`, and the first call never ends. The kernel logs `AppleH11ANEInterface … SyncRequest timed out after 100000 ms`, the caller gets `ANEProgramProcessRequestDirect() … Request cancelled` after 100 s, `CSNE_CMD_STOP` times out too, and the driver cold-restarts the Neural Engine once its last client is gone. Until then **every process's** Neural Engine requests wait their 100 s and fail (a daemon that keeps asking, `mediaanalysisd` one night, keeps it from restarting).
- **two or more functions**: ANELargeModelCompilerService crashes with heap corruption shortly after the last such task (a chat model's part with 1155 tasks: the tasks that read past their kernel are 103, 294, 295, 448, 639, 640, 793, 984, 985, and it crashed at task 1013 in four runs of four; `ZinKernelBuilder::BuildKernelFromLLIRData`, `ZinAneHloConverter::ParseInstructions` or MLIR teardown on the stack). Core AI reports `failedToSpecialize`. Each crash also lengthens launchd's throttle on the service (reported separately).

The same `.aimodel` compiled on the M1 itself runs and is correct. The h14g, h15g, h16g and h17g LLIR of the same programs has no task that reads past its kernel.

How well the reading of the LLIR matches the device: of the h13g programs tried on the M1, every one that hung or crashed the service has such tasks, and none of the about 20 that ran has one; and the chat models rewritten so that their activations are laid out alike (no such task in any of 202 h13g programs of nine models) run from their `.aimodelc` (a part with two functions of each of three models tried on the M1, four parts in all). FLUX.2 klein's h13g program of an attention over 8704 keys (noise, or a crash of the service; reported as "h13g .aimodelc with attention over 8192 or more keys") has 1296 such tasks.

## Steps to reproduce

1. `xcrun coreai-build compile share.aimodel --platform macOS --architecture h13g --preferred-compute neural-engine --output out/`
2. Read `out/share.h13g.aimodelc/main-h13g-delegates/MPSGraph/mpsExecutable.mpsgraphpackage/binary_0.llir.bundle/*/h13g/*.mlir.bc`: tasks 6 and 7 (`anehlo.ne` with `anehlo.kload`) load a `memref.view` of type `memref<128 x i8>` and hold 8 `anehlo.coeff_dma_offset_size` with offsets 0 to 896.
3. On an M1, in a process with no Core AI entry for the program, open `share.aimodel` beside its `.aimodelc` with `AIModel(contentsOf:)` and call `main` once.

## Expected results

The shared kernel is as large as the largest reader needs (as it is when a reader with 8 sets happens to be the last: a chat model's layer cut before its last activation has a view of 1024 bytes, and runs; with that activation, of one set, the view is 128 bytes and the first call hangs), or the tasks do not share it. ANELargeModelCompilerService rejects LLIR whose coefficient offsets lie past the kernel instead of reading there, and a program that cannot run fails its request without taking the Neural Engine from other processes for 100 s.

## Actual results

The first call does not end; the Neural Engine is restarted after 100 s. With two functions, ANELargeModelCompilerService crashes.

## Environment

M1 16 GB (h13g), macOS 27.2.1 (26B131), coreai-build 3605.6.3 (Metal Toolchain 27.2.47.0).

## Attachments

- `share.aimodel` and `share-flat.aimodel` with their h13g `.aimodelc`
- the list of tasks of both (kernel view sizes and coefficient offsets)
- the kernel's log of the 100 s request and the restart; a crash report of ANELargeModelCompilerService
- a chat model's part (`model.part1.aimodel`, two functions) before and after the change of layout, with their `.aimodelc`

## Second problem seen on the way (smaller)

A matrix product with a constant of ones on the **left** (`ones (rows, 1)^T × e`, a sum over rows) is fully placed by coreai-build for h13g, and the M1 cannot finish the `.aimodelc`: `Kernel evaluation failed: … mps.constant_… : memref<1x1x1x2x32x1xf16>`, `failedToSpecialize`. With the ones on the right (`e^T × ones`) it finishes and runs. No task reads past a kernel in this program.
