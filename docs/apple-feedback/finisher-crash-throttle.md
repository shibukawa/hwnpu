# Each crash of ANELargeModelCompilerService lengthens launchd's throttle on it: every later .aimodelc takes 80 s to 20 min to finish, until reboot

- **Area**: Core AI / Neural Engine (aned, ANELargeModelCompilerService), launchd
- **Type**: Performance

## Description

A `.aimodelc` is finished on the device by `ANELargeModelCompilerService`, which exits after a compile; launchd then defers its next start (FB25624129: `service spawn deferred by 10 seconds due to throttle`). The deferral is the rest of an interval since the service's last start, and that interval grows when the service crashes:

| time (one M1, one boot) | what happened | interval |
|---|---|---|
| 16:56–17:50 | 17 small programs finished twice; the service crashed 5 times on one of them (heap corruption in `ZinKernelBuilder::BuildKernelFromLLIRData`) | 10 s |
| 18:02 | another crash (`mlir::Operation::~Operation`), the service started twice within 2 s | |
| 22:42 (4.5 idle hours later) | first finish of the evening | 80 s |
| 23:17, 23:18 | two more crashes | 320 s |
| 23:24 | one more crash | 640 s |
| 23:54 | one more crash | 1200 s |

`log show --predicate 'process == "launchd" AND eventMessage CONTAINS "deferred by"'` shows `deferred by 75`, `315`, `585`, `1200 seconds`. The start right after a crash is not deferred; the next one waits the whole new interval. The interval did not go back in 4.5 hours without any use.

Finishing is one program at a time for the whole machine, so after a few crashes every app's `.aimodelc` takes minutes each to load: a chat model of 17 programs took 23 minutes at 80 s, and one program takes 20 minutes at 1200 s. Compiling the same programs on the device takes 7 minutes in all. The crashes themselves are bugs of the finishing service (reported separately); the cost of one such bug is paid by every later load on the machine.

## Steps to reproduce

1. Open a `.aimodelc` that crashes ANELargeModelCompilerService (attached: `model.part1.h13g.aimodelc` with its `.aimodel`, two functions; it crashes at task 1013 of 1155 of the second) with `AIModel(contentsOf:)` in a process with no Core AI entry for it. Core AI reports `failedToSpecialize`.
2. Repeat a few times (each attempt crashes the service twice).
3. Open any other `.aimodelc` twice in a row, and read launchd's log.

## Expected results

The time to finish an `.aimodelc` does not depend on earlier crashes of the finishing service for other programs; at least it recovers without a reboot.

## Actual results

`service spawn deferred by 1200 seconds due to throttle`: 20 minutes per `.aimodelc`, for every process, until the machine restarts.

## Environment

M1 16 GB (h13g), macOS 27.2.1 (26B131), coreai-build 3605.6.3 (Metal Toolchain 27.2.47.0).

## Attachments

- launchd's log lines of the day (redacted)
- the crash reports of ANELargeModelCompilerService
- `model.part1.aimodel` and its h13g `.aimodelc`
