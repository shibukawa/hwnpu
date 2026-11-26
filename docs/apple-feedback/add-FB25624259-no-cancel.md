# Addition to FB25624259: a compile goes on after its client exits, and the next process waits behind it

Seen 2026-11-25 on a MacBook Air M2 16 GB, macOS 27.2.1 (26B131).

When the process that asked for a Neural Engine compile exits (killed, or stopped by a timeout) while ANECompilerService is compiling its program, the compile is not cancelled. ANECompilerService keeps compiling at 100 % of a core until it ends, and because compiles run one at a time for the whole machine, the next process's first compile waits behind a program nobody will load:

- an image model's process was stopped while a 1024×1024 VAE stage compiled; the next process (another model) waited about 20 minutes before its first compile started.
- a process was stopped while a 4-block image-model block compiled; the next process's first program, a small text encoder that compiles in 11 s, took 14 min 56 s.

The app cannot cancel it either: the compile belongs to aned's connection to ANECompilerService, not to the app's process.

Expected: when the client of a compile disconnects, and no other client waits for the same program, the compile is cancelled; or the queue serves a waiting client ahead of a compile nobody waits for.
