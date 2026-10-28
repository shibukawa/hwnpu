package ze

import "runtime"

func runtimeKeepAlive(v any) { runtime.KeepAlive(v) }
