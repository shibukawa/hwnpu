package mem

import (
	"bytes"
	"os"
	"runtime"
	"strconv"
)

// Available is the memory a program can still take without the system swapping, in bytes: Linux's
// MemAvailable (free memory and the page cache it can drop), elsewhere half the machine's memory
// (macOS keeps little free and counts the cache apart); 0 when not known.
func Available() int64 {
	if runtime.GOOS == "linux" {
		if b, err := os.ReadFile("/proc/meminfo"); err == nil {
			for _, line := range bytes.Split(b, []byte("\n")) {
				if rest, ok := bytes.CutPrefix(line, []byte("MemAvailable:")); ok {
					if f := bytes.Fields(rest); len(f) > 0 {
						if n, err := strconv.ParseInt(string(f[0]), 10, 64); err == nil {
							return n << 10
						}
					}
				}
			}
		}
	}
	return Physical() / 2
}
