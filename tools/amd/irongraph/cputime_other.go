//go:build !unix

package main

import "time"

// cpuTime is 0 where getrusage is not (the host's CPU time is not reported).
func cpuTime() time.Duration { return 0 }
