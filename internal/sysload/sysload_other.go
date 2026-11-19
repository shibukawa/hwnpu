//go:build !linux && !darwin

package sysload

func readCPU() (cpuTicks, bool)       { return cpuTicks{}, false }
func readMemory() (used, total int64) { return 0, 0 }
func readGPUs() []GPU                 { return nil }
