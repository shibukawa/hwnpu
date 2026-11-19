//go:build !linux && !darwin

package sysload

func readProcs([]int) map[int]procTimes { return nil }
