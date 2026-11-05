//go:build !darwin && !linux && !windows

package cache

import "os"

const canClone = false

func clonefile(src, dst string) bool { return false }

func sharedPrefix(a, b string, limit int64) int64 { return 0 }

func rangeBlock(f *os.File) int64 { return 0 }

func cloneRange(src, dst *os.File, off, n int64) bool { return false }
