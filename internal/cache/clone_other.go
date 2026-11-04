//go:build !darwin

package cache

const canClone = false

func clonefile(src, dst string) bool { return false }
