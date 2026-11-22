//go:build !darwin

package chat

// hubArch is "" on a machine that is not a Mac: no Neural Engine.
func hubArch() string { return "" }
