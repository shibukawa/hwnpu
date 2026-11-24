//go:build unix

package chain

import "golang.org/x/sys/unix"

// mlock wires b in RAM (BONSAIAI_MOE_WIRE), and munlock lets it go.
func mlock(b []byte) error { return unix.Mlock(b) }

func munlock(b []byte) error { return unix.Munlock(b) }
