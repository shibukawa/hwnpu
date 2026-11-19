//go:build unix

package experts

import "golang.org/x/sys/unix"

func mlock(b []byte) error { return unix.Mlock(b) }

func munlock(b []byte) { unix.Munlock(b) }
