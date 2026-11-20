//go:build unix

package experts

import "golang.org/x/sys/unix"

func mlock(b []byte) error { return unix.Mlock(b) }

func munlock(b []byte) { unix.Munlock(b) }

// willneed asks the kernel to read b's pages ahead.
func willneed(b []byte) { unix.Madvise(b, unix.MADV_WILLNEED) }
