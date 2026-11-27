//go:build darwin

package chat

import (
	"os"

	"golang.org/x/sys/unix"
)

func setNoCache(f *os.File) {
	_, _ = unix.FcntlInt(f.Fd(), unix.F_NOCACHE, 1)
}

func dropPageCache(*os.File, bool) {}
