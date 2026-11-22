package daemon

import (
	"net"

	"golang.org/x/sys/unix"
)

// peerUID is the user of the process at the other end of a Unix domain socket's connection
// (SO_PEERCRED).
func peerUID(c net.Conn) (int, bool) {
	uc, ok := c.(*net.UnixConn)
	if !ok {
		return 0, false
	}
	raw, err := uc.SyscallConn()
	if err != nil {
		return 0, false
	}
	uid, found := 0, false
	raw.Control(func(fd uintptr) {
		cred, err := unix.GetsockoptUcred(int(fd), unix.SOL_SOCKET, unix.SO_PEERCRED)
		if err == nil {
			uid, found = int(cred.Uid), true
		}
	})
	return uid, found
}
