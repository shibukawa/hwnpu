//go:build !darwin

package daemon

import "net"

// peerUID is not read yet off the Mac (Linux's SO_PEERCRED comes with its service); the socket's
// mode keeps other users out.
func peerUID(net.Conn) (int, bool) { return 0, false }
