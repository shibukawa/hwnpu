//go:build !darwin && !linux

package daemon

import "net"

// peerUID is not read on this system; the socket's
// mode keeps other users out.
func peerUID(net.Conn) (int, bool) { return 0, false }
