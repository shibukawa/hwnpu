package daemon

import (
	"fmt"
	"io"
	"net"
	"os"
)

// sameUser is a listener that closes the connections of other users' processes (the socket's
// mode 0600 already keeps them out; this holds when a socket's mode is not ours to set).
type sameUser struct {
	net.Listener
	log io.Writer
}

func (l sameUser) Accept() (net.Conn, error) {
	for {
		c, err := l.Listener.Accept()
		if err != nil {
			return nil, err
		}
		if uid, ok := peerUID(c); ok && uid != os.Getuid() {
			fmt.Fprintf(l.log, "bonsaiaid: refused a connection of user %d\n", uid)
			c.Close()
			continue
		}
		return c, nil
	}
}
