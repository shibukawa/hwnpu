package airpc

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"time"
)

// Handler serves sessions over HTTP connections upgraded to protocol proto (GET with
// Connection: Upgrade and Upgrade: proto, answered 101): each is a session whose root is
// root(s). It returns when the session ends, so that a server counting its requests sees the
// session as one going on (hwnpud's engine does not exit idle while an app uses it).
func Handler(proto string, root func(s *Session) any) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.EqualFold(r.Header.Get("Upgrade"), proto) {
			http.Error(w, "airpc: this route takes an upgrade to "+proto, http.StatusUpgradeRequired)
			return
		}
		conn, rw, err := http.NewResponseController(w).Hijack()
		if err != nil {
			http.Error(w, "airpc: "+err.Error(), http.StatusInternalServerError)
			return
		}
		defer conn.Close()
		fmt.Fprintf(rw, "HTTP/1.1 101 Switching Protocols\r\nConnection: Upgrade\r\nUpgrade: %s\r\n\r\n", proto)
		if err := rw.Flush(); err != nil {
			return
		}
		Serve(conn, rw.Reader, root)
	})
}

// Dial upgrades conn (a new connection to the server) to protocol proto at path and returns
// the client end of the session over it. A server that answers otherwise is an error with its
// answer.
func Dial(ctx context.Context, conn net.Conn, path, proto string) (*Session, error) {
	if d, ok := ctx.Deadline(); ok {
		conn.SetDeadline(d)
	}
	stop := context.AfterFunc(ctx, func() { conn.Close() })
	defer stop()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://hwnpud"+path, nil)
	if err != nil {
		conn.Close()
		return nil, err
	}
	req.Header.Set("Connection", "Upgrade")
	req.Header.Set("Upgrade", proto)
	if err := req.Write(conn); err != nil {
		conn.Close()
		return nil, err
	}
	br := bufio.NewReader(conn)
	resp, err := http.ReadResponse(br, req)
	if err != nil {
		conn.Close()
		return nil, err
	}
	if resp.StatusCode != http.StatusSwitchingProtocols {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
		resp.Body.Close()
		conn.Close()
		return nil, fmt.Errorf("airpc: %s at %s: %s %s", proto, path, resp.Status, strings.TrimSpace(string(b)))
	}
	if !stop() {
		conn.Close()
		return nil, ctx.Err()
	}
	conn.SetDeadline(time.Time{})
	return NewClient(conn, br), nil
}
