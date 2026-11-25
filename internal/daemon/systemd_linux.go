package daemon

import (
	"fmt"
	"net"
	"os"
	"strconv"
	"strings"
	"syscall"
)

// listenFDsStart is the first descriptor systemd passes (SD_LISTEN_FDS_START).
const listenFDsStart = 3

// activated returns the listeners systemd passed this process (LISTEN_FDS, LISTEN_PID,
// LISTEN_FDNAMES: bonsaiaid.socket's FileDescriptorName is the socket name), nil when systemd did
// not start it with any. The variables are taken out of the environment, so that the engine
// does not see them as its own.
func activated(name string) ([]net.Listener, error) {
	pid, fds, names := os.Getenv("LISTEN_PID"), os.Getenv("LISTEN_FDS"), os.Getenv("LISTEN_FDNAMES")
	for _, k := range activationEnv {
		os.Unsetenv(k)
	}
	if pid == "" || fds == "" || pid != strconv.Itoa(os.Getpid()) {
		return nil, nil
	}
	n, err := strconv.Atoi(fds)
	if err != nil || n < 0 {
		return nil, fmt.Errorf("LISTEN_FDS=%q", fds)
	}
	var named []string
	if names != "" {
		named = strings.Split(names, ":")
	}
	var out []net.Listener
	for i := range n {
		fd := listenFDsStart + i
		syscall.CloseOnExec(fd) // not the engine's
		if i < len(named) && named[i] != name && named[i] != "unknown" {
			syscall.Close(fd)
			continue
		}
		f := os.NewFile(uintptr(fd), name)
		ln, err := net.FileListener(f)
		f.Close() // FileListener dups it
		if err != nil {
			for _, l := range out {
				l.Close()
			}
			return nil, err
		}
		out = append(out, ln)
	}
	return out, nil
}
