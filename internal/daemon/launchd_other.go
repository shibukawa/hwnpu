//go:build !darwin && !linux

package daemon

import "net"

// activated is the service manager's socket activation (launchd's, systemd's), which this system does not have.
func activated(string) ([]net.Listener, error) { return nil, nil }
