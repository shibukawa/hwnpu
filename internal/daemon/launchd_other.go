//go:build !darwin

package daemon

import "net"

// activated is launchd's socket activation, which only the Mac has (systemd's comes with Linux).
func activated(string) ([]net.Listener, error) { return nil, nil }
