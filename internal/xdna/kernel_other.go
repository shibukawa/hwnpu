//go:build !linux

package xdna

import "errors"

// OpenNode is only available on Linux.
func OpenNode(path string) (Driver, error) { return nil, errors.New("xdna: amdxdna is Linux only") }
