//go:build !unix && !windows

package openvino

import "errors"

// supported is false: the shared buffers are mapped with Unix or Windows calls.
const supported = false

func shmDir() string { return "" }

func mapFile(path string, size int) ([]byte, error) {
	return nil, errors.New("openvino: not supported on this platform")
}

func unmap(b []byte) error { return nil }

func processAlive(pid int) bool { return true }
