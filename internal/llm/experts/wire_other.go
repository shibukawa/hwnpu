//go:build !unix

package experts

import "errors"

func mlock(b []byte) error { return errors.New("not supported here") }

func munlock(b []byte) {}

func willneed([]byte) {}

func incore([]byte) bool { return true }
