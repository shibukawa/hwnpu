//go:build !unix

package chain

import "errors"

// mlock is not there off unix: a reserve of memory for the experts then wires nothing and says so.
func mlock([]byte) error { return errors.ErrUnsupported }

func munlock([]byte) error { return nil }
