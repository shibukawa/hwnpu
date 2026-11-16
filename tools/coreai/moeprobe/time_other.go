//go:build !darwin

package main

import (
	"errors"
	"time"
)

// timeGroups loads programs on the Neural Engine: there is none here. Writing them works on any OS.
func timeGroups([]string, int, int, time.Duration) error {
	return errors.New("moeprobe: -k needs Core AI (macOS 27 on Apple silicon); writing the programs runs anywhere")
}
