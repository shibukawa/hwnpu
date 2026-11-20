//go:build arm64

package mxfp4

import "syscall"

func i8mm() bool {
	v, err := syscall.SysctlUint32("hw.optional.arm.FEAT_I8MM")
	return err == nil && v == 1
}
