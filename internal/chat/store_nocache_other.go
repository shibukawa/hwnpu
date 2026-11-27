//go:build !darwin && !linux

package chat

import "os"

func setNoCache(*os.File) {}

func dropPageCache(*os.File, bool) {}
