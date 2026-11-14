//go:build !linux && !darwin && !windows

package npuinfo

import "context"

func collect(ctx context.Context, opts Options, info *Info) error { return ErrNotImplemented }
