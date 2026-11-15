//go:build !unix && !windows

package cache

import (
	"errors"
	"io/fs"
)

func diskSpace(string) (int64, int64, error) { return 0, 0, errors.ErrUnsupported }

type fileKey struct{}

func keyOf(fs.FileInfo) (fileKey, bool) { return fileKey{}, false }
