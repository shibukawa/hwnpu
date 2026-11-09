//go:build !unix

package llm

import "os"

func tryLock(*os.File) (bool, error) { return true, nil }
