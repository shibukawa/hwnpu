//go:build !darwin

package mem

// Resident is how many of b's bytes are in memory now; not told here.
func Resident(b []byte) (bytes int64, ok bool) { return 0, false }
