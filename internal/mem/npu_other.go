//go:build !darwin

package mem

// NPUWired is the kernel's wired memory of Apple's Neural Engine driver; not known here.
func NPUWired() (bytes int64, ok bool) { return 0, false }
