//go:build !darwin

package catalog

// cpuBrand is the machine's CPU as macOS names it; "" elsewhere, where no Neural Engine limits
// the programs held.
func cpuBrand() string { return "" }
