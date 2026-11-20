package mem

// Pressure is how short of memory the system is: Free the percent of the memory a program can
// still have (macOS' kern.memorystatus_level, what memory_pressure prints as "free percentage":
// free memory and the cache the system can drop; Linux's MemAvailable of MemTotal) and Level the
// system's own verdict (1 normal, 2 warning, 4 critical: kern.memorystatus_vm_pressure_level; on
// Linux from Free). ok is false where neither is read.
type Pressure struct {
	Free  int
	Level int
}

// PressureNow reads the Pressure now.
func PressureNow() (Pressure, bool) { return pressureNow() }
