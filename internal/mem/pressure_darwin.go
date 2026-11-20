package mem

import "syscall"

func pressureNow() (Pressure, bool) {
	free, err := syscall.SysctlUint32("kern.memorystatus_level")
	if err != nil {
		return Pressure{}, false
	}
	p := Pressure{Free: int(free), Level: 1}
	if l, err := syscall.SysctlUint32("kern.memorystatus_vm_pressure_level"); err == nil && l > 0 {
		p.Level = int(l)
	}
	return p, true
}
