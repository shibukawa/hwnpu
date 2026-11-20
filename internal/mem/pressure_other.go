//go:build !darwin

package mem

func pressureNow() (Pressure, bool) {
	total, avail := Physical(), int64(0)
	if total <= 0 {
		return Pressure{}, false
	}
	if avail = Available(); avail <= 0 || avail == total/2 { // Available's guess, not a reading
		return Pressure{}, false
	}
	p := Pressure{Free: int(100 * avail / total), Level: 1}
	switch {
	case p.Free < 5:
		p.Level = 4
	case p.Free < 10:
		p.Level = 2
	}
	return p, true
}
