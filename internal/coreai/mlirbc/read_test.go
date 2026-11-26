package mlirbc

import "testing"

// The use-list orders of an operation's results or a block's arguments (the Neural Engine
// compiler's files have them, Core AI's own none) are read past, whatever their form: one value
// (no count, no index) or several, a full shuffle or pairs of indices.
func TestSkipUseListOrders(t *testing.T) {
	v := func(n uint64) byte { return byte(n<<1 | 1) } // a varint below 128
	flagged := func(n uint64, flag bool) byte {
		if flag {
			return v(n<<1 | 1)
		}
		return v(n << 1)
	}
	for _, c := range []struct {
		name   string
		values int
		data   []byte
	}{
		{"one value, a shuffle of three uses", 1, []byte{flagged(3, false), v(2), v(0), v(1)}},
		{"one value, a pair", 1, []byte{flagged(2, true), v(0), v(1)}},
		{"two of three values", 3, []byte{v(2), v(0), flagged(2, false), v(1), v(0), v(2), flagged(2, true), v(0), v(3)}},
	} {
		r := &reader{b: append(c.data, 0x55)} // a byte of what follows
		if err := skipUseListOrders(r, c.values); err != nil {
			t.Errorf("%s: %v", c.name, err)
		} else if r.off != len(c.data) {
			t.Errorf("%s: read %d of %d bytes", c.name, r.off, len(c.data))
		}
	}
	if err := skipUseListOrders(&reader{b: []byte{flagged(3, false), v(0)}}, 1); err == nil {
		t.Error("orders cut short were read")
	}
}
