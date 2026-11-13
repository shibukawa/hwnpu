package types

import (
	"errors"
	"fmt"
	"testing"
)

func TestIsOutOfMemory(t *testing.T) {
	for _, c := range []struct {
		err  error
		want bool
	}{
		{&DriverError{Backend: "levelzero", Op: "pfnGraphInitialize", Code: 0x70000002, Msg: "ERROR_OUT_OF_HOST_MEMORY"}, true},
		{fmt.Errorf("dit_blocks_20.xml: %w", &DriverError{Backend: "levelzero", Op: "pfnCreate", Code: 0x7ffffffe, Msg: "ERROR_UNKNOWN: bad allocation"}), true},
		{errors.New("dit_blocks_20.xml: compile: std::bad_alloc"), true},
		{errors.New("dit_blocks_20.xml: Bad Allocation"), true},
		{&DriverError{Backend: "levelzero", Op: "zeFenceHostSynchronize", Code: 0x70000001, Msg: "ERROR_DEVICE_LOST"}, false},
		{errors.New("no such file"), false},
		{nil, false},
	} {
		if got := IsOutOfMemory(c.err); got != c.want {
			t.Errorf("%v: %v, want %v", c.err, got, c.want)
		}
	}
}
