package types

import (
	"errors"
	"fmt"
	"strings"
)

// Sentinel errors shared by the backends and re-exported by the public package.
var (
	ErrUnsupported = errors.New("hwnpu: not supported by this device/backend")
	ErrLibrary     = errors.New("hwnpu: runtime library not found")
)

// ErrNotOnNPU is matched by PlacementError.
var ErrNotOnNPU = errors.New("hwnpu: model does not run entirely on the NPU")

// PlacementError reports a graph that does not run entirely on the NPU, either because strict
// mode rejected it or because the NPU compiler does not support some operations.
// errors.Is(err, ErrNotOnNPU) is true.
type PlacementError struct {
	Report PlacementReport
	Cause  error // e.g. the compiler error, when the NPU could not compile the graph at all
}

func (e *PlacementError) Error() string {
	var msg string
	if !e.Report.Verified && len(e.Report.Ops) == 0 {
		msg = fmt.Sprintf("%v: placement could not be verified: %s", ErrNotOnNPU, e.Report.Reason)
	} else {
		ops := make([]string, 0, len(e.Report.Ops))
		for _, op := range e.Report.Ops {
			ops = append(ops, fmt.Sprintf("%s(%s)->%s", op.Type, op.Name, op.Device))
		}
		msg = fmt.Sprintf("%v: %d operation(s) placed elsewhere: %s", ErrNotOnNPU, len(e.Report.Ops), strings.Join(ops, ", "))
	}
	if e.Cause != nil {
		msg += ": " + e.Cause.Error()
	}
	return msg
}

func (e *PlacementError) Is(target error) bool { return target == ErrNotOnNPU }
func (e *PlacementError) Unwrap() error        { return e.Cause }

// DriverError is a failed call into a vendor runtime.
type DriverError struct {
	Backend string
	Op      string
	Code    int64
	Msg     string
}

func (e *DriverError) Error() string {
	return fmt.Sprintf("hwnpu %s: %s: %s (code %#x)", e.Backend, e.Op, e.Msg, e.Code)
}

// IsOutOfMemory reports whether err is a driver running out of the memory a graph is loaded into
// (Level Zero's ERROR_OUT_OF_HOST_MEMORY or ERROR_OUT_OF_DEVICE_MEMORY, or Intel's driver compiler
// failing with C++'s std::bad_alloc, "bad allocation", when the graphs already loaded hold the
// memory): one that may load once other graphs are closed.
func IsOutOfMemory(err error) bool {
	if err == nil {
		return false
	}
	var d *DriverError
	if errors.As(err, &d) && (d.Code == 0x70000002 || d.Code == 0x70000003) {
		return true
	}
	msg := strings.ToLower(err.Error())
	for _, s := range []string{"out_of_host_memory", "out_of_device_memory", "bad allocation", "bad_alloc"} {
		if strings.Contains(msg, s) {
			return true
		}
	}
	return false
}

// IsDeviceLost reports whether err is the NPU lost by the driver (Level Zero's DEVICE_LOST), as
// Windows does to a job that runs past its time limit.
func IsDeviceLost(err error) bool {
	var d *DriverError
	if errors.As(err, &d) {
		return d.Code == 0x70000001
	}
	return err != nil && strings.Contains(err.Error(), "DEVICE_LOST")
}

// BuildError is a failed graph compilation with the compiler log.
type BuildError struct {
	Log string
	Err error
}

func (e *BuildError) Error() string {
	if e.Log == "" {
		return fmt.Sprintf("hwnpu: build failed: %v", e.Err)
	}
	return fmt.Sprintf("hwnpu: build failed: %v\n%s", e.Err, e.Log)
}

func (e *BuildError) Unwrap() error { return e.Err }

// VersionError reports a runtime component that is too old or too new.
type VersionError struct {
	Component  string
	Have, Need string
}

func (e *VersionError) Error() string {
	return fmt.Sprintf("hwnpu: %s version %s is not supported (need %s)", e.Component, e.Have, e.Need)
}
