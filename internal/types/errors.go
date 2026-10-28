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
