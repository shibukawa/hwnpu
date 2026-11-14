// Package npuinfo reads what the operating system tells about NPUs (neural processing units):
// the devices, which processes use them, how busy they are, and what the drivers and the
// firmware report.
//
// For now it gathers what each OS and driver gives, as it is, in OS-specific structs
// (Info.Linux, Info.Windows, Info.Darwin); a common API comes later, once it is known what
// every OS can give. It imports only the standard library, so that other modules (gopsutil,
// for example) can use it without new dependencies.
//
// Reading must not change what it reads: on Linux, amdxdna's queries power up a sleeping NPU
// (and keep it up for the driver's autosuspend delay, 5 s), so they run only while the NPU is
// already awake unless Options.Wake is set.
package npuinfo

import (
	"context"
	"encoding/json"
	"errors"
	"runtime"
	"time"
)

// ErrNotImplemented is returned on operating systems this package cannot read yet.
var ErrNotImplemented = errors.New("npuinfo: not implemented on this OS")

// Options changes what Get reads.
type Options struct {
	// Wake lets queries run that power up a sleeping NPU (amdxdna's ioctls on Linux). Without
	// it, a monitor that polls does not keep the NPU awake.
	Wake bool
	// SampleInterval is the time between the two samples that rate counters need (Windows'
	// performance counters). 0 means 500 ms.
	SampleInterval time.Duration
	// SkipCaches leaves out walking compile cache directories (macOS), which can be slow.
	SkipCaches bool
}

// Info is everything read at one time. Only the field of the running OS is set.
type Info struct {
	Time    time.Time    `json:"time"`
	OS      string       `json:"os"`
	Arch    string       `json:"arch"`
	Linux   *LinuxInfo   `json:"linux,omitempty"`
	Windows *WindowsInfo `json:"windows,omitempty"`
	Darwin  *DarwinInfo  `json:"darwin,omitempty"`
	// Errors lists what could not be read; the rest is still filled in.
	Errors []string `json:"errors,omitempty"`
}

func (i Info) String() string {
	b, _ := json.MarshalIndent(i, "", "  ")
	return string(b)
}

// Get reads with the default options.
func Get() (*Info, error) { return GetWithContext(context.Background(), Options{}) }

// GetWithContext reads what the running OS gives. It returns an error only when nothing could
// be read at all; partial failures go to Info.Errors.
func GetWithContext(ctx context.Context, opts Options) (*Info, error) {
	info := &Info{Time: time.Now(), OS: runtime.GOOS, Arch: runtime.GOARCH}
	err := collect(ctx, opts, info)
	return info, err
}

func (i *Info) addErr(err error) {
	if err != nil {
		i.Errors = append(i.Errors, err.Error())
	}
}

// guard runs f and turns a panic into an error, so one reader that goes wrong on an untested
// OS version does not lose the rest.
func guard(name string, info *Info, f func() error) {
	defer func() {
		if r := recover(); r != nil {
			info.Errors = append(info.Errors, name+": panic: "+panicString(r))
		}
	}()
	if err := f(); err != nil {
		info.Errors = append(info.Errors, name+": "+err.Error())
	}
}

func panicString(r any) string {
	switch v := r.(type) {
	case error:
		return v.Error()
	case string:
		return v
	}
	b, _ := json.Marshal(r)
	return string(b)
}
