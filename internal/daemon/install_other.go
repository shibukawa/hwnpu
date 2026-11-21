//go:build !darwin

package daemon

import (
	"errors"
	"io"
)

var errService = errors.New("hwnpud install: only the Mac's LaunchAgent so far (systemd's user units come with Linux; docs/design/hwnpud.md); run hwnpud serve by hand")

// Install is the Mac's for now.
func Install(string, io.Writer) error { return errService }

// Uninstall is the Mac's for now.
func Uninstall(io.Writer) error { return errService }

// Installed is the Mac's for now.
func Installed() string { return "" }

// PlistPath is the Mac's for now.
func PlistPath() string { return "" }
