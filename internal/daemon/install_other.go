//go:build !darwin && !linux

package daemon

import (
	"errors"
	"io"
)

var errService = errors.New("hwnpu service install: only the Mac's LaunchAgent and systemd's user units; run it in the foreground with hwnpu service start")

// Install is the Mac's and Linux's.
func Install(string, io.Writer) error { return errService }

// Uninstall is the Mac's and Linux's.
func Uninstall(io.Writer) error { return errService }

// Installed is the Mac's and Linux's.
func Installed() string { return "" }

// ServiceFile is the Mac's and Linux's.
func ServiceFile() string { return "" }

// LogWhere is the supervisor's standard error here.
func LogWhere() string { return "hwnpud serve's standard error" }

// ServiceEnviron is the Mac's and Linux's.
func ServiceEnviron() []string { return nil }
