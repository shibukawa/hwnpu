//go:build !darwin

package main

import "errors"

// errDarwin is what probe and run give where there is no Core AI: they load programs on the
// Neural Engine. ops, list, lower and merge work on any OS.
var errDarwin = errors.New("aicheck: probe and run need Core AI (macOS 27 on Apple silicon); ops, list, lower and merge run anywhere")

func probe([]string, bool) error { return errDarwin }

func run([]string, string, float64, bool, bool, int, string) error { return errDarwin }
