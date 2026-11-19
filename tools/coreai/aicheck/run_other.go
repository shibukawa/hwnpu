//go:build !darwin

package main

import "errors"

// errDarwin is what the commands that load programs on the Neural Engine give where there is no
// Core AI (probe, run, cycle, moe and the gather and MoE measurements). ops, list, lower and
// merge work on any OS.
var errDarwin = errors.New("aicheck: this command needs Core AI (macOS 27 on Apple silicon); ops, list, lower and merge run anywhere")

func probe([]string, bool) error { return errDarwin }

func run([]string, string, float64, bool, bool, int, string) error { return errDarwin }

func cycle([]string, int) error { return errDarwin }

func moe([]string) error { return errDarwin }

func locked([]string) error { return errDarwin }

func gatherCheck([]string) error { return errDarwin }

func moeServe([]string) error { return errDarwin }

func fdivCheck() error { return errDarwin }

func mx([]string) error { return errDarwin }

func moe4([]string) error { return errDarwin }

func moePack([]string) error { return errDarwin }
