//go:build headless

package main

import "errors"

// runWindow is the window's place in a build without one (-tags headless), for a machine
// without a display or without the libraries Ebitengine is built with: cameras, models and
// status work.
func runWindow(windowOptions) error {
	return errors.New("this backdrop was built without its window (-tags headless): build it without the tag")
}
