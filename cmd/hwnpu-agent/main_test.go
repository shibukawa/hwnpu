package main

import (
	"bufio"
	"strings"
	"testing"
)

func TestPreview(t *testing.T) {
	if got := preview("a\nb\n", 5); got != "a\nb" {
		t.Errorf("short: %q", got)
	}
	if got := preview("1\n2\n3\n4\n5\n6\n7\n", 5); got != "1\n2\n3\n4\n5\n… 2 more lines" {
		t.Errorf("long: %q", got)
	}
}

func TestRead(t *testing.T) {
	in := bufio.NewReader(strings.NewReader("one \\\n two\n\nlast"))
	if got, ok := read(in); !ok || got != "one \n two" {
		t.Errorf("continued: %q, %v", got, ok)
	}
	if got, ok := read(in); !ok || got != "" {
		t.Errorf("empty: %q, %v", got, ok)
	}
	if got, ok := read(in); !ok || got != "last" {
		t.Errorf("no newline: %q, %v", got, ok)
	}
	if _, ok := read(in); ok {
		t.Error("read past the end")
	}
}
