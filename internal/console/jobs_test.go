package console

import (
	"bytes"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// TestJobsNotLive writes a line for each job as it ends, numbered in the order they end, and runs
// them side by side.
func TestJobsNotLive(t *testing.T) {
	var b bytes.Buffer
	j := NewJobs(&b)
	var at, most atomic.Int32
	for i := range 6 {
		j.Add(fmt.Sprintf("p%d", i), func() (string, error) {
			n := at.Add(1)
			for {
				m := most.Load()
				if n <= m || most.CompareAndSwap(m, n) {
					break
				}
			}
			time.Sleep(20 * time.Millisecond)
			at.Add(-1)
			return "int8", nil
		})
	}
	if err := j.run(3); err != nil {
		t.Fatal(err)
	}
	if most.Load() != 3 {
		t.Errorf("%d at a time, want 3", most.Load())
	}
	lines := strings.Split(strings.TrimSuffix(b.String(), "\n"), "\n")
	if len(lines) != 6 {
		t.Fatalf("%d lines: %q", len(lines), b.String())
	}
	for i, l := range lines {
		if !strings.HasPrefix(l, fmt.Sprintf("  [%d/6] p", i+1)) || !strings.HasSuffix(l, ", int8") {
			t.Errorf("line %d: %q", i, l)
		}
	}
	if j.Len() != 0 {
		t.Errorf("%d jobs left on the list", j.Len())
	}
}

// TestJobsError stops the jobs not started at the first error and returns it.
func TestJobsError(t *testing.T) {
	var b bytes.Buffer
	j := NewJobs(&b)
	bad := errors.New("bad weights")
	var ran atomic.Int32
	for i := range 10 {
		j.Add(fmt.Sprintf("p%d", i), func() (string, error) {
			ran.Add(1)
			if i == 1 {
				return "", bad
			}
			time.Sleep(10 * time.Millisecond)
			return "", nil
		})
	}
	if err := j.run(2); !errors.Is(err, bad) {
		t.Fatalf("%v, want %v", err, bad)
	}
	if n := ran.Load(); n >= 10 {
		t.Errorf("%d jobs ran after the error", n)
	}
	if !strings.Contains(b.String(), "  p1 failed after ") {
		t.Errorf("no failure line in %q", b.String())
	}
}

// TestJobsLive draws the bar and the jobs running, and erases them all at the end.
func TestJobsLive(t *testing.T) {
	var b liveBuffer
	j := NewJobs(&b)
	for i := range 4 {
		j.Add(fmt.Sprintf("dit_blocks_%d.aimodel", 4*i), func() (string, error) {
			time.Sleep(250 * time.Millisecond)
			return "int8", nil
		})
	}
	if err := j.run(2); err != nil {
		t.Fatal(err)
	}
	s := b.String()
	for _, want := range []string{"▕░░░░░░░░░░░░░░░░░░░░░░░░▏ 0/4", "2 at a time", "    dit_blocks_0.aimodel", "    dit_blocks_4.aimodel", "\x1b[2A"} {
		if !strings.Contains(s, want) {
			t.Errorf("no %q in %q", want, s)
		}
	}
	if !strings.HasSuffix(s, "\r\x1b[2A\x1b[J") && !strings.HasSuffix(s, "\r\x1b[J") {
		t.Errorf("the lines are not erased at the end: %q", s[max(0, len(s)-40):])
	}
	if strings.Contains(s, "int8") {
		t.Errorf("a job's line was left: %q", s)
	}
}
