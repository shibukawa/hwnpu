package main

import (
	"fmt"
	"slices"
	"strings"
	"sync"
	"time"
)

// Timers are the timers set: each rings once, at its time (Ring is called in a goroutine of its
// own).
type Timers struct {
	mu   sync.Mutex
	next int
	set  []*Timer
	// Ring is called when a timer's time has come (nil: nothing); Added when one is set.
	Ring, Added func(t *Timer)
	// now is the clock (time.Now; a test's).
	now func() time.Time
}

// Timer is one timer.
type Timer struct {
	ID    int
	Label string // "" for none
	For   time.Duration
	At    time.Time
	t     *time.Timer
}

func NewTimers() *Timers { return &Timers{now: time.Now} }

// Set sets a timer of d, with a label (or "").
func (ts *Timers) Set(d time.Duration, label string) *Timer {
	ts.mu.Lock()
	defer ts.mu.Unlock()
	ts.next++
	t := &Timer{ID: ts.next, Label: strings.TrimSpace(label), For: d, At: ts.now().Add(d)}
	t.t = time.AfterFunc(d, func() {
		ts.mu.Lock()
		i := slices.Index(ts.set, t)
		if i >= 0 {
			ts.set = slices.Delete(ts.set, i, i+1)
		}
		ring := ts.Ring
		ts.mu.Unlock()
		if i >= 0 && ring != nil {
			ring(t)
		}
	})
	ts.set = append(ts.set, t)
	if ts.Added != nil {
		ts.Added(t)
	}
	return t
}

// Cancel stops the timers of a label ("": all of them) and returns them.
func (ts *Timers) Cancel(label string) []*Timer {
	ts.mu.Lock()
	defer ts.mu.Unlock()
	label = strings.TrimSpace(label)
	var out []*Timer
	ts.set = slices.DeleteFunc(ts.set, func(t *Timer) bool {
		if label != "" && (t.Label == "" || !strings.Contains(t.Label, label) && !strings.Contains(label, t.Label)) {
			return false
		}
		t.t.Stop()
		out = append(out, t)
		return true
	})
	return out
}

// List is the timers that have not rung, the soonest first.
func (ts *Timers) List() []*Timer {
	ts.mu.Lock()
	defer ts.mu.Unlock()
	out := slices.Clone(ts.set)
	slices.SortFunc(out, func(a, b *Timer) int { return a.At.Compare(b.At) })
	return out
}

// Left is how long until the timer rings.
func (ts *Timers) Left(t *Timer) time.Duration { return max(0, t.At.Sub(ts.now())) }

// Describe says what the timers are, for the model and for a spoken answer: "ラーメンのタイマー
// があと2分30秒".
func (ts *Timers) Describe() string {
	list := ts.List()
	if len(list) == 0 {
		return "タイマーはありません。"
	}
	var parts []string
	for _, t := range list {
		parts = append(parts, fmt.Sprintf("%sがあと%s", t.Name(), spokenDuration(ts.Left(t).Truncate(time.Second))))
	}
	return strings.Join(parts, "、") + "です。"
}

// Name is how the timer is spoken of: "ラーメンのタイマー", "3分のタイマー".
func (t *Timer) Name() string {
	if t.Label != "" {
		return t.Label + "のタイマー"
	}
	return spokenDuration(t.For) + "のタイマー"
}

// spokenDuration is a duration as it is said: "1時間30分", "2分30秒", "45秒".
func spokenDuration(d time.Duration) string {
	d = d.Truncate(time.Second)
	h, m, s := int(d/time.Hour), int(d%time.Hour/time.Minute), int(d%time.Minute/time.Second)
	var b strings.Builder
	if h > 0 {
		fmt.Fprintf(&b, "%d時間", h)
	}
	if m > 0 {
		fmt.Fprintf(&b, "%d分", m)
	}
	if s > 0 || b.Len() == 0 {
		fmt.Fprintf(&b, "%d秒", s)
	}
	return b.String()
}
