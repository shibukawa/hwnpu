package console

import (
	"io"
	"strings"
	"testing"
	"time"
)

func resetPhases() {
	phases.mu.Lock()
	defer phases.mu.Unlock()
	phases.since, phases.running, phases.loading, phases.loads, phases.ran = time.Now(), false, 0, nil, false
	phases.wall, phases.active, phases.dl = [2]time.Duration{}, 0, [2]time.Duration{}
}

func near(t *testing.T, what string, got, want time.Duration) {
	t.Helper()
	if got < want-15*time.Millisecond || got > want+40*time.Millisecond {
		t.Errorf("%s %v, want about %v", what, got, want)
	}
}

func TestPhases(t *testing.T) {
	resetPhases()
	const u = 60 * time.Millisecond
	time.Sleep(u) // preparing
	stop := Downloading("a")
	inner := Downloading("b") // at the same time: counted once
	time.Sleep(u)
	inner(1)
	time.Sleep(u)
	stop(2)
	stop(3) // twice: once
	Running()
	time.Sleep(u)
	done := Loading("outer")
	nested := Loading("inner")
	time.Sleep(u)
	nested()
	time.Sleep(u)
	done()
	time.Sleep(u)
	stop = Downloading("c") // while running
	time.Sleep(u)
	stop(-1)
	dl, prep, run := PhaseTimes()
	near(t, "downloading", dl, 3*u)
	near(t, "preparing", prep, 3*u)
	near(t, "running", run, 2*u)
}

func TestTimes(t *testing.T) {
	resetPhases()
	var b strings.Builder
	Times(&b)
	if b.Len() != 0 {
		t.Errorf("before Running: %q", b.String())
	}
	Running()
	Times(&b)
	if got := b.String(); !strings.HasPrefix(got, "took ") || strings.Contains(got, "downloading") || !strings.Contains(got, "running") {
		t.Errorf("Times: %q", got)
	}
}

func TestProfile(t *testing.T) {
	resetPhases()
	defer func(on bool) { Profile = on }(Profile)
	Profile = true
	prof.records, prof.writers = nil, nil
	Downloading("owner/x/model.safetensors")(2_000_000)
	Downloading("owner/x/linked")(-1) // no download: not listed
	outer := Loading("prog.parts.json")
	Loading("prog.part1.mlpackage")()
	Loading("prog.parts.json")() // the same inside: listed once
	outer()
	OnProfile(func(w io.Writer) { io.WriteString(w, "profile q=8\n") })
	var b strings.Builder
	WriteProfile(&b)
	got := b.String()
	for _, want := range []string{"download", "2.0 MB", "owner/x/model.safetensors", "load", "s    prog.part1.mlpackage", "prog.parts.json", "profile q=8"} {
		if !strings.Contains(got, want) {
			t.Errorf("no %q in\n%s", want, got)
		}
	}
	if strings.Count(got, "prog.parts.json") != 1 {
		t.Errorf("a load inside a load of the same is listed:\n%s", got)
	}
	if strings.Contains(got, "linked") {
		t.Errorf("a file linked, not downloaded, is listed:\n%s", got)
	}
	if i, j := strings.Index(got, "part1"), strings.Index(got, "parts.json"); i < j {
		t.Errorf("the inner load is before the one it is in:\n%s", got)
	}
	Profile = false
	b.Reset()
	WriteProfile(&b)
	if b.Len() != 0 {
		t.Errorf("without HWNPU_PROFILE: %q", b.String())
	}
}
