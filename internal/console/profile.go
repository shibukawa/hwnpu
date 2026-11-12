package console

import (
	"fmt"
	"io"
	"os"
	"slices"
	"sync"
	"time"
)

// Profile is on when HWNPU_PROFILE is set: the logs keep their details (the files of the
// programs), and the command's last lines list each download and each program loaded with its
// time (WriteProfile), and what the packages that keep a profile of their own add (OnProfile:
// the steps of a language model, internal/llm/decode).
var Profile = os.Getenv("HWNPU_PROFILE") != ""

const (
	kindDownload = "download"
	kindLoad     = "load"
	kindLookup   = "lookup"
)

// Looked records, for the profile, the sizes of n files looked up before their downloads in d.
func Looked(n int, d time.Duration) {
	record(kindLookup, fmt.Sprintf("the sizes of %d files", n), 0, d, -1)
}

type profRecord struct {
	kind, what string
	depth      int // a load inside another
	start      time.Time
	d          time.Duration
	bytes      int64
}

var prof struct {
	mu      sync.Mutex
	records []profRecord
	writers []func(io.Writer)
}

func record(kind, what string, depth int, d time.Duration, bytes int64) {
	if !Profile {
		return
	}
	prof.mu.Lock()
	prof.records = append(prof.records, profRecord{kind, what, depth, time.Now().Add(-d), d, bytes})
	prof.mu.Unlock()
}

// OnProfile adds f to what WriteProfile writes.
func OnProfile(f func(io.Writer)) {
	prof.mu.Lock()
	prof.writers = append(prof.writers, f)
	prof.mu.Unlock()
}

// WriteProfile writes, when HWNPU_PROFILE is set, the downloads (bytes, time, speed) and the
// programs loaded (time; a load inside another indented under it) in the order they started,
// and what OnProfile added.
func WriteProfile(w io.Writer) {
	if !Profile || w == nil {
		return
	}
	prof.mu.Lock()
	records, writers := slices.Clone(prof.records), prof.writers
	prof.mu.Unlock()
	slices.SortStableFunc(records, func(a, b profRecord) int { return a.start.Compare(b.start) })
	var dl, load time.Duration
	var bytes int64
	for _, r := range records {
		switch r.kind {
		case kindDownload:
			dl += r.d
			bytes += r.bytes
			fmt.Fprintf(w, "profile: download %9.3f s  %s  %.1f MB/s  %s\n", r.d.Seconds(), gb(r.bytes), float64(r.bytes)/max(r.d.Seconds(), 1e-9)/1e6, r.what)
		case kindLookup:
			fmt.Fprintf(w, "profile: lookup   %9.3f s  %s\n", r.d.Seconds(), r.what)
		case kindLoad:
			if r.depth == 0 {
				load += r.d
			}
			fmt.Fprintf(w, "profile: load     %9.3f s  %*s%s\n", r.d.Seconds(), 2*r.depth, "", r.what)
		}
	}
	switch {
	case bytes > 0 && load > 0:
		fmt.Fprintf(w, "profile: %s downloaded in %s, programs loaded in %s (outermost loads)\n", gb(bytes), Duration(dl), Duration(load))
	case bytes > 0:
		fmt.Fprintf(w, "profile: %s downloaded in %s\n", gb(bytes), Duration(dl))
	case load > 0:
		fmt.Fprintf(w, "profile: programs loaded in %s (outermost loads)\n", Duration(load))
	}
	reads.mu.Lock()
	if reads.n > 0 {
		fmt.Fprintf(w, "profile: weights read again %d times, %s in %s (groups read into one bank in turn)\n", reads.n, gb(reads.bytes), Duration(reads.d))
	}
	reads.mu.Unlock()
	for _, f := range writers {
		f(w)
	}
}

// gb is n bytes for a person.
func gb(n int64) string {
	switch {
	case n < 1e6:
		return fmt.Sprintf("%.1f kB", float64(n)/1e3)
	case n < 1e9:
		return fmt.Sprintf("%.1f MB", float64(n)/1e6)
	}
	return fmt.Sprintf("%.2f GB", float64(n)/1e9)
}
