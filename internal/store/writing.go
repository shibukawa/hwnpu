package store

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path"
	"strings"
	"sync"
	"time"
)

// A writeback marks the folder it writes in (an entry's, converted/MODEL/TARGET/EXPORT/VERSION, or
// a repository's, original/OWNER/NAME) with a file .writing-ID while it writes, refreshed every
// WritingRefresh (docs/cache-sharing-design.md 7.1). A folder is whole or it is not; Clean removes what a
// writeback cut off left in one that is not, unless a marker in it was refreshed in the last
// WritingStale: a writeback going on there.

// WritingPrefix starts the name of a marker.
const WritingPrefix = ".writing-"

// WritingRefresh is how often a marker is written again; WritingStale is how long one not
// written again is taken as of a writeback that was cut off.
var (
	WritingRefresh = 5 * time.Minute
	WritingStale   = time.Hour
)

// writingMark is what a marker holds.
type writingMark struct {
	Host    string    `json:"host"`
	PID     int       `json:"pid"`
	Started time.Time `json:"started"`
	Seen    time.Time `json:"seen"` // when it was last written
}

// MarkWriting puts a marker in folder of s and writes it again every WritingRefresh until done
// is called, which removes it.
func MarkWriting(ctx context.Context, s Source, folder string) (done func(), err error) {
	var rnd [8]byte
	rand.Read(rnd[:])
	key := path.Join(folder, WritingPrefix+hex.EncodeToString(rnd[:]))
	host, _ := os.Hostname()
	m := writingMark{Host: host, PID: os.Getpid(), Started: time.Now().UTC()}
	write := func(ctx context.Context) error {
		m.Seen = time.Now().UTC()
		b, _ := json.Marshal(m)
		return s.Store.Write(ctx, key, append(b, '\n'))
	}
	if err := write(ctx); err != nil {
		return nil, err
	}
	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		t := time.NewTicker(WritingRefresh)
		defer t.Stop()
		for {
			select {
			case <-stop:
				return
			case <-ctx.Done():
				return
			case <-t.C:
				write(ctx) // one that fails is written at the next tick
			}
		}
	}()
	var once sync.Once
	return func() {
		once.Do(func() {
			close(stop)
			wg.Wait()
			s.Store.Delete(context.WithoutCancel(ctx), key)
		})
	}, nil
}

// isMarker reports whether key is a marker.
func isMarker(key string) bool { return strings.HasPrefix(path.Base(key), WritingPrefix) }

// isTemp reports whether key is a temporary file of an upload (.put-), moved into place when done.
func isTemp(key string) bool { return strings.HasPrefix(path.Base(key), ".put-") }

// writing reports whether one of the markers of s was written in the last WritingStale. One not
// there (removed since) or not whole is not; one that cannot be read is an error.
func writing(ctx context.Context, s Source, markers []string, now time.Time) (bool, error) {
	for _, k := range markers {
		b, err := s.Store.Read(ctx, k)
		if err != nil {
			if errors.Is(err, ErrNotFound) {
				continue
			}
			return false, err
		}
		var m writingMark
		if json.Unmarshal(b, &m) != nil {
			continue // not one bonsaiai wrote whole: Write moves it into place
		}
		if now.Sub(m.Seen) < WritingStale {
			return true, nil
		}
	}
	return false, nil
}
