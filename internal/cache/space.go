package cache

import (
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// How much of the disk the cache takes (docs/cache-sharing.md 7.7). Every download into the
// root asks Room first:
//
//   - HWNPU_CACHE_MIN_FREE (10GB unless set; 0 for none) is what is left free on the root's disk:
//     a download that would leave less is not started.
//   - HWNPU_CACHE_MAX_SIZE (none unless set) is the most the cache holds: before a download that
//     would go over, the units least recently used are removed (LRU) until it fits, and when the
//     disk is short of HWNPU_CACHE_MIN_FREE too. Without it nothing is removed by itself (hwnpu
//     cache prune and clear remove).
//
// A size is bytes, or a number with KB, MB, GB, TB (powers of 1000) or KiB, MiB, GiB, TiB;
// HWNPU_CACHE_MIN_FREE may be a percentage of the disk too (5%).
const (
	MaxSizeEnv = "HWNPU_CACHE_MAX_SIZE"
	MinFreeEnv = "HWNPU_CACHE_MIN_FREE"
)

// DefaultMinFree is HWNPU_CACHE_MIN_FREE when it is not set.
const DefaultMinFree int64 = 10e9

// KeepRecent is how recently a unit was used for it not to be removed by itself or by hwnpu
// cache prune: another process may be reading it.
const KeepRecent = time.Hour

// Limits are what the cache may take of the disk.
type Limits struct {
	// MaxSize is the most the cache holds, 0 for no limit: units are removed by LRU only with one.
	MaxSize int64
	// MinFree is the bytes left free on the disk; MinFreePercent the part of the disk (the larger
	// of the two). Both 0: none.
	MinFree        int64
	MinFreePercent float64
}

// MinFreeOf is what l leaves free on a disk of total bytes.
func (l Limits) MinFreeOf(total int64) int64 {
	return max(l.MinFree, int64(l.MinFreePercent/100*float64(total)))
}

// String says l as the variables do.
func (l Limits) String() string {
	free := FormatSize(l.MinFree)
	if l.MinFreePercent > 0 {
		free = strconv.FormatFloat(l.MinFreePercent, 'f', -1, 64) + "%"
		if l.MinFree > 0 {
			free += " or " + FormatSize(l.MinFree)
		}
	}
	size := "no limit"
	if l.MaxSize > 0 {
		size = FormatSize(l.MaxSize)
	}
	return fmt.Sprintf("max size %s, min free %s", size, free)
}

var setLimits atomic.Pointer[Limits]

// SetLimits sets the limits of this process over the variables' (nil for the variables' again):
// an application's own (modelstore.Options).
func SetLimits(l *Limits) { setLimits.Store(l) }

// CurrentLimits are the limits in effect: SetLimits', else the variables'.
func CurrentLimits() (Limits, error) {
	if l := setLimits.Load(); l != nil {
		return *l, nil
	}
	l := Limits{MinFree: DefaultMinFree}
	if testing.Testing() {
		l.MinFree = 0 // go test downloads small files: a full disk of CI is not its matter
	}
	if v := strings.TrimSpace(os.Getenv(MinFreeEnv)); v != "" {
		l.MinFree = 0
		if p, ok := strings.CutSuffix(v, "%"); ok {
			f, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
			if err != nil || f < 0 || f >= 100 {
				return l, fmt.Errorf("%s=%s: not a percentage", MinFreeEnv, v)
			}
			l.MinFreePercent = f
		} else {
			n, err := ParseSize(v)
			if err != nil {
				return l, fmt.Errorf("%s: %w", MinFreeEnv, err)
			}
			l.MinFree = n
		}
	}
	if v := strings.TrimSpace(os.Getenv(MaxSizeEnv)); v != "" {
		n, err := ParseSize(v)
		if err != nil {
			return l, fmt.Errorf("%s: %w", MaxSizeEnv, err)
		}
		l.MaxSize = n
	}
	return l, nil
}

var sizeUnits = []struct {
	suffix string
	n      float64
}{
	{"kib", 1 << 10}, {"mib", 1 << 20}, {"gib", 1 << 30}, {"tib", 1 << 40},
	{"kb", 1e3}, {"mb", 1e6}, {"gb", 1e9}, {"tb", 1e12},
	{"k", 1e3}, {"m", 1e6}, {"g", 1e9}, {"t", 1e12}, {"b", 1},
}

// ParseSize reads a size: bytes, or a number with a unit (200GB, 1.5TiB, 500M).
func ParseSize(s string) (int64, error) {
	v := strings.ToLower(strings.TrimSpace(s))
	mul := 1.0
	for _, u := range sizeUnits {
		if p, ok := strings.CutSuffix(v, u.suffix); ok {
			v, mul = strings.TrimSpace(p), u.n
			break
		}
	}
	f, err := strconv.ParseFloat(v, 64)
	if err != nil || f < 0 || math.IsInf(f, 0) || f*mul > math.MaxInt64 {
		return 0, fmt.Errorf("%q is not a size (200GB, 1.5TiB, 0)", s)
	}
	return int64(f * mul), nil
}

// FormatSize says n bytes in GB, MB or KB (powers of 1000).
func FormatSize(n int64) string {
	switch a := math.Abs(float64(n)); {
	case a >= 1e12:
		return fmt.Sprintf("%.2f TB", float64(n)/1e12)
	case a >= 1e9:
		return fmt.Sprintf("%.1f GB", float64(n)/1e9)
	case a >= 1e6:
		return fmt.Sprintf("%.1f MB", float64(n)/1e6)
	case a >= 1e3:
		return fmt.Sprintf("%.1f KB", float64(n)/1e3)
	}
	return fmt.Sprintf("%d B", n)
}

// DiskSpace is the bytes free to this user and the size of the disk dir is on (the nearest
// folder that is there, when dir is not).
func DiskSpace(dir string) (free, total int64, err error) {
	d := dir
	for {
		if _, err := os.Stat(d); err == nil {
			break
		}
		p := filepath.Dir(d)
		if p == d {
			break
		}
		d = p
	}
	return diskSpace(d)
}

// ErrNoRoom is Room's error for a download the limits do not let in.
var ErrNoRoom = errors.New("not enough room for the cache")

var (
	roomMu sync.Mutex
	// scanned is what the root held when it was scanned for MaxSize (the time and bytes), and
	// added what Room let in since: the units are walked once a process, not at every file.
	scanned     *Usage
	scannedRoot string
	added       int64
)

// Room checks that need bytes more may go in the cache, for the file at path under the root
// (being written: its unit is never removed). With a MaxSize, it removes the units least
// recently used until the cache fits it with need more and the disk keeps its MinFree (units
// used in the last KeepRecent, in this process, or with work queued are kept). It returns an
// error (ErrNoRoom) when the disk would be left with less than MinFree, or the cache over
// MaxSize, even so. log gets what is removed (nil for nowhere).
func Room(need int64, path string, log func(format string, args ...any)) error {
	if need < 0 {
		need = 0
	}
	l, err := CurrentLimits()
	if err != nil {
		return err
	}
	if l.MaxSize <= 0 && l.MinFree <= 0 && l.MinFreePercent <= 0 {
		return nil
	}
	roomMu.Lock()
	defer roomMu.Unlock()
	root := Root()
	free, total, err := DiskSpace(root)
	if err != nil {
		return nil // a disk that does not say: no check
	}
	minFree := l.MinFreeOf(total)
	short := minFree - (free - need)
	var over int64
	if l.MaxSize > 0 {
		if scanned == nil || scannedRoot != root {
			u, err := Scan()
			if err != nil {
				return err
			}
			scanned, scannedRoot, added = &u, root, 0
		}
		over = scanned.Bytes + added + need - l.MaxSize
	}
	if short <= 0 && over <= 0 {
		added += need
		return nil
	}
	keep := map[string]bool{}
	if u, ok := UnitOf(path); ok {
		keep[u] = true
	}
	if short > 0 {
		// short of disk: what is made again soon goes first, with or without a MaxSize (Core AI's
		// entries of programs with prebuilt ones here, SetRestorable; docs/cache-sharing.md 7.7)
		if freed := evictRestorable(short, log); freed > 0 {
			if free, _, err = DiskSpace(root); err == nil {
				short = minFree - (free - need)
			}
		}
	}
	if (short > 0 || over > 0) && l.MaxSize > 0 {
		freed := evictLRU(scanned, max(short, over), keep, log)
		over -= freed
		if free, _, err = DiskSpace(root); err == nil {
			short = minFree - (free - need)
		}
	}
	what := filepath.Base(path)
	if need > 0 {
		what += " needs " + FormatSize(need) + ", and"
	} else {
		what += ":"
	}
	switch {
	case short > 0:
		return fmt.Errorf("%w: %s %s has %s free: less than %s would be left (%s; hwnpu cache prune frees room, %s moves the cache)",
			ErrNoRoom, what, root, FormatSize(free), FormatSize(minFree), MinFreeEnv, RootEnv)
	case over > 0:
		return fmt.Errorf("%w: %s the cache (%s) would go %s over %s=%s: what is left was used in the last %v or is in use (hwnpu cache clear removes it)",
			ErrNoRoom, what, root, FormatSize(over), MaxSizeEnv, FormatSize(l.MaxSize), KeepRecent)
	}
	added += need
	return nil
}

// evictLRU removes the units of u least recently used, oldest first, until bytes are freed, but
// none Removable keeps or keep names. It returns the bytes freed (the units' own) and takes them
// out of u.
func evictLRU(u *Usage, bytes int64, keep map[string]bool, log func(string, ...any)) int64 {
	var freed int64
	now := time.Now()
	var left []Unit
	units := u.ByUse()
	for i, x := range units {
		if freed >= bytes {
			left = append(left, units[i:]...)
			break
		}
		if keep[x.Path] || !Removable(x, now, KeepRecent) {
			left = append(left, x)
			continue
		}
		if err := x.Remove(); err != nil {
			if log != nil {
				log("cache: removing %s: %v", x.Name, err)
			}
			left = append(left, x)
			continue
		}
		if log != nil {
			log("cache: removed %s (%s, last used %s) to make room", x.Name, FormatSize(x.Bytes), x.Used.Format(time.DateTime))
		}
		freed += x.Bytes
		u.Bytes -= x.Bytes
	}
	u.Units = left
	return freed
}

// restorable, when set (the Core AI backend), lists what outside the root is made again in
// little time and may go first when the disk is short: Core AI's entries of programs that have
// prebuilt ones here (their weights again; a few seconds to make from the prebuilt program, not
// the Neural Engine compiler's minutes).
var restorable func() []string

// SetRestorable sets what lists the paths (folders) made again soon (restorable).
func SetRestorable(f func() []string) { restorable = f }

// evictRestorable removes the paths restorable lists, the least recently used first, until
// bytes are freed, none used in the last KeepRecent or by this process (Use). It returns the
// bytes freed.
func evictRestorable(bytes int64, log func(string, ...any)) int64 {
	if restorable == nil {
		return 0
	}
	var units []Unit
	seen := map[fileKey]bool{}
	for _, p := range restorable() {
		b, n, t := tree(p, seen)
		if n > 0 {
			units = append(units, Unit{Name: p, Kind: KindRestorable, Path: p, Paths: []string{p}, Bytes: b, Files: n, Used: t})
		}
	}
	u := Usage{Units: units}
	var freed int64
	now := time.Now()
	for _, x := range u.ByUse() {
		if freed >= bytes {
			break
		}
		if !Removable(x, now, KeepRecent) {
			continue
		}
		if err := os.RemoveAll(x.Path); err != nil {
			if log != nil {
				log("cache: removing %s: %v", x.Path, err)
			}
			continue
		}
		if log != nil {
			log("cache: removed %s (%s, made again from its prebuilt program when next loaded) to make room", x.Path, FormatSize(x.Bytes))
		}
		freed += x.Bytes
	}
	return freed
}

// KindRestorable is the kind of what restorable lists (outside the root).
const KindRestorable = "restorable"

// Use says this process uses path, a folder outside the root (restorable): it is not removed
// while the process runs.
func Use(path string) {
	usedMu.Lock()
	usedHere[path] = true
	usedMu.Unlock()
}
