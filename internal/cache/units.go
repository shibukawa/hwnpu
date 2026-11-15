package cache

import (
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"
)

// What the cache holds is counted, and removed, by unit: what is used and made again whole.
const (
	KindOriginal  = "original"  // original/OWNER/NAME: a repository's files (downloaded again)
	KindConverted = "converted" // models/NAME (programs), converted/MODEL/TARGET/EXPORT/vN (made or downloaded again)
	KindExport    = "export"    // entries/TARGET/NAME: an export for AMD's NPU
	KindKernels   = "kernels"   // iron/KEY: a kernel IRON compiled
	KindCompiled  = "compiled"  // KEY.mlmodelc, KEY.blob...: a program compiled for this machine
)

// Kinds are the kinds of units, in the order they are shown.
var Kinds = []string{KindOriginal, KindConverted, KindExport, KindKernels, KindCompiled}

// Unit is a part of the cache that is removed whole.
type Unit struct {
	Name  string // its path under the root, with /: original/Qwen/Qwen3-1.7B (compiled/KEY for a compiled program)
	Kind  string
	Path  string   // its folder (or file)
	Paths []string // what Remove removes: Path, and a compiled program's other files
	Bytes int64    // its files' (a file linked from another unit counted once, in the first)
	Files int
	// Used is when it was last used: the latest of its folder's time (Touch) and its files'.
	Used time.Time
}

// Usage is what the root holds.
type Usage struct {
	Root  string
	Units []Unit
	// Other are the bytes of each folder of the root that is not in units (prompts, history,
	// work, ...): never removed by hwnpu cache prune.
	Other map[string]int64
	Bytes int64 // all of the root's
}

// ByKind are the bytes of the units of each kind.
func (u Usage) ByKind() map[string]int64 {
	m := map[string]int64{}
	for _, x := range u.Units {
		m[x.Kind] += x.Bytes
	}
	return m
}

// ByUse are the units, the least recently used first.
func (u Usage) ByUse() []Unit {
	s := slices.Clone(u.Units)
	slices.SortStableFunc(s, func(a, b Unit) int { return a.Used.Compare(b.Used) })
	return s
}

// Scan walks the root and counts what it holds.
func Scan() (Usage, error) {
	root := Root()
	u := Usage{Root: root, Other: map[string]int64{}}
	des, err := os.ReadDir(root)
	if os.IsNotExist(err) {
		return u, nil
	}
	if err != nil {
		return u, err
	}
	seen := map[fileKey]bool{}
	add := func(x Unit) {
		x.Paths = append([]string{x.Path}, x.Paths...)
		for _, p := range x.Paths {
			b, n, t := tree(p, seen)
			x.Bytes += b
			x.Files += n
			if t.After(x.Used) {
				x.Used = t
			}
		}
		if x.Files == 0 {
			return // a folder made for a download that did not start
		}
		u.Units = append(u.Units, x)
		u.Bytes += x.Bytes
	}
	compiled := map[string]*Unit{}
	var keys []string
	for _, de := range des {
		name := de.Name()
		p := filepath.Join(root, name)
		switch name {
		case "original":
			for _, d := range dirs(p, 2) {
				add(Unit{Kind: KindOriginal, Path: d})
			}
			continue
		case "models", "iron":
			kind := KindConverted
			if name == "iron" {
				kind = KindKernels
			}
			for _, d := range dirs(p, 1) {
				add(Unit{Kind: kind, Path: d})
			}
			continue
		case "entries":
			for _, d := range dirs(p, 2) {
				add(Unit{Kind: KindExport, Path: d})
			}
			continue
		case "converted":
			in := u.Bytes
			for _, d := range entryDirs(p) {
				add(Unit{Kind: KindConverted, Path: d})
			}
			b, _, _ := tree(p, map[fileKey]bool{})
			b -= u.Bytes - in // what no entry.json is with (a download cut off)
			u.Other[name] += b
			u.Bytes += b
			continue
		}
		if key, _, ok := splitEntry(name); ok && !strings.HasPrefix(name, ".tmp-") {
			c := compiled[key]
			if c == nil {
				c = &Unit{Name: "compiled/" + key, Kind: KindCompiled}
				compiled[key] = c
				keys = append(keys, key)
			}
			c.Paths = append(c.Paths, p)
			continue
		}
		b, _, _ := tree(p, seen)
		u.Other[name] += b
		u.Bytes += b
	}
	for _, k := range keys {
		c := compiled[k]
		c.Path, c.Paths = c.Paths[0], c.Paths[1:]
		add(*c)
	}
	for i := range u.Units {
		if u.Units[i].Name == "" {
			rel, _ := filepath.Rel(root, u.Units[i].Path)
			u.Units[i].Name = filepath.ToSlash(rel)
		}
	}
	return u, nil
}

// dirs are the entries depth folders below dir (original/OWNER/NAME: 2), the ones that are
// being written (.part) left out.
func dirs(dir string, depth int) []string {
	out := []string{dir}
	for range depth {
		var next []string
		for _, d := range out {
			des, _ := os.ReadDir(d)
			for _, de := range des {
				if !strings.HasSuffix(de.Name(), ".part") {
					next = append(next, filepath.Join(d, de.Name()))
				}
			}
		}
		out = next
	}
	return out
}

// entryDirs are the folders under dir with an entry.json.
func entryDirs(dir string) []string {
	var out []string
	filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil || !d.IsDir() {
			return nil
		}
		if _, err := os.Stat(filepath.Join(p, "entry.json")); err == nil {
			out = append(out, p)
			return filepath.SkipDir
		}
		return nil
	})
	return out
}

// tree is the bytes of the files under p (or p, a file) not in seen, which it adds them to, the
// number of the files, and the latest time of p and of its files.
func tree(p string, seen map[fileKey]bool) (int64, int, time.Time) {
	var n int64
	var files int
	var t time.Time
	filepath.WalkDir(p, func(q string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		fi, err := d.Info()
		if err != nil {
			return nil
		}
		if mt := fi.ModTime(); mt.After(t) && (q == p || !d.IsDir()) {
			t = mt
		}
		if !fi.Mode().IsRegular() {
			return nil
		}
		files++
		if k, ok := keyOf(fi); ok {
			if seen[k] {
				return nil
			}
			seen[k] = true
		}
		n += fi.Size()
		return nil
	})
	return n, files, t
}

var (
	usedMu sync.Mutex
	// usedHere are the units this process used (Touch): not removed while it runs.
	usedHere = map[string]bool{}
	busy     func() []string
)

// SetBusy sets what says the paths work is queued for (store's queue of writebacks): their
// units are not removed.
func SetBusy(f func() []string) { busy = f }

// UnitOf is the folder of the unit path is in under the root (path itself for a compiled
// program), and whether it is in one.
func UnitOf(path string) (string, bool) {
	root := Root()
	rel, err := filepath.Rel(root, path)
	if err != nil || rel == "." || strings.HasPrefix(rel, "..") {
		return "", false
	}
	parts := strings.Split(filepath.ToSlash(rel), "/")
	n := 0
	switch parts[0] {
	case "original", "entries":
		n = 3
	case "models", "iron":
		n = 2
	case "converted":
		n = 5 // converted/MODEL/TARGET/EXPORT/vN
	default:
		if _, _, ok := splitEntry(parts[0]); ok && len(parts) >= 1 {
			n = 1
		}
	}
	if n == 0 || len(parts) < n {
		return "", false
	}
	unit := strings.TrimSuffix(strings.Join(parts[:n], "/"), ".part")
	return filepath.Join(root, filepath.FromSlash(unit)), true
}

// Touch says the unit path is in is used now: it is kept while this process runs, and its
// folder's time is set to now (at most every few minutes), which LRU orders by.
func Touch(path string) {
	unit, ok := UnitOf(path)
	if !ok {
		return
	}
	usedMu.Lock()
	usedHere[unit] = true
	usedMu.Unlock()
	if fi, err := os.Stat(unit); err == nil && time.Since(fi.ModTime()) > 5*time.Minute {
		now := time.Now()
		os.Chtimes(unit, now, now)
	}
}

// Removable reports whether hwnpu may remove x now: not used in the last keep (another process
// may read it), nor by this process, nor with work queued for it.
func Removable(x Unit, now time.Time, keep time.Duration) bool {
	if now.Sub(x.Used) < keep {
		return false
	}
	usedMu.Lock()
	here := usedHere[x.Path]
	usedMu.Unlock()
	if here {
		return false
	}
	if busy != nil {
		for _, p := range busy() {
			if u, ok := UnitOf(p); ok && u == x.Path {
				return false
			}
		}
	}
	return true
}

// Remove removes x. A repository's lock goes first, so that a process reading it after takes
// none of its files as there.
func (x Unit) Remove() error {
	if x.Kind == KindOriginal {
		if err := os.Remove(filepath.Join(x.Path, "hwnpu-lock.json")); err != nil && !os.IsNotExist(err) {
			return err
		}
	}
	if x.Kind == KindConverted {
		os.Remove(filepath.Join(x.Path, "entry.json")) // a folder without it is not whole
	}
	var errs []error
	for _, p := range x.Paths {
		if err := os.RemoveAll(p); err != nil {
			errs = append(errs, err)
		}
	}
	for d := filepath.Dir(x.Path); d != Root() && strings.HasPrefix(d, Root()); d = filepath.Dir(d) {
		if os.Remove(d) != nil { // the folders left empty (original/OWNER)
			break
		}
	}
	forgetScan()
	if len(errs) > 0 {
		return errs[0]
	}
	return nil
}

// forgetScan makes Room scan the root again (what it held changed).
func forgetScan() {
	if roomMu.TryLock() {
		scanned = nil
		roomMu.Unlock()
	}
}

// PruneOptions say what PlanPrune removes.
type PruneOptions struct {
	// MaxSize: the units least recently used go until the cache holds at most this (0: none).
	MaxSize int64
	// OlderThan: the units not used for this long go (0: none).
	OlderThan time.Duration
	// Kinds are the kinds of units looked at (nil: all).
	Kinds []string
	// Keep is how recently used a unit is kept whatever the rest says (KeepRecent when 0).
	Keep time.Duration
}

// PlanPrune is what Prune removes of u, the least recently used first. Units Removable keeps
// are never in it.
func PlanPrune(u Usage, o PruneOptions) []Unit {
	keep := o.Keep
	if keep == 0 {
		keep = KeepRecent
	}
	now := time.Now()
	total := u.Bytes
	var out []Unit
	for _, x := range u.ByUse() {
		if o.Kinds != nil && !slices.Contains(o.Kinds, x.Kind) {
			continue
		}
		old := o.OlderThan > 0 && now.Sub(x.Used) >= o.OlderThan
		over := o.MaxSize > 0 && total > o.MaxSize
		if !old && !over || !Removable(x, now, keep) {
			continue
		}
		out = append(out, x)
		total -= x.Bytes
	}
	return out
}
