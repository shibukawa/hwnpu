package coreai

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"
)

// A packed build (BuildOptions.Packed) takes the prebuilt program of a program from another
// export of the same source, as an earlier version of the model's that did not change the
// program, rather than compiling it again (an hour a program on h13g): docs/prebuilt-reuse.md.
// Only when all of these hold of the one taken:
//
//  1. its source is the program's (main.hash, and metadata.json's sourceHash)
//  2. it is for the architecture (its folder and name)
//  3. the same coreai-build made it (metadata.json's producer)
//  4. the recipe leaves the program to the device neither now (Left: nothing is compiled) nor in
//     its record (the rules of the recipes only leave programs to the device)
//  5. its record says it was built (or reused) with no more compiles at once than now, it is on
//     the Neural Engine, and no device failed it
//  6. it is packed in this PackFormat and its maps make their files from the program's bytecode
//
// A program whose record says the device failed it is not compiled again (the same source makes
// the same program): the build fails on it, and says the rule that leaves it to the device.

// errBadOnDevice: a prebuilt program of the program's source failed on a device.
var errBadOnDevice = errors.New("a prebuilt program of the same source failed on the device")

// candidate is a packed prebuilt program a build may take.
type candidate struct {
	root   string // the PackedRoot (or its .prev) it is in
	export string // the export it was built for, as its record names it
	own    bool   // of the export being built (its PackedRoot, not .prev)
	key    string
	dir    string
	rec    *Record
	entry  *ProgramRecord // nil when the record has none of it (or there is no record)
}

func (c *candidate) label() string {
	if strings.HasSuffix(c.root, prevSuffix) {
		return c.export + " (set aside)"
	}
	return c.export
}

// prevSuffix ends the folder beside a PackedRoot where a build sets aside the prebuilt programs
// it replaces (they may be taken again).
const prevSuffix = ".prev"

// reuser decides, for a packed build, which programs to take from earlier builds.
type reuser struct {
	tool string // coreai-build's version, "" when not known
	jobs int
	dry  bool
	log  func(format string, a ...any)

	indexes map[string]map[string][]*candidate // by export and arch, by source

	mu      sync.Mutex
	stashed []stashed
}

type stashed struct{ root, key string }

// index is the candidates for the export's programs for arch, by their source: the export's
// own, those it set aside, and those of the exports beside it (and set aside there).
func (u *reuser) index(export, arch string) map[string][]*candidate {
	k := export + "\x00" + arch
	if idx, ok := u.indexes[k]; ok {
		return idx
	}
	own := PackedRoot(export, arch)
	roots := []string{own, own + prevSuffix}
	others, _ := filepath.Glob(filepath.Join(filepath.Dir(filepath.Clean(export)), "*"+PackedSuffix))
	slices.Sort(others)
	for _, o := range others {
		if o == filepath.Clean(export)+PackedSuffix {
			continue
		}
		roots = append(roots, filepath.Join(o, arch), filepath.Join(o, arch)+prevSuffix)
	}
	idx := map[string][]*candidate{}
	for _, root := range roots {
		if _, err := os.Stat(root); err != nil {
			continue
		}
		rec, err := ReadRecord(root)
		if err != nil {
			u.log("%v (its prebuilt programs are taken as without a record)\n", err)
			rec = nil
		}
		name := filepath.Base(strings.TrimSuffix(filepath.Dir(root), PackedSuffix))
		if rec != nil && rec.Export != "" {
			name = rec.Export
		}
		for _, p := range packedPrograms(root, arch) {
			c := &candidate{root: root, export: name, own: root == own, key: p.key, dir: p.dir, rec: rec}
			if rec != nil {
				c.entry = rec.Programs[p.key]
			}
			idx[p.source] = append(idx[p.source], c)
		}
	}
	if u.indexes == nil {
		u.indexes = map[string]map[string][]*candidate{}
	}
	u.indexes[k] = idx
	return idx
}

// decide is what becomes of r: the candidate to take (r.Out itself: kept), or why none is (""
// when there was none of its source), or errBadOnDevice.
func (u *reuser) decide(r *BuildResult) (*candidate, string, error) {
	src := sourceHex(r.Asset)
	export, ok := ExportOf(r.Asset)
	if src == "" || !ok {
		return nil, "", nil
	}
	cands := slices.Clone(u.index(export, r.Arch)[src])
	for _, c := range cands {
		if e := c.entry; e != nil && e.Source == src && e.Device.Bad() {
			r.bad = c
			at := ""
			if e.Device.Function != "" {
				at = " at " + e.Device.Function
			}
			return nil, "", fmt.Errorf("%w: %s%s (%s, %s by %s)", errBadOnDevice, e.Device.Status, at, c.label(), e.Device.Checked, e.Device.By)
		}
	}
	// the one in place first, then those a device finished
	rank := func(c *candidate) int {
		switch {
		case c.dir == r.Out:
			return 0
		case c.entry != nil && c.entry.Device != nil && c.entry.Device.Status == DeviceOK:
			return 1
		}
		return 2
	}
	slices.SortStableFunc(cands, func(a, b *candidate) int { return rank(a) - rank(b) })
	why := ""
	for _, c := range cands {
		reason := u.check(c, src, c.dir == r.Out)
		if reason == "" && c.dir != r.Out && !u.dry {
			if err := checkPacked(c.dir, r.Asset); err != nil {
				reason = err.Error()
			}
		}
		if reason == "" {
			return c, "", nil
		}
		if why == "" {
			why = "the one of " + c.label() + ": " + reason
		}
	}
	return nil, why, nil
}

// check is why the candidate is not taken for a program of source src, "" when it is
// (conditions 1 to 5 and 6's format; the maps are checked after). inPlace: it is where the
// program goes, which is kept without a record (a build from before records made it).
func (u *reuser) check(c *candidate, src string, inPlace bool) string {
	md, err := readMetadata(c.dir)
	if err != nil {
		return "no metadata.json"
	}
	if !strings.EqualFold(md.SourceHash, src) {
		return "its metadata.json is of another source"
	}
	switch {
	case u.tool != "":
		if md.Producer != ToolName(u.tool) {
			return fmt.Sprintf("made by %s, %s here", md.Producer, ToolName(u.tool))
		}
	case !inPlace:
		return "coreai-build's version is not known here"
	}
	if c.rec != nil && c.rec.Pack != PackFormat {
		return fmt.Sprintf("packed in format %d, %d here", c.rec.Pack, PackFormat)
	}
	e := c.entry
	if !inPlace && (e == nil || e.Legacy && (e.Device == nil || e.Device.Status != DeviceOK)) {
		return "no record of how it was built (" + RecordFile + ")"
	}
	if e != nil {
		switch {
		case e.Source != src:
			return "its record is of another source"
		case e.Result != ResultBuilt && e.Result != ResultReused:
			return "recorded as " + e.Result
		case e.Jobs > u.jobs:
			return fmt.Sprintf("compiled %d at once, %d now", e.Jobs, u.jobs)
		}
	}
	if !onANE(c.dir) {
		return "off the Neural Engine"
	}
	return ""
}

// take copies the candidate's program to out, setting aside the one there.
func (u *reuser) take(c *candidate, asset, arch, out string) error {
	if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
		return err
	}
	tmp, err := os.MkdirTemp(filepath.Dir(out), ".reuse-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmp)
	part := filepath.Join(tmp, filepath.Base(out))
	if err := copyDir(part, c.dir); err != nil {
		return err
	}
	if err := u.stash(asset, arch, out); err != nil {
		return err
	}
	return os.Rename(part, out)
}

// stash sets aside the prebuilt program at out (of the asset for arch), when one is there, in
// its PackedRoot's .prev, where a later build may take it.
func (u *reuser) stash(asset, arch, out string) error {
	if _, err := os.Stat(out); err != nil {
		return nil
	}
	export, ok := ExportOf(asset)
	if !ok {
		return os.RemoveAll(out)
	}
	root := PackedRoot(export, arch)
	rel, err := filepath.Rel(root, out)
	if err != nil || strings.HasPrefix(rel, "..") {
		return os.RemoveAll(out)
	}
	dst := filepath.Join(root+prevSuffix, rel)
	if err := os.RemoveAll(dst); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	if err := os.Rename(out, dst); err != nil {
		return err
	}
	u.mu.Lock()
	u.stashed = append(u.stashed, stashed{root, ProgramKey(export, asset)})
	u.mu.Unlock()
	return nil
}

// write writes the records of the build's PackedRoots: each program's result over what the
// record had, and those set aside in .prev's. left are the programs LeftToDevice. A root
// that is not there (nothing was packed) gets none.
func (u *reuser) write(results []BuildResult, left []string, archs []string, model string) error {
	now := time.Now().UTC()
	type group struct {
		export string
		rs     []*BuildResult
	}
	groups := map[string]*group{}
	var order []string
	add := func(export, arch string) *group {
		root := PackedRoot(export, arch)
		g, ok := groups[root]
		if !ok {
			g = &group{export: export}
			groups[root] = g
			order = append(order, root)
		}
		return g
	}
	for i := range results {
		r := &results[i]
		if export, ok := ExportOf(r.Asset); ok {
			g := add(export, r.Arch)
			g.rs = append(g.rs, r)
		}
	}
	for _, a := range left {
		if export, ok := ExportOf(a); ok {
			for _, arch := range archs {
				add(export, arch)
			}
		}
	}
	var firstErr error
	keep := func(err error) {
		if err != nil && firstErr == nil {
			firstErr = err
		}
	}
	for _, root := range order {
		g := groups[root]
		arch := filepath.Base(root)
		old, err := ReadRecord(root)
		if err != nil {
			u.log("%v (written again)\n", err)
		}
		// those set aside take their entries to .prev's record
		var moved []string
		for _, s := range u.stashed {
			if s.root == root {
				moved = append(moved, s.key)
			}
		}
		if len(moved) > 0 {
			keep(u.writePrev(root, arch, old, moved))
		}
		if _, err := os.Stat(root); err != nil {
			continue
		}
		rec := &Record{Arch: arch, Pack: PackFormat, Export: filepath.Base(g.export), Model: model, Programs: map[string]*ProgramRecord{}}
		if old != nil {
			for k, e := range old.Programs {
				rec.Programs[k] = e
			}
			if rec.Model == "" {
				rec.Model = old.Model
			}
			rec.Tool = old.Tool
		}
		if u.tool != "" {
			rec.Tool = ToolName(u.tool)
		}
		for _, a := range left {
			if e, ok := ExportOf(a); ok && e == g.export {
				rec.Programs[ProgramKey(e, a)] = &ProgramRecord{Source: sourceHex(a), Result: ResultLeft, Rule: "LeftToDevice"}
			}
		}
		for _, r := range g.rs {
			key, src := ProgramKey(g.export, r.Asset), sourceHex(r.Asset)
			switch {
			case r.Left != "":
				rec.Programs[key] = &ProgramRecord{Source: src, Result: ResultLeft, Rule: r.Left}
			case errors.Is(r.Err, context.Canceled) || errors.Is(r.Err, context.DeadlineExceeded):
			case errors.Is(r.Err, errBadOnDevice):
				if r.bad != nil && r.bad.dir == r.Out {
					break // the record says so already
				}
				e := &ProgramRecord{Source: src, Result: ResultFailed, Error: r.Err.Error()}
				if r.bad != nil {
					d := *r.bad.entry.Device
					e.Device, e.From, e.Tool = &d, r.bad.export, r.bad.entry.Tool
				}
				rec.Programs[key] = e
			case r.Err != nil:
				msg := r.Err.Error()
				if len(msg) > 500 {
					msg = msg[:500]
				}
				rec.Programs[key] = &ProgramRecord{Source: src, Result: ResultFailed, Tool: rec.Tool, Jobs: r.jobs, Built: now.Format(time.RFC3339), Seconds: int(r.Elapsed.Seconds()), Error: msg}
			case r.Skipped:
				if e := rec.Programs[key]; e == nil || e.Source != src {
					rec.Programs[key] = legacyRecord(r.Out, src)
				}
			case r.Reused != "":
				e := legacyRecord(r.Out, src)
				if r.from != nil && r.from.entry != nil {
					c := *r.from.entry
					if c.Device != nil {
						d := *c.Device
						c.Device = &d
					}
					e = &c
				}
				e.Result, e.From, e.Error = ResultReused, r.Reused, ""
				rec.Programs[key] = e
			case r.ran:
				rec.Programs[key] = &ProgramRecord{Source: src, Result: ResultBuilt, Tool: rec.Tool, Jobs: r.jobs, Built: now.Format(time.RFC3339), Seconds: int(r.Elapsed.Round(time.Second).Seconds())}
			}
		}
		keep(WriteRecord(root, rec))
	}
	return firstErr
}

// writePrev writes the entries of the programs set aside from root (keys, as old had them) in
// its .prev's record.
func (u *reuser) writePrev(root, arch string, old *Record, keys []string) error {
	prev := root + prevSuffix
	rec, err := ReadRecord(prev)
	if err != nil || rec == nil {
		rec = &Record{Arch: arch, Pack: PackFormat, Programs: map[string]*ProgramRecord{}}
		if old != nil {
			rec.Export, rec.Model, rec.Tool, rec.Pack = old.Export, old.Model, old.Tool, old.Pack
		}
	}
	if rec.Export == "" {
		rec.Export = filepath.Base(strings.TrimSuffix(filepath.Dir(root), PackedSuffix))
	}
	for _, k := range keys {
		var e *ProgramRecord
		if old != nil {
			e = old.Programs[k]
		}
		dir := filepath.Join(prev, filepath.FromSlash(k)+"."+arch+".aimodelc")
		if src := sourceHex(dir); e == nil || e.Source != src {
			e = legacyRecord(dir, src)
		}
		rec.Programs[k] = e
	}
	return WriteRecord(prev, rec)
}
