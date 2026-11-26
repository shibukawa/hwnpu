package coreai

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const testTool = "3605.6.3"

// fakeAsset makes the program name of export, its bytecode body (main.hash its SHA-256).
func fakeAsset(t *testing.T, export, name, body string) string {
	t.Helper()
	d := filepath.Join(export, name+".aimodel")
	os.MkdirAll(d, 0o755)
	os.WriteFile(filepath.Join(d, "main.mlirb"), []byte(body), 0o644)
	sum := sha256.Sum256([]byte(body))
	os.WriteFile(filepath.Join(d, "main.hash"), sum[:], 0o644)
	return d
}

// fakePacked makes a packed prebuilt program of the program name (bytecode body) for arch in
// root, made by coreai-build tool, its weights a map onto the bytecode.
func fakePacked(t *testing.T, root, name, arch, body, tool string, ane bool) string {
	t.Helper()
	d := filepath.Join(root, name+"."+arch+".aimodelc")
	os.MkdirAll(d, 0o755)
	sum := sha256.Sum256([]byte(body))
	os.WriteFile(filepath.Join(d, "main.hash"), sum[:], 0o644)
	md, _ := json.Marshal(metadata{Producer: ToolName(tool), CreationDate: "20261125T103530Z", SourceHash: strings.ToUpper(hex.EncodeToString(sum[:]))})
	os.WriteFile(filepath.Join(d, "metadata.json"), md, 0o644)
	m := filepath.Join(d, "main-"+arch+"-delegates", "MPSGraph", "mpsExecutable.mpsgraphpackage", "manifest.plist")
	os.MkdirAll(filepath.Dir(m), 0o755)
	mark := "nothing"
	if ane {
		mark = "mps.fullyPlacedOnANE"
	}
	os.WriteFile(m, []byte(mark), 0o644)
	w := []byte(body)
	ws := sha256.Sum256(w)
	mp, _ := json.Marshal(resMap{Format: resMapFormat, Source: "main.mlirb", Size: int64(len(w)), SHA256: hex.EncodeToString(ws[:]), Ops: []mapOp{{From: 0, Len: int64(len(w))}}})
	os.WriteFile(filepath.Join(d, "resources.bin"+MapSuffix), mp, 0o644)
	return d
}

func hexOf(body string) string {
	s := sha256.Sum256([]byte(body))
	return hex.EncodeToString(s[:])
}

// reuseCase: models/<dir> with the earlier export old.coreai (packed for h15g) and the new
// new.coreai of the program dit_0 of the same source.
type reuseCase struct {
	dir, old, new, asset string
}

func newReuseCase(t *testing.T) reuseCase {
	dir := t.TempDir()
	c := reuseCase{dir: dir, old: filepath.Join(dir, "m-v5.coreai"), new: filepath.Join(dir, "m-v6.coreai")}
	fakeAsset(t, c.old, "dit_0", "bytecode dit_0")
	c.asset = fakeAsset(t, c.new, "dit_0", "bytecode dit_0")
	return c
}

func (c reuseCase) oldRoot() string { return PackedRoot(c.old, "h15g") }

func (c reuseCase) record(t *testing.T, root string, e *ProgramRecord) {
	t.Helper()
	if err := WriteRecord(root, &Record{Arch: "h15g", Pack: PackFormat, Export: filepath.Base(c.old), Tool: ToolName(testTool), Programs: map[string]*ProgramRecord{"dit_0": e}}); err != nil {
		t.Fatal(err)
	}
}

func (c reuseCase) build(t *testing.T, o BuildOptions) ([]BuildResult, string, error) {
	t.Helper()
	var log bytes.Buffer
	o.Archs, o.Packed, o.Log = []string{"h15g"}, true, &log
	if o.Tool == "" {
		o.Tool = testTool
	}
	if o.Jobs == 0 {
		o.Jobs = 3
	}
	rs, err := Build(context.Background(), []string{c.new}, o)
	return rs, log.String(), err
}

func good() *ProgramRecord {
	return &ProgramRecord{Source: hexOf("bytecode dit_0"), Result: ResultBuilt, Tool: ToolName(testTool), Jobs: 3,
		Device: &DeviceCheck{Status: DeviceOK, Checked: "2026-11-26", By: "test"}}
}

// What a packed build takes from the earlier export, and what it compiles (docs/prebuilt-reuse.md).
func TestBuildReuse(t *testing.T) {
	for _, tc := range []struct {
		name   string
		setup  func(t *testing.T, c reuseCase)
		o      BuildOptions
		want   string // in the log
		bad    bool   // fails on errBadOnDevice
		reused bool
	}{
		{name: "same source, good record", want: "would take the one of m-v5.coreai", setup: func(t *testing.T, c reuseCase) {
			fakePacked(t, c.oldRoot(), "dit_0", "h15g", "bytecode dit_0", testTool, true)
			c.record(t, c.oldRoot(), good())
		}},
		{name: "unchecked on the device", want: "would take the one of m-v5.coreai", setup: func(t *testing.T, c reuseCase) {
			fakePacked(t, c.oldRoot(), "dit_0", "h15g", "bytecode dit_0", testTool, true)
			e := good()
			e.Device = nil
			c.record(t, c.oldRoot(), e)
		}},
		{name: "another coreai-build", want: "made by coreai-build-3605.5.0, coreai-build-3605.6.3 here", setup: func(t *testing.T, c reuseCase) {
			fakePacked(t, c.oldRoot(), "dit_0", "h15g", "bytecode dit_0", "3605.5.0", true)
			c.record(t, c.oldRoot(), good())
		}},
		{name: "failed on the device", bad: true, want: "failedToSpecialize at s1024x1024", setup: func(t *testing.T, c reuseCase) {
			fakePacked(t, c.oldRoot(), "dit_0", "h15g", "bytecode dit_0", testTool, true)
			e := good()
			e.Device = &DeviceCheck{Status: "failedToSpecialize", Function: "s1024x1024", Checked: "2026-11-26", By: "test"}
			c.record(t, c.oldRoot(), e)
		}},
		{name: "left by the recipe now", want: "left to the device by the recipe", o: BuildOptions{Left: func(string, string) string { return "why, 2026-11-26" }}, setup: func(t *testing.T, c reuseCase) {
			fakePacked(t, c.oldRoot(), "dit_0", "h15g", "bytecode dit_0", testTool, true)
			c.record(t, c.oldRoot(), good())
		}},
		{name: "left by the recipe then", want: "would compile", setup: func(t *testing.T, c reuseCase) {
			os.MkdirAll(c.oldRoot(), 0o755)
			c.record(t, c.oldRoot(), &ProgramRecord{Source: hexOf("bytecode dit_0"), Result: ResultLeft, Rule: "why, 2026-11-20"})
		}},
		{name: "recorded as failed", want: "recorded as failed", setup: func(t *testing.T, c reuseCase) {
			fakePacked(t, c.oldRoot(), "dit_0", "h15g", "bytecode dit_0", testTool, true)
			e := good()
			e.Result, e.Device = ResultFailed, nil
			c.record(t, c.oldRoot(), e)
		}},
		{name: "more at once then", want: "compiled 3 at once, 1 now", o: BuildOptions{Jobs: 1}, setup: func(t *testing.T, c reuseCase) {
			fakePacked(t, c.oldRoot(), "dit_0", "h15g", "bytecode dit_0", testTool, true)
			c.record(t, c.oldRoot(), good())
		}},
		{name: "off the Neural Engine", want: "off the Neural Engine", setup: func(t *testing.T, c reuseCase) {
			fakePacked(t, c.oldRoot(), "dit_0", "h15g", "bytecode dit_0", testTool, false)
			c.record(t, c.oldRoot(), good())
		}},
		{name: "another source", want: "would compile to " /* and no reason */, setup: func(t *testing.T, c reuseCase) {
			fakePacked(t, c.oldRoot(), "dit_0", "h15g", "bytecode of another", testTool, true)
		}},
		{name: "set aside in its own .prev", want: "would take the one of m-v6.coreai (set aside)", setup: func(t *testing.T, c reuseCase) {
			prev := PackedRoot(c.new, "h15g") + prevSuffix
			fakePacked(t, prev, "dit_0", "h15g", "bytecode dit_0", testTool, true)
			WriteRecord(prev, &Record{Arch: "h15g", Pack: PackFormat, Export: filepath.Base(c.new), Programs: map[string]*ProgramRecord{"dit_0": good()}})
		}},
		{name: "no record", want: "no record of how it was built", setup: func(t *testing.T, c reuseCase) {
			fakePacked(t, c.oldRoot(), "dit_0", "h15g", "bytecode dit_0", testTool, true)
		}},
		{name: "no record, backfilled", want: "would take the one of m-v5.coreai", setup: func(t *testing.T, c reuseCase) {
			fakePacked(t, c.oldRoot(), "dit_0", "h15g", "bytecode dit_0", testTool, true)
			if keys, err := Backfill(c.oldRoot(), "h15g", "m-v5.coreai", "M", 3, "2026-11-26"); err != nil || len(keys) != 1 {
				t.Fatal(keys, err)
			}
		}},
		{name: "backfilled, failed on the device", bad: true, want: "failedToSpecialize", setup: func(t *testing.T, c reuseCase) {
			fakePacked(t, c.oldRoot(), "dit_0", "h15g", "bytecode dit_0", testTool, true)
			MarkDevice(c.old, "h15g", nil, DeviceCheck{Status: "failedToSpecialize"})
			Backfill(c.oldRoot(), "h15g", "m-v5.coreai", "M", 3, "2026-11-26")
		}},
		{name: "no record, in place", want: "prebuilt already", setup: func(t *testing.T, c reuseCase) {
			fakePacked(t, PackedRoot(c.new, "h15g"), "dit_0", "h15g", "bytecode dit_0", testTool, true)
		}},
		{name: "another pack format", want: "packed in format 1", setup: func(t *testing.T, c reuseCase) {
			fakePacked(t, c.oldRoot(), "dit_0", "h15g", "bytecode dit_0", testTool, true)
			WriteRecord(c.oldRoot(), &Record{Arch: "h15g", Pack: 1, Export: "m-v5.coreai", Programs: map[string]*ProgramRecord{"dit_0": good()}})
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := newReuseCase(t)
			tc.setup(t, c)
			o := tc.o
			o.DryRun = true
			rs, log, err := c.build(t, o)
			if tc.bad != (err != nil && IsBadOnDevice(err)) {
				t.Errorf("err %v", err)
			}
			if !strings.Contains(log, tc.want) {
				t.Errorf("log, want %q:\n%s", tc.want, log)
			}
			if tc.name == "another source" && strings.Contains(log, "(not ") {
				t.Errorf("a reason for a program of another source:\n%s", log)
			}
			if len(rs) != 1 {
				t.Fatalf("results %v", rs)
			}
		})
	}
}

// A build takes the earlier export's program: copied in, and its record says where from, with
// what the device made of it; the one in place (of another coreai-build) is set aside.
func TestBuildReuseTakes(t *testing.T) {
	c := newReuseCase(t)
	fakePacked(t, c.oldRoot(), "dit_0", "h15g", "bytecode dit_0", testTool, true)
	c.record(t, c.oldRoot(), good())
	newRoot := PackedRoot(c.new, "h15g")
	fakePacked(t, newRoot, "dit_0", "h15g", "bytecode dit_0", "3605.5.0", true)
	text := fakeAsset(t, c.new, "text_0", "bytecode text") // left to the device (LeftToDevice)

	rs, log, err := c.build(t, BuildOptions{Model: "M"})
	if err != nil {
		t.Fatalf("%v\n%s", err, log)
	}
	if len(rs) != 1 || rs[0].Reused != "m-v5.coreai" {
		t.Fatalf("results %+v\n%s", rs, log)
	}
	out := PackedPath(c.asset, "h15g")
	if !sameSource(c.asset, out) || !onANE(out) {
		t.Errorf("not taken: %s", out)
	}
	if md, _ := readMetadata(out); md.Producer != ToolName(testTool) {
		t.Errorf("the one in place was kept: %s", md.Producer)
	}
	rec, err := ReadRecord(newRoot)
	if err != nil || rec == nil {
		t.Fatalf("record %v %v", rec, err)
	}
	e := rec.Programs["dit_0"]
	if e == nil || e.Result != ResultReused || e.From != "m-v5.coreai" || e.Device == nil || e.Device.Status != DeviceOK || e.Jobs != 3 {
		t.Errorf("entry %+v", e)
	}
	if l := rec.Programs["text_0"]; l == nil || l.Result != ResultLeft || l.Rule != "LeftToDevice" || l.Source != sourceHex(text) {
		t.Errorf("text entry %+v", l)
	}
	if rec.Model != "M" || rec.Tool != ToolName(testTool) || rec.Export != "m-v6.coreai" {
		t.Errorf("record %+v", rec)
	}
	// set aside, with its entry
	prev := filepath.Join(newRoot+prevSuffix, "dit_0.h15g.aimodelc")
	if md, _ := readMetadata(prev); md.Producer != ToolName("3605.5.0") {
		t.Errorf("not set aside: %+v", md)
	}
	pr, _ := ReadRecord(newRoot + prevSuffix)
	if pr == nil || pr.Programs["dit_0"] == nil || !pr.Programs["dit_0"].Legacy {
		t.Errorf("record of .prev %+v", pr)
	}

	// built again: kept in place
	_, log, err = c.build(t, BuildOptions{})
	if err != nil || !strings.Contains(log, "prebuilt already") {
		t.Errorf("%v\n%s", err, log)
	}
}

// A program whose map does not make its file from the bytecode is not taken.
func TestCheckPacked(t *testing.T) {
	c := newReuseCase(t)
	d := fakePacked(t, c.oldRoot(), "dit_0", "h15g", "bytecode dit_0", testTool, true)
	if err := checkPacked(d, c.asset); err != nil {
		t.Fatal(err)
	}
	other := fakeAsset(t, filepath.Join(c.dir, "x.coreai"), "dit_0", "bytecode of the same length")
	if err := checkPacked(d, other); err == nil {
		t.Error("a map onto another bytecode was taken")
	}
}

// -mark writes what the device made of the programs; a failure stays over an ok of another
// function, and a person's ok of the whole program replaces it.
func TestMarkDevice(t *testing.T) {
	c := newReuseCase(t)
	root := PackedRoot(c.old, "h15g")
	fakePacked(t, root, "dit_single_0", "h15g", "a", testTool, true)
	fakePacked(t, root, "dit_single_1", "h15g", "b", testTool, true)
	fakePacked(t, root, "dit_double_0", "h15g", "c", testTool, true)
	marked, err := MarkDevice(c.old, "h15g", []string{"dit_single_*"}, DeviceCheck{Status: "failedToSpecialize", Function: "s1024x1024"})
	if err != nil || strings.Join(marked, " ") != "dit_single_0 dit_single_1" {
		t.Fatalf("%v %v", marked, err)
	}
	rec, _ := ReadRecord(root)
	e := rec.Programs["dit_single_0"]
	if !e.Device.Bad() || !e.Legacy || e.Tool != ToolName(testTool) || rec.Programs["dit_double_0"] != nil {
		t.Errorf("record %+v %+v", rec, e)
	}
	MarkDevice(c.old, "h15g", []string{"dit_single_0"}, DeviceCheck{Status: DeviceOK, Function: "s512x512"})
	rec, _ = ReadRecord(root)
	if !rec.Programs["dit_single_0"].Device.Bad() {
		t.Error("an ok of another function replaced the failure")
	}
	MarkDevice(c.old, "h15g", []string{"dit_single_0"}, DeviceCheck{Status: DeviceOK})
	rec, _ = ReadRecord(root)
	if rec.Programs["dit_single_0"].Device.Bad() {
		t.Error("an ok of the whole program did not replace the failure")
	}
	if _, err := MarkDevice(c.old, "h15g", []string{"nothing_*"}, DeviceCheck{Status: DeviceOK}); err == nil {
		t.Error("no program marked, no error")
	}
}

// A legacy program a device finished is taken without a backfilled record.
func TestBuildReuseLegacyMarkedOK(t *testing.T) {
	c := newReuseCase(t)
	fakePacked(t, c.oldRoot(), "dit_0", "h15g", "bytecode dit_0", testTool, true)
	if _, err := MarkDevice(c.old, "h15g", nil, DeviceCheck{Status: DeviceOK}); err != nil {
		t.Fatal(err)
	}
	_, log, err := c.build(t, BuildOptions{DryRun: true})
	if err != nil || !strings.Contains(log, "would take the one of m-v5.coreai") {
		t.Errorf("%v\n%s", err, log)
	}
}

// Backfill records what the files say of programs made before records, keeps a device's
// mark, and leaves those recorded by a build.
func TestBackfill(t *testing.T) {
	c := newReuseCase(t)
	root := c.oldRoot()
	fakePacked(t, root, "a", "h15g", "a", testTool, true)
	fakePacked(t, root, "b", "h15g", "b", testTool, true)
	MarkDevice(c.old, "h15g", []string{"a"}, DeviceCheck{Status: "failedToSpecialize"})
	keys, err := Backfill(root, "h15g", "m-v5.coreai", "M", 3, "2026-11-26")
	if err != nil || strings.Join(keys, " ") != "a b" {
		t.Fatal(keys, err)
	}
	rec, _ := ReadRecord(root)
	a, b := rec.Programs["a"], rec.Programs["b"]
	if a.Legacy || !a.Device.Bad() || b.Legacy || b.Jobs != 3 || b.Tool != ToolName(testTool) || b.Backfilled != "2026-11-26" || b.Result != ResultBuilt {
		t.Errorf("a %+v b %+v", a, b)
	}
	if rec.Model != "M" || rec.Tool != ToolName(testTool) {
		t.Errorf("record %+v", rec)
	}
	if keys, err := Backfill(root, "h15g", "m-v5.coreai", "M", 3, "2026-11-27"); err != nil || len(keys) != 0 {
		t.Errorf("recorded again: %v %v", keys, err)
	}
}

// A rule that leaves a program to the device now sets aside its prebuilt program there (a
// device would take it from the bundle); without the rule again, it is taken back.
func TestBuildLeftSetsAside(t *testing.T) {
	c := newReuseCase(t)
	newRoot := PackedRoot(c.new, "h15g")
	out := fakePacked(t, newRoot, "dit_0", "h15g", "bytecode dit_0", testTool, true)
	WriteRecord(newRoot, &Record{Arch: "h15g", Pack: PackFormat, Export: "m-v6.coreai", Programs: map[string]*ProgramRecord{"dit_0": good()}})
	rs, log, err := c.build(t, BuildOptions{Left: func(string, string) string { return "why, 2026-11-26" }})
	if err != nil || rs[0].Left == "" {
		t.Fatalf("%v %+v\n%s", err, rs, log)
	}
	if _, err := os.Stat(out); err == nil {
		t.Error("the prebuilt program of a program left to the device stayed")
	}
	if rec, _ := ReadRecord(newRoot); rec.Programs["dit_0"].Result != ResultLeft {
		t.Errorf("record %+v", rec.Programs["dit_0"])
	}
	_, log, err = c.build(t, BuildOptions{})
	if err != nil || !strings.Contains(log, "taken from m-v6.coreai (set aside)") {
		t.Errorf("%v\n%s", err, log)
	}
}

// A run marks an ok or a failedToSpecialize, no other failure (another process's), and writes
// only what changes the record.
func TestDeviceMark(t *testing.T) {
	if d, ok := DeviceMark("s512", "", "2026-11-26", "x"); !ok || d.Status != DeviceOK {
		t.Errorf("ok: %+v %v", d, ok)
	}
	if d, ok := DeviceMark("s512", "Error Domain=CoreAI failedToSpecialize", "2026-11-26", "x"); !ok || d.Status != FailedToSpecialize || !d.Bad() {
		t.Errorf("failedToSpecialize: %+v %v", d, ok)
	}
	if _, ok := DeviceMark("s512", "ANEProgramProcessRequestDirect() Failed with status=0xf Request cancelled", "2026-11-26", "x"); ok {
		t.Error("a cancelled request was marked")
	}
	r := &Record{Programs: map[string]*ProgramRecord{"a": {Source: "s"}}}
	ok := DeviceCheck{Status: DeviceOK, Function: "f1"}
	bad := DeviceCheck{Status: FailedToSpecialize, Function: "f1"}
	if !r.NeedsMark("a", "s", ok) || !r.NeedsMark("b", "s", ok) || !r.NeedsMark("a", "other", ok) {
		t.Error("a first mark is not written")
	}
	r.Programs["a"].Device = &DeviceCheck{Status: DeviceOK, Function: "f0"}
	if r.NeedsMark("a", "s", ok) {
		t.Error("an ok over an ok is written again")
	}
	if !r.NeedsMark("a", "s", bad) {
		t.Error("a failure over an ok is not written")
	}
	r.Programs["a"].Device = &DeviceCheck{Status: FailedToSpecialize, Function: "f1"}
	if r.NeedsMark("a", "s", bad) || r.NeedsMark("a", "s", DeviceCheck{Status: DeviceOK, Function: "f2"}) {
		t.Error("the same failure, or another function's ok, is written over the failure")
	}
	if !r.NeedsMark("a", "s", ok) {
		t.Error("the failed function's ok does not replace the failure")
	}
}
