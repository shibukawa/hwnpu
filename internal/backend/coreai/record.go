package coreai

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"
)

// The record of a build (RecordFile, in a PackedRoot beside the packed programs, so that the
// bundle carries it): for each program of the export, what its prebuilt program for the
// architecture was compiled from, how (coreai-build's version, the compiles at once, the
// recipe's rule), and what became of it on a device of the architecture. Build takes a program
// of another export (an earlier version of the model's) with the same source only when the record
// says it was good (docs/prebuilt-reuse.md): coreai-build has written programs that a device then
// failed to finish (FLUX.2 klein v6's dit_single_* on h15g, failedToSpecialize, 2026-11-26), and
// the same source compiled again would be the same program.

// RecordFile is the name of the record in a PackedRoot.
const RecordFile = "build.json"

// recordFormat is the version of the record's layout.
const recordFormat = 1

// PackFormat is the version of what Pack writes (the shared layers' prebuiltFormat): a program
// packed in another is not taken.
const PackFormat = 2

// What became of a program in a build (ProgramRecord.Result).
const (
	ResultBuilt  = "built"  // compiled by coreai-build
	ResultReused = "reused" // copied from the prebuilt program of another export of the same source
	ResultFailed = "failed" // coreai-build failed, or the same source failed on the device
	ResultLeft   = "left"   // left to the device (the recipe's rule, or LeftToDevice)
)

// What a device made of a prebuilt program (DeviceCheck.Status); any other is a failure (the
// error's name: failedToSpecialize).
const (
	DeviceOK        = "ok"
	DeviceUnchecked = "unchecked"
)

// Record is a build's record of an export's prebuilt programs for an architecture.
type Record struct {
	Format int    `json:"format"`
	Arch   string `json:"arch"`
	// Tool is the coreai-build of the last build (its metadata.json's producer,
	// "coreai-build-3605.6.3").
	Tool string `json:"tool,omitempty"`
	// Pack is the PackFormat of the programs.
	Pack   int    `json:"pack"`
	Model  string `json:"model,omitempty"`
	Export string `json:"export"`
	// Programs are by the program's path in the export without .aimodel (ProgramKey).
	Programs map[string]*ProgramRecord `json:"programs"`
}

// ProgramRecord is what a build made of a program.
type ProgramRecord struct {
	// Source is its bytecode's hash (main.hash, in hex).
	Source string `json:"source"`
	Result string `json:"result"`
	// Rule is why it is left to the device (the recipe's rule's why and checked, or
	// LeftToDevice).
	Rule string `json:"rule,omitempty"`
	// Tool is the coreai-build that compiled it.
	Tool string `json:"tool,omitempty"`
	// Jobs are the compiles that ran at once with it (1: alone); 0 when not known.
	Jobs int `json:"jobs,omitempty"`
	// Built is when it was compiled (RFC 3339), Seconds how long it took.
	Built   string `json:"built,omitempty"`
	Seconds int    `json:"seconds,omitempty"`
	// From is the export it was copied from (ResultReused).
	From string `json:"from,omitempty"`
	// Legacy: the prebuilt program was made before records, and how (the compiles at once, the
	// recipe) is not known. Another export takes it only when a device finished it
	// (DeviceOK) or the build trusts such ones (BuildOptions.TrustOld).
	Legacy bool   `json:"legacy,omitempty"`
	Error  string `json:"error,omitempty"`
	// Device is what a device of the architecture made of it, nil when none was asked.
	Device *DeviceCheck `json:"device,omitempty"`
}

// DeviceCheck is what a device made of a prebuilt program.
type DeviceCheck struct {
	// Status is DeviceOK, DeviceUnchecked, or the failure's name (failedToSpecialize).
	Status string `json:"status"`
	// Checked is the day (YYYY-MM-DD).
	Checked string `json:"checked,omitempty"`
	// Function is the function that failed (s1024x1024), when one did.
	Function string `json:"function,omitempty"`
	// Message is the device's error.
	Message string `json:"message,omitempty"`
	// By is what checked it (hwnpu-image, hwnpu-dev prebuild -mark).
	By string `json:"by,omitempty"`
}

// Bad reports whether the device failed the program.
func (d *DeviceCheck) Bad() bool {
	return d != nil && d.Status != "" && d.Status != DeviceOK && d.Status != DeviceUnchecked
}

// ReadRecord reads the record in root (a PackedRoot); nil without an error when there is none.
func ReadRecord(root string) (*Record, error) {
	b, err := os.ReadFile(filepath.Join(root, RecordFile))
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var r Record
	if err := json.Unmarshal(b, &r); err != nil {
		return nil, fmt.Errorf("%s: %w", filepath.Join(root, RecordFile), err)
	}
	if r.Format != recordFormat {
		return nil, fmt.Errorf("%s: format %d (%d is read)", filepath.Join(root, RecordFile), r.Format, recordFormat)
	}
	if r.Programs == nil {
		r.Programs = map[string]*ProgramRecord{}
	}
	return &r, nil
}

// WriteRecord writes r in root (through a temporary file, renamed).
func WriteRecord(root string, r *Record) error {
	r.Format = recordFormat
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(root, "."+RecordFile+"-*")
	if err != nil {
		return err
	}
	if _, err := tmp.Write(append(b, '\n')); err != nil {
		tmp.Close()
		os.Remove(tmp.Name())
		return err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmp.Name())
		return err
	}
	os.Chmod(tmp.Name(), 0o644)
	return os.Rename(tmp.Name(), filepath.Join(root, RecordFile))
}

// ProgramKey is the program at asset's name in a record of its export: its path in it without
// .aimodel (vae_s0).
func ProgramKey(export, asset string) string {
	rel, err := filepath.Rel(export, asset)
	if err != nil {
		rel = filepath.Base(asset)
	}
	return strings.TrimSuffix(filepath.ToSlash(rel), ".aimodel")
}

// sourceHex is the asset's (or a prebuilt program's) main.hash in hex, "" when there is none.
func sourceHex(dir string) string {
	b, err := os.ReadFile(filepath.Join(dir, "main.hash"))
	if err != nil || len(b) == 0 {
		return ""
	}
	return hex.EncodeToString(b)
}

// metadata is what a prebuilt program's metadata.json says of how it was made.
type metadata struct {
	Producer     string `json:"producer"`
	CreationDate string `json:"creationDate"`
	SourceHash   string `json:"sourceHash"`
}

func readMetadata(out string) (metadata, error) {
	var m metadata
	b, err := os.ReadFile(filepath.Join(out, "metadata.json"))
	if err != nil {
		return m, err
	}
	return m, json.Unmarshal(b, &m)
}

// ToolName is coreai-build's name of itself in a prebuilt program's metadata.json (producer) for
// the version ("coreai-build-3605.6.3" for 3605.6.3).
func ToolName(version string) string { return "coreai-build-" + version }

// MarkDevice writes what a device made of the prebuilt programs of the export for arch whose
// names match one of programs (path.Match on ProgramKey; every one when none) into the record
// of their PackedRoot, and returns the names marked. A failure stays over an ok of another
// function (the program is bad when one of its functions fails); an ok of no function (a
// person's, of the whole program) replaces it. A program with no
// record yet (from before records) gets one of what its files say.
func MarkDevice(export, arch string, programs []string, d DeviceCheck) ([]string, error) {
	root := PackedRoot(export, arch)
	if _, err := os.Stat(root); err != nil {
		return nil, fmt.Errorf("%s: no prebuilt programs for %s here", filepath.Base(export), arch)
	}
	rec, err := ReadRecord(root)
	if err != nil {
		return nil, err
	}
	if rec == nil {
		rec = &Record{Arch: arch, Pack: PackFormat, Export: filepath.Base(export), Programs: map[string]*ProgramRecord{}}
	}
	var marked []string
	for _, p := range packedPrograms(root, arch) {
		key := p.key
		if len(programs) > 0 && !slices.ContainsFunc(programs, func(pat string) bool {
			ok, _ := path.Match(strings.TrimSuffix(pat, ".aimodel"), key)
			return ok
		}) {
			continue
		}
		e := rec.Programs[key]
		if e == nil || e.Source != p.source {
			e = legacyRecord(p.dir, p.source)
			rec.Programs[key] = e
		}
		if e.Device.Bad() && !d.Bad() && d.Function != "" && d.Function != e.Device.Function {
			continue // another function failed: the program stays bad
		}
		dc := d
		e.Device = &dc
		marked = append(marked, key)
	}
	if len(marked) == 0 {
		return nil, fmt.Errorf("%s: no prebuilt program for %s named %s", filepath.Base(export), arch, strings.Join(programs, " "))
	}
	return marked, WriteRecord(root, rec)
}

// legacyRecord is the record of a prebuilt program made before records: built, by the
// coreai-build its metadata.json names, the compiles at once not known.
func legacyRecord(dir, source string) *ProgramRecord {
	e := &ProgramRecord{Source: source, Result: ResultBuilt, Legacy: true}
	if m, err := readMetadata(dir); err == nil {
		e.Tool = m.Producer
		e.Built = m.CreationDate
	}
	return e
}

// packedProgram is a prebuilt program under a PackedRoot.
type packedProgram struct {
	key, dir, source string
}

// packedPrograms are the prebuilt programs for arch under root (a PackedRoot or its .prev),
// in the order of their names.
func packedPrograms(root, arch string) []packedProgram {
	suffix := "." + arch + ".aimodelc"
	var out []packedProgram
	filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil || !d.IsDir() || p == root {
			return nil
		}
		if strings.HasPrefix(d.Name(), ".") || strings.HasSuffix(d.Name(), ".part") {
			return filepath.SkipDir // a compile's or a pack's leftovers
		}
		if !strings.HasSuffix(d.Name(), ".aimodelc") {
			return nil
		}
		if strings.HasSuffix(d.Name(), suffix) {
			if src := sourceHex(p); src != "" {
				rel, _ := filepath.Rel(root, p)
				out = append(out, packedProgram{key: strings.TrimSuffix(filepath.ToSlash(rel), suffix), dir: p, source: src})
			}
		}
		return filepath.SkipDir
	})
	return out
}
