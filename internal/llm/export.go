package llm

import (
	"context"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu/internal/cache"
)

// Where a model's converted programs go in its directory. Core AI's are an export: a directory
// NAME.coreai beside the checkpoint, holding the programs (.aimodel) and what they need, and
// NAME.coreai.done beside it once it is whole (cache.WriteDone, holding the export's path), as
// every command's are (hwnpu-chat's, the image models', the audio models'). An export is what
// is converted at once: one that is cut off is converted again from nothing. Core ML's packages
// (what the CPU interpreter runs) and Intel's OpenVINO IR keep their own places.

// CoreAIExport is the export NAME.coreai in dir and its mark NAME.coreai.done.
func CoreAIExport(dir, name string) (export, done string) {
	export = filepath.Join(dir, name+".coreai")
	return export, export + ".done"
}

// ExportDone reports whether the export (or program) whose mark is done is whole: the mark is
// there (it holds the export's path, cache.WriteDone, or nothing: an earlier hwnpu's).
func ExportDone(done string) bool {
	_, err := os.Stat(done)
	return err == nil
}

// StartExport makes export an empty directory, what an earlier conversion cut off removed.
func StartExport(export string) error {
	if err := os.RemoveAll(export); err != nil {
		return err
	}
	return os.MkdirAll(export, 0o755)
}

// FinishExport marks export whole (done, written last) and removes old, the files and
// directories an earlier hwnpu kept the same programs in (a model's coreai/, NAME.aimodel and
// NAME.aimodel.done), which nothing reads any more.
func FinishExport(export, done string, old ...string) error {
	if err := cache.WriteDone(done, export); err != nil {
		return err
	}
	for _, o := range old {
		os.RemoveAll(o)
	}
	return nil
}

// Sharer fetches Core AI exports from the shared layers (docs/cache-sharing.md 7) and puts those
// made here in them.
type Sharer interface {
	// Fetch makes export from the first layer that has it and reports whether one had it.
	Fetch(ctx context.Context, export string) bool
	// Put puts export, made here and whole, in the layers that are written (in the background).
	Put(ctx context.Context, export string)
}

// SharedExports is the Sharer of the exports (chat sets it); nil: none is shared.
var SharedExports Sharer

// FetchExport reports whether the export whose mark is done is whole: here, or made from the
// shared layers (SharedExports), its mark then written.
func FetchExport(ctx context.Context, export, done string) bool {
	if ExportDone(done) {
		return true
	}
	if SharedExports == nil || !SharedExports.Fetch(ctx, export) {
		return false
	}
	return cache.WriteDone(done, export) == nil
}

// ShareExport puts the export made here (FinishExport) in the shared layers that are written.
func ShareExport(ctx context.Context, export string) {
	if SharedExports != nil {
		SharedExports.Put(ctx, export)
	}
}

// ProgramPath is where the single program NAME of a model goes in dir for backend, the mark that
// says it is whole, and the export to start before writing it ("" when there is none): Core AI's
// is model.aimodel in the export NAME.coreai (CoreAIExport); Core ML's NAME.mlpackage and
// Intel's NAME.xml sit in dir with NAME.EXT.done.
func ProgramPath(dir, name, backend string) (program, done, export string) {
	if backend == BackendCoreAI {
		export, done = CoreAIExport(dir, name)
		return filepath.Join(export, "model"+ProgramExt(backend)), done, export
	}
	program = filepath.Join(dir, name+ProgramExt(backend))
	return program, program + ".done", ""
}

// OldCoreAIProgram is where an earlier hwnpu kept the single Core AI program NAME in dir
// (NAME.aimodel and its mark), for FinishExport to remove.
func OldCoreAIProgram(dir, name string) []string {
	p := filepath.Join(dir, name+ProgramExt(BackendCoreAI))
	return []string{p, p + ".done"}
}

// ProgramReady reports whether a program of ProgramPath (NAME in dir) is whole: its Core AI
// export here or from the shared layers (FetchExport; the old program NAME.aimodel then
// removed), Core ML's and Intel's program with its .done.
func ProgramReady(ctx context.Context, dir, name, done, export string) bool {
	if export == "" {
		return ExportDone(done)
	}
	if !FetchExport(ctx, export, done) {
		return false
	}
	for _, o := range OldCoreAIProgram(dir, name) {
		os.RemoveAll(o)
	}
	return true
}

// StartProgram removes what an earlier conversion left of a program of ProgramPath and starts
// its export when it has one.
func StartProgram(program, export string) error {
	if export != "" {
		return StartExport(export)
	}
	return os.RemoveAll(program)
}

// FinishProgram marks a program of ProgramPath (NAME in dir) whole: its export with
// FinishExport (and the old Core AI program NAME.aimodel removed), put in the shared layers
// (ShareExport), or done written empty.
func FinishProgram(ctx context.Context, dir, name, done, export string) error {
	if export != "" {
		if err := FinishExport(export, done, OldCoreAIProgram(dir, name)...); err != nil {
			return err
		}
		ShareExport(ctx, export)
		return nil
	}
	return os.WriteFile(done, nil, 0o644)
}
