package llm

import (
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

// ExportDone reports whether the export (or program) whose mark is done is whole.
func ExportDone(done string) bool {
	_, ok := cache.ReadDone(done)
	if !ok {
		// a mark of an earlier hwnpu that holds nothing
		if fi, err := os.Stat(done); err == nil && fi.Size() == 0 {
			return true
		}
	}
	return ok
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

// StartProgram removes what an earlier conversion left of a program of ProgramPath and starts
// its export when it has one.
func StartProgram(program, export string) error {
	if export != "" {
		return StartExport(export)
	}
	return os.RemoveAll(program)
}

// FinishProgram marks a program of ProgramPath (NAME in dir) whole: its export with
// FinishExport (and the old Core AI program NAME.aimodel removed), or done written empty.
func FinishProgram(dir, name, done, export string) error {
	if export != "" {
		return FinishExport(export, done, OldCoreAIProgram(dir, name)...)
	}
	return os.WriteFile(done, nil, 0o644)
}
