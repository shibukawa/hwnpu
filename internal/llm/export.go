package llm

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"

	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/syncx"
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
	if _, err := os.Stat(done); err != nil {
		return false
	}
	if StaleExport != nil {
		if export, ok := cache.ReadDone(done); ok && StaleExport(export) {
			return false
		}
	}
	return true
}

// StaleExport, when set (internal/chat), reports whether the whole export (a directory) is of
// another version than its model's recipe makes now (model-recipes.yaml apple.version): made
// again, or fetched, as one missing. Its name stays when only the recipe's version is raised.
var StaleExport func(export string) bool

// StartExport makes export an empty directory, what an earlier conversion cut off removed.
func StartExport(ctx context.Context, export string) error {
	if err := MayConvert(ctx, export); err != nil {
		return err
	}
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

// Sharer fetches Core AI exports from the shared layers (docs/cache-sharing-design.md 7) and puts those
// made here in them.
type Sharer interface {
	// Fetch makes export from the first layer that has it and reports whether one had it.
	Fetch(ctx context.Context, export string) bool
	// Put puts export, made here and whole, in the layers that are written (in the background).
	Put(ctx context.Context, export string)
}

type fetchOnlyKey struct{}

// WithFetchOnly is ctx in which nothing is converted: what is neither here nor in the shared
// layers is not made (MayConvert), while the files a conversion would read are still fetched
// (hwnpu-dev models fetch, which brings a model here and to the sources written back to, to
// try conversions on it after).
func WithFetchOnly(ctx context.Context) context.Context {
	return context.WithValue(ctx, fetchOnlyKey{}, true)
}

// FetchOnly reports whether ctx converts nothing (WithFetchOnly).
func FetchOnly(ctx context.Context) bool { b, _ := ctx.Value(fetchOnlyKey{}).(bool); return b }

// NotFetchedError is what MayConvert returns in a ctx that converts nothing: What (a path or a
// name) is neither here nor in the shared layers.
type NotFetchedError struct{ What string }

func (e *NotFetchedError) Error() string {
	return filepath.Base(e.What) + " is neither here nor in the sources, and nothing is converted now (hwnpu-dev models package makes it)"
}

// MayConvert is called where a conversion would start, once what it makes was looked for here
// and in the shared layers: nil, the error of fetching it when the cache had no room for it
// (FetchFailed), or a *NotFetchedError in a ctx that converts nothing.
func MayConvert(ctx context.Context, what string) error {
	if err := FetchFailed(what); err != nil {
		return err
	}
	if FetchOnly(ctx) {
		return &NotFetchedError{What: what}
	}
	return nil
}

// noRoom holds, by export, the error of the last fetch from the shared layers the cache had no
// room for (cache.ErrNoRoom).
var noRoom syncx.SyncMap[string, error]

// NoteFetch is told of the end of a fetch of export from the shared layers (nil: it was made).
// One the cache has no room for is kept for FetchFailed: converting it here would need the
// checkpoint, more room still, so it is not tried.
func NoteFetch(export string, err error) {
	if errors.Is(err, cache.ErrNoRoom) {
		// of the errors of each layer, the one without room alone
		if j, ok := err.(interface{ Unwrap() []error }); ok {
			for _, e := range j.Unwrap() {
				if errors.Is(e, cache.ErrNoRoom) {
					err = e
					break
				}
			}
		}
		noRoom.Store(filepath.Clean(export), err)
	} else {
		noRoom.Delete(filepath.Clean(export))
	}
}

// FetchFailed is the error of the last fetch of export from the shared layers when the cache
// had no room for it (NoteFetch), else nil: what is to download the checkpoint and convert it
// returns it instead.
func FetchFailed(export string) error {
	if err, ok := noRoom.Load(filepath.Clean(export)); ok {
		return err
	}
	return nil
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
// removed), Core ML's package here or from the shared layers (the CPU's program, in place of the
// ONNX file it is converted from), Intel's program with its .done.
func ProgramReady(ctx context.Context, dir, name, done, export string) bool {
	if export == "" {
		if pkg := strings.TrimSuffix(done, ".done"); filepath.Ext(pkg) == ".mlpackage" {
			return FetchExport(ctx, pkg, done)
		}
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
func StartProgram(ctx context.Context, program, export string) error {
	if export != "" {
		return StartExport(ctx, export)
	}
	if err := MayConvert(ctx, program); err != nil {
		return err
	}
	return os.RemoveAll(program)
}

// DropStaleCoreML removes the Core ML package program (and its mark done) when it is whole but
// lacks the part file part (one of before its model wrote it beside the program): it is then
// taken from the shared layers or converted again, with it.
func DropStaleCoreML(program, done, part string) {
	if filepath.Ext(program) != ".mlpackage" || !ExportDone(done) {
		return
	}
	if _, err := os.Stat(part); err == nil {
		return
	}
	os.RemoveAll(program)
	os.Remove(done)
}

// FinishProgram marks a program of ProgramPath (NAME in dir) whole: its export with
// FinishExport (and the old Core AI program NAME.aimodel removed), put in the shared layers
// (ShareExport), or done written empty; a Core ML package is put in the shared layers too.
func FinishProgram(ctx context.Context, dir, name, done, export string) error {
	if export != "" {
		if err := FinishExport(export, done, OldCoreAIProgram(dir, name)...); err != nil {
			return err
		}
		ShareExport(ctx, export)
		return nil
	}
	if err := os.WriteFile(done, nil, 0o644); err != nil {
		return err
	}
	if pkg := strings.TrimSuffix(done, ".done"); filepath.Ext(pkg) == ".mlpackage" {
		ShareExport(ctx, pkg)
	}
	return nil
}
