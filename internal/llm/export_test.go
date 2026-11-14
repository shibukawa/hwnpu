package llm

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

// TestProgramPath puts Core AI's single program in an export NAME.coreai (model.aimodel) and
// removes the old NAME.aimodel when the export is finished; Core ML's and Intel's stay as they were.
func TestProgramPath(t *testing.T) {
	dir := t.TempDir()
	prog, done, export := ProgramPath(dir, "swiftf0-r1-fp16", BackendCoreAI)
	if want := filepath.Join(dir, "swiftf0-r1-fp16.coreai", "model.aimodel"); prog != want {
		t.Errorf("program %s, want %s", prog, want)
	}
	if done != export+".done" || filepath.Base(export) != "swiftf0-r1-fp16.coreai" {
		t.Errorf("export %s, done %s", export, done)
	}
	old := OldCoreAIProgram(dir, "swiftf0-r1-fp16")
	for _, o := range old {
		os.MkdirAll(filepath.Dir(o), 0o755)
		os.WriteFile(o, nil, 0o644)
	}
	if ExportDone(done) {
		t.Fatal("done before it was written")
	}
	if err := StartProgram(prog, export); err != nil {
		t.Fatal(err)
	}
	os.WriteFile(filepath.Join(export, "stale"), nil, 0o644)
	if err := StartProgram(prog, export); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(export, "stale")); err == nil {
		t.Error("StartProgram left what an earlier conversion wrote")
	}
	if err := FinishProgram(context.Background(), dir, "swiftf0-r1-fp16", done, export); err != nil {
		t.Fatal(err)
	}
	if !ExportDone(done) {
		t.Error("not done after FinishProgram")
	}
	for _, o := range old {
		if _, err := os.Stat(o); err == nil {
			t.Errorf("%s was not removed", o)
		}
	}

	prog, done, export = ProgramPath(dir, "rtmo-r1-fp16-640", BackendCoreML)
	if prog != filepath.Join(dir, "rtmo-r1-fp16-640.mlpackage") || done != prog+".done" || export != "" {
		t.Errorf("Core ML: %s %s %q", prog, done, export)
	}
	// a mark of an earlier hwnpu, empty, still says done
	os.WriteFile(done, nil, 0o644)
	if !ExportDone(done) {
		t.Error("an empty mark is not done")
	}
}
