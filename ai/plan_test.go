package ai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type fakeModel struct{ closed *int }

func (f fakeModel) Close() error { *f.closed++; return nil }

// fakeRoles replaces the roles' preparing and loading for a test: prepare writes a file in the
// model's directory and records it, load checks it is there.
func fakeRoles(t *testing.T) (prepared, loads, closed *int) {
	prepared, loads, closed = new(int), new(int), new(int)
	saved := roles
	t.Cleanup(func() { roles = saved })
	prep := func(e prepareEnv) (*record, error) {
		*prepared++
		if err := os.MkdirAll(e.dir, 0o755); err != nil {
			return nil, err
		}
		p := filepath.Join(e.dir, "program.mlpackage")
		if err := os.WriteFile(p, []byte(e.model.Name), 0o644); err != nil {
			return nil, err
		}
		e.log.Write([]byte("converting " + e.model.Name + "\r\n"))
		return &record{Transcribe: &transcribeRecord{Dir: e.dir, Encoder: p}}, nil
	}
	load := func(ctx context.Context, l located, env loadEnv) (io.Closer, error) {
		*loads++
		if _, err := os.Stat(l.Transcribe.Encoder); err != nil {
			return nil, err
		}
		return fakeModel{closed}, nil
	}
	roles = map[Role]role{RoleTranscribe: {prep, load}, RoleTranslate: {prep, load}, RoleSpeech: {prep, load}}
	return
}

func TestResolve(t *testing.T) {
	for _, c := range []struct {
		name   string
		memory int
		model  string
		role   Role
	}{
		{"transcribe", 8, "whisper-large-v3-turbo", RoleTranscribe},
		{"translate", 8, "plamo-3-translate-0.8b", RoleTranslate},
		{"translate", 16, "plamo-3-translate-2b", RoleTranslate},
		{"plamo3translate:0.8b", 64, "plamo-3-translate-0.8b", RoleTranslate},
		{"speech", 8, "LFM3-Audio-2B-TTS", RoleSpeech},
		{"vad", 8, "", RoleVAD},
	} {
		r, err := resolve(c.name, c.memory)
		if err != nil {
			t.Errorf("%s at %d GB: %v", c.name, c.memory, err)
			continue
		}
		if r.model.Name != c.model || r.role != c.role {
			t.Errorf("%s at %d GB: %s as %s, want %s as %s", c.name, c.memory, r.model.Name, r.role, c.model, c.role)
		}
	}
	if _, err := resolve("chat", 8); err == nil {
		t.Errorf("chat: no error for a model the package does not run")
	}
	if _, err := resolve("no-such-thing", 8); err == nil {
		t.Errorf("no error for an unknown name")
	}
}

func TestPlanRunOpen(t *testing.T) {
	prepared, loads, closed := fakeRoles(t)
	ctx := context.Background()
	store := Store{Dir: t.TempDir()}
	reg := NewRegistry().Require("transcribe", "translate", "vad").Memory(8)

	p, err := reg.Plan(ctx, store)
	if err != nil {
		t.Fatal(err)
	}
	if p.Ready() || len(p.Items) != 3 || !p.Items[2].Ready {
		t.Fatalf("plan of an empty store: %+v", p.Items)
	}
	if p.DownloadBytes() <= 0 || p.MemoryGB() <= 0 {
		t.Errorf("download %d bytes, %.1f GB running: the definitions have both", p.DownloadBytes(), p.MemoryGB())
	}
	if ls := p.Licenses(); len(ls) != 2 || !strings.HasPrefix(ls[0].URL, "https://huggingface.co/") {
		t.Errorf("licenses %+v", ls)
	}
	var stages []Stage
	var lines []string
	if err := p.Run(ctx, func(pr Progress) {
		if pr.Message == "" {
			stages = append(stages, pr.Stage)
		} else {
			lines = append(lines, pr.Message)
		}
	}); err != nil {
		t.Fatal(err)
	}
	if *prepared != 2 {
		t.Errorf("prepared %d models, want 2", *prepared)
	}
	if len(lines) != 2 || lines[0] != "converting whisper-large-v3-turbo" {
		t.Errorf("progress lines %q", lines)
	}
	if stages[len(stages)-1] != StageDone {
		t.Errorf("stages %v", stages)
	}
	// the warm-up loads each once where there is an NPU
	if p.target != "" && (*loads != 2 || *closed != 2) {
		t.Errorf("warm-up: %d loads, %d closed, want 2 and 2", *loads, *closed)
	}

	// the store says what was prepared, with paths relative to it
	raw, err := os.ReadFile(filepath.Join(store.Dir, StoreFile))
	if err != nil {
		t.Fatal(err)
	}
	var m manifest
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	if r := m.Uses["translate"]; r == nil || r.Model != "plamo-3-translate-0.8b" || r.Transcribe.Encoder != "plamo-3-translate-0.8b/program.mlpackage" {
		t.Errorf("store.json: %s", raw)
	}

	// a second plan has nothing to do
	p2, err := reg.Plan(ctx, store)
	if err != nil {
		t.Fatal(err)
	}
	if !p2.Ready() || p2.DownloadBytes() != 0 {
		t.Errorf("second plan: %+v", p2.Items)
	}
	// a plan for 16 GB picks another translation model: to prepare
	p3, err := NewRegistry().Require("translate").Memory(16).Plan(ctx, store)
	if err != nil {
		t.Fatal(err)
	}
	if p3.Ready() || p3.Items[0].Model != "plamo-3-translate-2b" {
		t.Errorf("plan for 16 GB: %+v", p3.Items)
	}

	rt, err := Open(ctx, store)
	if err != nil {
		t.Fatal(err)
	}
	*loads = 0
	for range 2 {
		if _, err := rt.get(ctx, "transcribe", RoleTranscribe); err != nil {
			t.Fatal(err)
		}
	}
	if *loads != 1 {
		t.Errorf("loaded %d times, want once", *loads)
	}
	if _, err := rt.get(ctx, "translate", RoleSpeech); err == nil || !strings.Contains(err.Error(), "prepared as translate") {
		t.Errorf("translate as speech: %v", err)
	}
	if _, err := rt.Speaker(ctx, "speech"); !errors.Is(err, ErrNotPrepared) {
		t.Errorf("speech: %v, want ErrNotPrepared", err)
	}
	if _, err := rt.VAD("vad"); err != nil {
		t.Error(err)
	}
	if got := rt.Uses(); len(got) != 2 {
		t.Errorf("uses %v", got)
	}
	*closed = 0
	if err := rt.Close(); err != nil || *closed != 1 {
		t.Errorf("close: %v, %d closed", err, *closed)
	}
}

// copyDir copies a directory tree.
func copyDir(t *testing.T, from, to string) {
	t.Helper()
	err := filepath.WalkDir(from, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(from, p)
		dst := filepath.Join(to, rel)
		if d.IsDir() {
			return os.MkdirAll(dst, 0o755)
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		return os.WriteFile(dst, b, 0o644)
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestBundle(t *testing.T) {
	fakeRoles(t)
	ctx := context.Background()
	built := Store{Dir: t.TempDir()}
	p, err := NewRegistry().Require("transcribe").Memory(8).Plan(ctx, built)
	if err != nil {
		t.Fatal(err)
	}
	p.NoWarmUp = true
	if err := p.Run(ctx, nil); err != nil {
		t.Fatal(err)
	}
	// the prepared store moved elsewhere is a bundle: its paths are relative
	bundle := filepath.Join(t.TempDir(), "Resources", "models")
	copyDir(t, built.Dir, bundle)
	app := Store{Bundles: []string{bundle}}
	p2, err := NewRegistry().Require("transcribe").Memory(8).Plan(ctx, app)
	if err != nil {
		t.Fatal(err)
	}
	if !p2.Ready() || p2.Items[0].Bundle != bundle {
		t.Errorf("plan with the bundle: %+v", p2.Items)
	}
	rt, _ := Open(ctx, app)
	defer rt.Close()
	if _, err := rt.get(ctx, "transcribe", RoleTranscribe); err != nil {
		t.Errorf("load from the bundle: %v", err)
	}
	// what is in no bundle cannot be prepared without a directory
	if _, err := NewRegistry().Require("translate").Plan(ctx, app); !errors.Is(err, ErrNotPrepared) {
		t.Errorf("translate without a directory: %v, want ErrNotPrepared", err)
	}

	// a bundle for another NPU
	m, err := readManifest(bundle)
	if err != nil {
		t.Fatal(err)
	}
	m.Target = "some-other-npu"
	if err := m.write(bundle); err != nil {
		t.Fatal(err)
	}
	if _, err := app.find("transcribe", "apple"); !errors.Is(err, ErrBundleTarget) {
		t.Errorf("bundle of another target: %v, want ErrBundleTarget", err)
	}
	// a file of the bundle missing: not prepared
	m.Target = ""
	m.write(bundle)
	os.Remove(filepath.Join(bundle, "whisper-large-v3-turbo", "program.mlpackage"))
	if _, err := app.find("transcribe", "apple"); !errors.Is(err, ErrNotPrepared) {
		t.Errorf("bundle with a file missing: %v, want ErrNotPrepared", err)
	}
}

func TestLineWriter(t *testing.T) {
	var got []string
	w := &lineWriter{fn: func(s string) { got = append(got, s) }}
	w.Write([]byte("\x1b[90mdownloading a\x1b[0m 10%\rdownloading a 10%\rdownloading a 50%"))
	w.Write([]byte("\ndone\n\n"))
	want := []string{"downloading a 10%", "downloading a 50%", "done"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Errorf("lines %q, want %q", got, want)
	}
}
