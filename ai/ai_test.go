package ai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/ai/flow"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/modelstore"
)

// fakeModel is what a fake role loads.
type fakeModel struct {
	name   string
	closed *int
	warmed *int
}

func (f *fakeModel) Close() error { *f.closed++; return nil }

// Warm counts the calls of Plan.Run's load (Warmer).
func (f *fakeModel) Warm(ctx context.Context) error { *f.warmed++; return nil }

// fakes counts what the fake roles did.
type fakes struct{ prepared, loads, closed, warmed int }

// role is a role of the kinds whose prepare writes a file in the model's directory and whose
// load checks that it is there.
func (f *fakes) role(name, feature string, kinds ...chat.Kind) *airole.Role {
	return &airole.Role{Name: name, Kinds: kinds, Feature: feature,
		Prepare: func(e airole.PrepareEnv) (files airole.Files, err error) {
			f.prepared++
			if err := os.MkdirAll(e.Dir, 0o755); err != nil {
				return files, err
			}
			p := filepath.Join(e.Dir, "program.xml")
			if err := os.WriteFile(p, []byte(e.Model.Name), 0o644); err != nil {
				return files, err
			}
			e.Log.Write([]byte("converting " + e.Model.Name + "\r\n"))
			files.Set("program", p)
			files.SetValue("format", e.NPU.Format)
			return files, nil
		},
		Load: func(ctx context.Context, files airole.Files, e airole.LoadEnv) (io.Closer, error) {
			f.loads++
			if _, err := os.Stat(files.Path("program")); err != nil {
				return nil, err
			}
			return &fakeModel{name: e.Model, closed: &f.closed, warmed: &f.warmed}, nil
		}}
}

// ownDefinitions keeps the machine's and the user's definition files, memory choices, model
// policy and accepted licenses out of a test, and makes the NPU Intel's whatever the machine
// has: its programs need no device to be planned for.
func ownDefinitions(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)
	t.Setenv("LOCALAPPDATA", dir)
	t.Setenv(chat.ModelsEnv, "")
	t.Setenv(chat.PolicyEnv, "")
	t.Setenv(chat.MemoryEnv, "")
	t.Setenv(chat.LicenseEnv, "")
	t.Setenv(llm.BackendEnv, llm.BackendIntel)
	saved := chat.SystemModelsFile
	chat.SystemModelsFile = func() string { return "" }
	t.Cleanup(func() {
		chat.SystemModelsFile = saved
		cache.SetRoot("")
		chat.LoadModels(nil, false, nil)
	})
}

// testStore is a store of a cache of its own, with no sources and no Hugging Face.
func testStore(t *testing.T) Store {
	off := false
	return Store{Models: modelstore.Options{Root: t.TempDir(), Sources: []string{}, HuggingFace: &off}}
}

func TestPlanRunOpen(t *testing.T) {
	ownDefinitions(t)
	var f fakes
	stt := Define[*fakeModel](f.role("transcribe", "", chat.KindTranscribe), "transcribe")
	tr := Define[*fakeModel](f.role("translate", "translation", chat.KindChat), "translate")
	tts := Define[*fakeModel](f.role("speak", "", chat.KindSpeech), "speech")
	ctx := context.Background()
	app := NewApp("test", stt, tr, stt).Memory(8)
	app.Store = testStore(t)

	p, err := app.Plan(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if p.Ready() || len(p.Items) != 2 || p.Target() != "intel" {
		t.Fatalf("plan of an empty store for %q: %+v", p.Target(), p.Items)
	}
	if p.Items[1].Model != "plamo-3-translate-0.8b" || p.Items[1].Role != "translate" || p.Items[1].Why == "" {
		t.Errorf("translate on 8 GB: %+v", p.Items[1])
	}
	if p.DownloadBytes() <= 0 || p.MemoryGB() <= 0 {
		t.Errorf("download %d bytes, %.1f GB running: the definitions have both", p.DownloadBytes(), p.MemoryGB())
	}
	if ls := p.Licenses(); len(ls) != 2 || !strings.HasPrefix(ls[0].URL, "https://huggingface.co/") || !ls[0].Accepted {
		t.Errorf("licenses %+v", ls)
	}

	// download: prepared, not loaded
	p.NoCompile = true
	var lines []string
	if err := p.Run(ctx, func(pr Progress) {
		if pr.Message != "" {
			lines = append(lines, pr.Message)
		}
	}); err != nil {
		t.Fatal(err)
	}
	if f.prepared != 2 || f.loads != 0 {
		t.Errorf("download: prepared %d, loaded %d, want 2 and 0", f.prepared, f.loads)
	}
	if len(lines) != 2 || lines[0] != "converting whisper-large-v3-turbo" {
		t.Errorf("progress lines %q", lines)
	}
	if !p.Ready() || p.Compiled() {
		t.Errorf("after the download: ready %v, compiled %v", p.Ready(), p.Compiled())
	}

	// the store says what was prepared, with paths relative to it
	raw, err := os.ReadFile(filepath.Join(app.Store.Models.Root, StoreFile))
	if err != nil {
		t.Fatal(err)
	}
	var m manifest
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	if r := m.Targets["intel"]["translate:translate"]; r == nil || r.Model != "plamo-3-translate-0.8b" ||
		r.Files.Path("program") != "models/plamo-3-translate-0.8b/program.xml" || r.Files.Values["format"] != "intel" {
		t.Errorf("store.json: %s", raw)
	}

	// prepare: what is ready is only loaded once, to compile
	p2, err := app.Plan(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !p2.Ready() || p2.Compiled() || p2.DownloadBytes() != 0 {
		t.Errorf("second plan: %+v", p2.Items)
	}
	var stages []Stage
	if err := p2.Run(ctx, func(pr Progress) { stages = append(stages, pr.Stage) }); err != nil {
		t.Fatal(err)
	}
	if f.prepared != 2 || f.loads != 2 || f.closed != 2 {
		t.Errorf("prepare: prepared %d, loaded %d, closed %d, want 2, 2 and 2", f.prepared, f.loads, f.closed)
	}
	if want := []Stage{StageCompile, StageDone, StageCompile, StageDone}; !slices.Equal(stages, want) {
		t.Errorf("stages %v, want %v", stages, want)
	}
	if f.warmed != 2 {
		t.Errorf("warmed %d times at the loads once, want 2", f.warmed)
	}
	if p3, err := app.Plan(ctx); err != nil || !p3.Compiled() {
		t.Errorf("third plan: %v, %+v", err, p3)
	}
	// a plan for 16 GB picks a larger translation model: to prepare
	if p, err := NewApp("big", tr).Memory(16).withStore(app.Store).Plan(ctx); err != nil || p.Ready() || p.Items[0].Model == "plamo-3-translate-0.8b" {
		t.Errorf("plan for 16 GB: %v, %+v", err, p)
	}
	// a role whose Prepare now makes more (its Revision) prepares again what an earlier one made
	stt.req.role.Revision = 1
	if p, err := app.Plan(ctx); err != nil || p.Ready() || p.Items[0].Ready || !p.Items[1].Ready {
		t.Errorf("after the role's revision: %v, %+v", err, p.Items)
	} else if err := p.Run(ctx, nil); err != nil {
		t.Error(err)
	} else if p, err := app.Plan(ctx); err != nil || !p.Ready() {
		t.Errorf("prepared again: %v, %+v", err, p.Items)
	}
	stt.req.role.Revision = 0

	// Plan.Log takes what the preparing writes, instead of a Progress.Message for each line
	if p, err := NewApp("logged", tr).withStore(testStore(t)).Plan(ctx); err != nil {
		t.Error(err)
	} else {
		var log strings.Builder
		messages := 0
		p.NoCompile, p.Log = true, &log
		err := p.Run(ctx, func(pr Progress) {
			if pr.Message != "" {
				messages++
			}
		})
		if err != nil || messages != 0 || !strings.Contains(log.String(), "converting ") {
			t.Errorf("with Plan.Log: %v, %d messages, log %q", err, messages, log.String())
		}
	}

	rt, err := app.Open(ctx)
	if err != nil {
		t.Fatal(err)
	}
	f.loads, f.closed = 0, 0
	a, err := stt.Get(ctx, rt)
	if err != nil {
		t.Fatal(err)
	}
	if b, _ := stt.Get(ctx, rt); a != b || f.loads != 1 || a.name != "whisper-large-v3-turbo" {
		t.Errorf("loaded %d times (%p, %p), want once", f.loads, a, b)
	}
	if _, err := tts.Get(ctx, rt); err == nil || !strings.Contains(err.Error(), "not one of test's models") {
		t.Errorf("a use the app did not register: %v", err)
	}
	if l := rt.Loaded(); len(l) != 1 || l[0].Use != "transcribe" || l[0].Model != "whisper-large-v3-turbo" || l[0].MemoryGB <= 0 {
		t.Errorf("loaded %+v", l)
	}
	// released and loaded again
	if err := stt.Release(rt); err != nil || f.closed != 1 || len(rt.Loaded()) != 0 {
		t.Errorf("release: %v, %d closed", err, f.closed)
	}
	if _, err := stt.Get(ctx, rt); err != nil || f.loads != 2 {
		t.Errorf("after the release: %v, %d loads", err, f.loads)
	}
	if _, err := tr.Get(ctx, rt); err != nil {
		t.Error(err)
	}
	if err := rt.Close(); err != nil || f.closed != 3 {
		t.Errorf("close: %v, %d closed", err, f.closed)
	}
	if _, err := stt.Get(ctx, rt); err == nil {
		t.Error("a closed runtime loaded a model")
	}
}

// withStore sets the app's store.
func (a *App) withStore(s Store) *App {
	a.Store = s
	return a
}

func TestNotPrepared(t *testing.T) {
	ownDefinitions(t)
	var f fakes
	stt := Define[*fakeModel](f.role("transcribe", "", chat.KindTranscribe), "transcribe")
	ctx := context.Background()
	app := NewApp("test", stt).withStore(testStore(t))

	rt, err := app.Open(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := stt.Get(ctx, rt); !errors.Is(err, ErrNotPrepared) || f.prepared != 0 {
		t.Errorf("a use not prepared: %v (prepared %d), want ErrNotPrepared", err, f.prepared)
	}
	rt.Close()

	// prepared without its compile ("models download"): still not run
	p, err := app.Plan(ctx)
	if err != nil {
		t.Fatal(err)
	}
	p.NoCompile = true
	if err := p.Run(ctx, nil); err != nil {
		t.Fatal(err)
	}
	rt, err = app.Open(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := stt.Get(ctx, rt); !errors.Is(err, ErrNotPrepared) || !strings.Contains(err.Error(), "not compiled") || f.loads != 0 {
		t.Errorf("a use not compiled: %v (loaded %d), want ErrNotPrepared", err, f.loads)
	}
	rt.Close()

	// prepared ("models prepare"): loaded once to compile it and warmed, then for the Get
	if err := app.Prepare(ctx, nil); err != nil {
		t.Fatal(err)
	}
	rt, err = app.Open(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer rt.Close()
	if _, err := stt.Get(ctx, rt); err != nil || f.prepared != 1 || f.loads != 2 || f.warmed != 1 {
		t.Errorf("prepared: %v, prepared %d, loaded %d, warmed %d", err, f.prepared, f.loads, f.warmed)
	}
}

func TestRoleAndNames(t *testing.T) {
	ownDefinitions(t)
	var f fakes
	ctx := context.Background()
	for name, role := range map[string]*airole.Role{
		"chat":          f.role("transcribe", "", chat.KindTranscribe),           // not a model of the role's kind
		"lfm3":          f.role("translate", "translation", chat.KindChat),       // without the role's feature
		"no-such-thing": f.role("transcribe", "", chat.KindTranscribe),           // not in the definitions
		"transcribe":    f.role("speak", "", chat.KindSpeech, chat.KindDecision), // a use of another kind
	} {
		app := NewApp("test", Define[*fakeModel](role, name)).Memory(8).withStore(testStore(t))
		if _, err := app.Plan(ctx); err == nil {
			t.Errorf("%s as %s: no error", name, role.Name)
		}
	}
	defer func() {
		if recover() == nil {
			t.Error("a Use no package made was registered")
		}
	}()
	NewApp("test", Use[*fakeModel]{})
}

func TestDefinitionsAndLicense(t *testing.T) {
	ownDefinitions(t)
	var f fakes
	ctx := context.Background()
	// the app's file over the built-in ones: its translate picks the 2B model on 8 GB machines
	// too, and it has a model of its own whose license asks to be accepted
	defs := filepath.Join(t.TempDir(), "models.yaml")
	if err := os.WriteFile(defs, []byte(`version: 1
models:
  my-whisper:
    kind: transcribe
    from: hf:example/my-whisper
    license: {id: my-license, url: "https://example.com/license", class: non-commercial, terms: "research only", accept: true}
    machine: 8
    bytes: 1e9
categories:
  translate:
    picks:
      - {machine: 8, model: plamo-3-translate-2b}
`), 0o644); err != nil {
		t.Fatal(err)
	}
	store := testStore(t)
	store.Models.Files = []string{defs}
	tr := Define[*fakeModel](f.role("translate", "translation", chat.KindChat), "translate")
	mine := Define[*fakeModel](f.role("transcribe", "", chat.KindTranscribe), "my-whisper")
	app := NewApp("test", tr, mine).Memory(8).withStore(store)
	p, err := app.Plan(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if p.Items[0].Model != "plamo-3-translate-2b" {
		t.Errorf("translate with the app's definitions: %s", p.Items[0].Model)
	}
	ls := p.Licenses()
	if len(ls) != 2 || ls[1].ID != "my-license" || !ls[1].Accept || ls[1].Accepted || ls[1].Class != "non-commercial" {
		t.Fatalf("licenses %+v", ls)
	}
	p.NoCompile = true
	var le *LicenseError
	if err := p.Run(ctx, nil); !errors.As(err, &le) || le.License.Model != "my-whisper" || f.prepared != 1 {
		t.Fatalf("a license not accepted: %v (prepared %d), want a LicenseError after the first model", err, f.prepared)
	}
	p.AcceptLicenses = true
	if err := p.Run(ctx, nil); err != nil || f.prepared != 2 {
		t.Fatalf("with the licenses accepted: %v, prepared %d", err, f.prepared)
	}
	// the acceptance is kept
	if p, err := app.Plan(ctx); err != nil || !p.Licenses()[1].Accepted {
		t.Errorf("the license after its acceptance: %v", err)
	}
}

// copyDir copies a directory tree.
func copyDir(t *testing.T, from, to string) {
	t.Helper()
	if err := os.CopyFS(to, os.DirFS(from)); err != nil {
		t.Fatal(err)
	}
}

func TestPreparedCache(t *testing.T) {
	ownDefinitions(t)
	var f fakes
	stt := Define[*fakeModel](f.role("transcribe", "", chat.KindTranscribe), "transcribe")
	tr := Define[*fakeModel](f.role("translate", "translation", chat.KindChat), "translate")
	ctx := context.Background()
	built := NewApp("build", stt).Memory(8).withStore(testStore(t))
	p, err := built.Plan(ctx)
	if err != nil {
		t.Fatal(err)
	}
	p.NoCompile = true
	if err := p.Run(ctx, nil); err != nil {
		t.Fatal(err)
	}
	// the prepared cache moved elsewhere is a prepared cache to ship: its paths are relative
	bundle := filepath.Join(t.TempDir(), "Resources", "prepared")
	copyDir(t, built.Store.Models.Root, bundle)
	store := testStore(t)
	store.Prepared = []string{bundle}
	app := NewApp("app", stt, tr).Memory(8).withStore(store)
	p2, err := app.Plan(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if it := p2.Items[0]; !it.Ready || it.Bundle != bundle || !it.Compiled {
		t.Errorf("plan with the prepared cache: %+v", it)
	}
	// what is in none is to prepare in the app's cache
	if p2.Items[1].Ready || p2.Ready() {
		t.Errorf("translate, in no prepared cache: %+v", p2.Items[1])
	}
	rt, err := app.Open(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer rt.Close()
	if _, err := stt.Get(ctx, rt); err != nil || f.prepared != 1 {
		t.Errorf("load from the prepared cache: %v (prepared %d)", err, f.prepared)
	}
	if _, err := os.Stat(filepath.Join(store.Models.Root, StoreFile)); err == nil {
		t.Error("loading from the prepared cache wrote this machine's store.json")
	}

	whisper, _ := chat.Lookup("whisper-large-v3-turbo")
	// a prepared cache for another NPU alone
	m, err := readManifest(bundle, false)
	if err != nil {
		t.Fatal(err)
	}
	m.Targets = map[string]map[string]*record{"some-other-npu": m.Targets["intel"]}
	if err := m.write(bundle); err != nil {
		t.Fatal(err)
	}
	if _, err := store.find("transcribe:transcribe", airole.NPU{Target: "intel"}, whisper, 0); !errors.Is(err, ErrBundleTarget) {
		t.Errorf("a prepared cache of another target: %v, want ErrBundleTarget", err)
	}
	// a file of it missing: not prepared
	m.Targets = map[string]map[string]*record{"intel": m.Targets["some-other-npu"]}
	m.write(bundle)
	os.Remove(filepath.Join(bundle, "models", "whisper-large-v3-turbo", "program.xml"))
	if _, err := store.find("transcribe:transcribe", airole.NPU{Target: "intel"}, whisper, 0); !errors.Is(err, ErrNotPrepared) {
		t.Errorf("a prepared cache with a file missing: %v, want ErrNotPrepared", err)
	}
	// a format this package does not read
	m.Format = storeFormat + 1
	m.write(bundle)
	if _, err := store.find("transcribe:transcribe", airole.NPU{Target: "intel"}, whisper, 0); !errors.Is(err, ErrBundleTarget) {
		t.Errorf("a prepared cache of a later format: %v, want ErrBundleTarget", err)
	}
}

func TestTargetsShareAStore(t *testing.T) {
	ownDefinitions(t)
	var f fakes
	stt := Define[*fakeModel](f.role("transcribe", "", chat.KindTranscribe), "transcribe")
	ctx := context.Background()
	app := NewApp("test", stt).Memory(8).withStore(testStore(t))
	// the same cache prepared for another NPU before (a home directory shared by two machines)
	old := &manifest{Format: storeFormat, Targets: map[string]map[string]*record{
		"coreai": {"transcribe:transcribe": {Model: "whisper-large-v3-turbo", Role: "transcribe"}}}}
	if err := old.write(app.Store.Models.Root); err != nil {
		t.Fatal(err)
	}
	p, err := app.Plan(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if p.Ready() {
		t.Fatal("what was prepared for another NPU counted as ready")
	}
	p.NoCompile = true
	if err := p.Run(ctx, nil); err != nil {
		t.Fatal(err)
	}
	m, err := readManifest(app.Store.Models.Root, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Targets) != 2 || m.Targets["coreai"]["transcribe:transcribe"] == nil || m.Targets["intel"]["transcribe:transcribe"] == nil {
		t.Errorf("store.json keeps both NPUs' uses: %+v", m.Targets)
	}
	// the programs of another variant of the model (as on a machine of other memory): to prepare
	m.Targets["intel"]["transcribe:transcribe"].Run.Context = 1234
	if err := m.write(app.Store.Models.Root); err != nil {
		t.Fatal(err)
	}
	if p, err := app.Plan(ctx); err != nil || p.Ready() {
		t.Errorf("a store of another variant: %v, ready %v", err, p != nil && p.Ready())
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

func TestLineUpByMemory(t *testing.T) {
	ownDefinitions(t)
	var f fakes
	role := f.role("translate", "translation", chat.KindChat)
	ctx := context.Background()
	store := testStore(t)
	model := func(u Use[*fakeModel], machine int) Item {
		t.Helper()
		p, err := NewApp("test", u).Memory(machine).withStore(store).Plan(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if p.MachineGB() != machine {
			t.Errorf("the plan is for %d GB, want %d", p.MachineGB(), machine)
		}
		return p.Items[0]
	}
	// alone, the use takes the largest model the machine runs; in this app the 2B from 16 GB and
	// the 8B only from 32 GB (the tiers given in any order)
	tr := Define[*fakeModel](role, "translate").At(32, "plamo3translate:8b").At(16, "plamo3translate:2b")
	for machine, want := range map[int]string{8: "plamo-3-translate-0.8b", 16: "plamo-3-translate-2b", 24: "plamo-3-translate-2b",
		32: "plamo-3-translate-8b", 64: "plamo-3-translate-8b"} {
		if it := model(tr, machine); it.Model != want || it.Use != "translate" {
			t.Errorf("on %d GB: %s as %s, want %s as translate", machine, it.Model, it.Use, want)
		}
	}
	if it := model(Define[*fakeModel](role, "translate"), 16); it.Model != "plamo-3-translate-8b" {
		t.Errorf("the use alone on 16 GB: %s, want the definitions' pick (the 8B)", it.Model)
	}
	if why := model(tr, 16).Why; !strings.Contains(why, "plamo3translate:2b on machines of 16 GB and up, by the app") {
		t.Errorf("why on 16 GB: %q", why)
	}
	// a tier set again replaces the one before; the first value is not changed
	if it := model(tr.At(16, "plamo3translate:0.8b"), 16); it.Model != "plamo-3-translate-0.8b" {
		t.Errorf("a tier set again: %s", it.Model)
	}
	if it := model(tr, 16); it.Model != "plamo-3-translate-2b" {
		t.Errorf("At changed the use it was called on: %s", it.Model)
	}
	// a helper kept small on every machine
	small := Define[*fakeModel](role, "translate").As(8)
	if it := model(small, 64); it.Model != "plamo-3-translate-0.8b" || !strings.Contains(it.Why, "picked as on a machine of 8 GB, by the app") {
		t.Errorf("as on 8 GB, on 64 GB: %s (%q)", it.Model, it.Why)
	}
	// a tier's name that the definitions do not have, or of another kind, fails the plan
	for _, name := range []string{"no-such-model", "whisper"} {
		if _, err := NewApp("test", tr.At(8, name)).Memory(8).withStore(store).Plan(ctx); err == nil {
			t.Errorf("a tier named %s: no error", name)
		}
	}

	// what the store holds prepared as the model of another memory is not loaded in its place
	if err := NewApp("small", tr).Memory(8).withStore(store).Prepare(ctx, nil); err != nil {
		t.Fatal(err)
	}
	stale, err := NewApp("large", tr).Memory(16).withStore(store).Open(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := tr.Get(ctx, stale); !errors.Is(err, ErrNotPrepared) || !strings.Contains(err.Error(), "prepared as plamo-3-translate-0.8b, and is plamo-3-translate-2b now") {
		t.Errorf("a store prepared for 8 GB, opened as 16 GB: %v, want ErrNotPrepared", err)
	}
	stale.Close()

	// the runtime prepares and loads the use as the app registered it, whatever value asks
	app := NewApp("test", tr).Memory(16).withStore(store)
	if err := app.Prepare(ctx, nil); err != nil {
		t.Fatal(err)
	}
	rt, err := app.Open(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer rt.Close()
	m, err := Define[*fakeModel](role, "translate").Get(ctx, rt)
	if err != nil || m.name != "plamo-3-translate-2b" {
		t.Errorf("loaded %v (%v), want the app's 2B", m, err)
	}
}

func TestUsableMemory(t *testing.T) {
	for machine, want := range map[int]float64{4: 0, 8: 3, 12: 6, 16: 9, 24: 15, 32: 22, 48: 35, 59: 45.3125, 64: 50, 128: 114} {
		if got := usableGB(machine); got != want {
			t.Errorf("a machine of %d GB has %v GB for models, want %v", machine, got, want)
		}
	}
	ownDefinitions(t)
	var f fakes
	stt := Define[*fakeModel](f.role("transcribe", "", chat.KindTranscribe), "transcribe")
	tr := Define[*fakeModel](f.role("translate", "translation", chat.KindChat), "translate")
	tts := Define[*fakeModel](f.role("speak", "", chat.KindSpeech), "speech")
	store := testStore(t)
	// on 16 GB the translator alone is the 8B: with the other two, over the 9 GB there is
	alone, err := NewApp("alone", stt, tr, tts).Memory(16).withStore(store).Plan(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if alone.UsableGB() != 9 || alone.Fits() {
		t.Errorf("the uses' own picks on 16 GB: %.1f GB of %.0f, fits %v; want over", alone.MemoryGB(), alone.UsableGB(), alone.Fits())
	}
	lined, err := NewApp("lined", stt, tr.At(16, "plamo3translate:2b"), tts).Memory(16).withStore(store).Plan(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if !lined.Fits() {
		t.Errorf("with the 2B on 16 GB: %.1f GB of %.0f, want it to fit", lined.MemoryGB(), lined.UsableGB())
	}
}

// TestNetwork: an app's models are those of its network's nodes; Run loads them before the
// sources are read, and a node's settings change while it runs.
func TestNetwork(t *testing.T) {
	ownDefinitions(t)
	var f fakes
	stt := Define[*fakeModel](f.role("transcribe", "", chat.KindTranscribe), "transcribe")
	tr := Define[*fakeModel](f.role("translate", "translation", chat.KindChat), "translate").At(16, "plamo3translate:2b")
	ctx := context.Background()

	loadsAtFirstRead := -1
	src := flow.Source[string]("mic").From(func(yield func(flow.Chunk[string], error) bool) {
		loadsAtFirstRead = f.loads
		for _, s := range []string{"a", "to:es", "b"} {
			if !yield(flow.Chunk[string]{Data: s}, nil) {
				return
			}
		}
	})
	hear := Node(stt, func(ctx context.Context, m *fakeModel, in flow.Stream[string]) flow.Stream[string] {
		return flow.Map(in, func(c flow.Chunk[string]) (string, error) { return c.Data, nil })
	})
	type options struct{ to string }
	trans := NewModelNode(tr, options{to: "en"}, func(ctx context.Context, m *fakeModel, in flow.Stream[string], o func() options) flow.Stream[string] {
		return flow.Map(in, func(c flow.Chunk[string]) (string, error) { return m.name + ":" + o().to + ":" + c.Data, nil })
	})
	// a node of the app's own between the two: a command changes the translator's settings
	commands := flow.Each("commands", func(c flow.Chunk[string], emit func(flow.Chunk[string])) error {
		if to, ok := strings.CutPrefix(c.Data, "to:"); ok {
			trans.Update(func(o *options) { o.to = to })
			return nil
		}
		emit(c)
		return nil
	})
	var got []string
	out := flow.Sink[string]("out").Do(func(c flow.Chunk[string]) error {
		got = append(got, c.Data)
		return nil
	})
	app := NewApp("net").Memory(16).withStore(testStore(t))
	app.From(src).To(hear).To(commands).To(trans).To(out)

	// the models of the nodes are the app's
	p, err := app.Plan(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Items) != 2 || p.Items[0].Use != "transcribe" || p.Items[1].Model != "plamo-3-translate-2b" {
		t.Fatalf("the plan of the network: %+v", p.Items)
	}
	var kinds []string
	for _, n := range app.Network() {
		kinds = append(kinds, n.Kind+" "+n.Name)
	}
	if want := []string{"source mic", "node transcribe", "node commands", "node translate", "sink out"}; !slices.Equal(kinds, want) {
		t.Errorf("the network %v, want %v", kinds, want)
	}

	// not prepared: the run fails before a source is read
	if err := app.Run(ctx); !errors.Is(err, ErrNotPrepared) || loadsAtFirstRead != -1 {
		t.Fatalf("a network of models not prepared: %v (the source read: %v)", err, loadsAtFirstRead != -1)
	}
	if err := app.Prepare(ctx, nil); err != nil {
		t.Fatal(err)
	}
	if err := app.Run(ctx); err != nil {
		t.Fatal(err)
	}
	if loadsAtFirstRead != 4 { // each loaded once to compile it, as "models prepare" does, then for the run
		t.Errorf("%d models were loaded when the source was first read, want both (twice)", loadsAtFirstRead)
	}
	if want := []string{"plamo-3-translate-2b:en:a", "plamo-3-translate-2b:es:b"}; !slices.Equal(got, want) {
		t.Errorf("the network gave %q, want %q", got, want)
	}
	if f.closed != f.loads {
		t.Errorf("%d models loaded, %d closed after the run", f.loads, f.closed)
	}

	// a model's node outside a run, and an app without a network
	lone := flow.Source[string]("lone").From(flow.From("x"))
	lone.To(Node(stt, func(ctx context.Context, m *fakeModel, in flow.Stream[string]) flow.Stream[string] { return in }))
	if err := flow.Run(ctx, lone); err == nil || !strings.Contains(err.Error(), "App.Run") {
		t.Errorf("a model's node run without its app: %v", err)
	}

	// a call: the same models once a request, each with a node of its own
	answered := NewApp("answers").Memory(16).withStore(app.Store)
	transOnce := NewModelNode(tr, options{to: "fr"}, func(ctx context.Context, m *fakeModel, in flow.Stream[string], o func() options) flow.Stream[string] {
		return flow.Map(in, func(c flow.Chunk[string]) (string, error) { return o().to + ":" + c.Data, nil })
	})
	count := flow.Each("count", func(c flow.Chunk[string], emit func(flow.Chunk[int])) error {
		emit(flow.Chunk[int]{Data: len(c.Data)})
		return nil
	})
	call := NewCall[string, int](answered, "length").Then(transOnce).Then(count)
	if p, err := answered.Plan(ctx); err != nil || len(p.Items) != 1 || p.Items[0].Use != "translate" || !p.Ready() {
		t.Fatalf("the plan of an app of a call: %v, %+v", err, p)
	}
	rt, err := answered.Open(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer rt.Close()
	before := f.loads
	for _, in := range []string{"abc", "hello"} {
		if n, err := call.Do(ctx, rt, in); err != nil || n != len("fr:")+len(in) {
			t.Errorf("the call on %q = %d, %v", in, n, err)
		}
	}
	if f.loads != before+1 {
		t.Errorf("the call's model was loaded %d times, want once", f.loads-before)
	}
	if all, err := call.All(ctx, rt, "x"); err != nil || !slices.Equal(all, []int{4}) {
		t.Errorf("All = %v, %v", all, err)
	}
	if err := NewApp("none", stt).withStore(testStore(t)).Run(ctx); err == nil || !strings.Contains(err.Error(), "no network") {
		t.Errorf("an app without a network: %v", err)
	}
}

// TestInTurns: of the uses an app runs one at a time the largest alone counts in what its models
// take together, and loading one releases the others of its group.
func TestInTurns(t *testing.T) {
	ownDefinitions(t)
	var f fakes
	stt := Define[*fakeModel](f.role("transcribe", "", chat.KindTranscribe), "transcribe")
	tr := Define[*fakeModel](f.role("translate", "translation", chat.KindChat), "translate")
	tts := Define[*fakeModel](f.role("speak", "", chat.KindSpeech), "speech")
	ctx := context.Background()
	store := testStore(t)
	together, err := NewApp("together", stt, tr, tts).Memory(16).withStore(store).Plan(ctx)
	if err != nil {
		t.Fatal(err)
	}
	mem := map[string]float64{}
	for _, it := range together.Items {
		mem[it.Use] = it.MemoryGB
	}
	app := NewApp("turns", stt).InTurns(tr, tts).Memory(16).withStore(store)
	p, err := app.Plan(ctx)
	if err != nil {
		t.Fatal(err)
	}
	want := mem["transcribe"] + max(mem["translate"], mem["speech"])
	if p.MemoryGB() != want || p.MemoryGB() >= together.MemoryGB() {
		t.Errorf("in turns: %.2f GB, want %.2f (all together %.2f)", p.MemoryGB(), want, together.MemoryGB())
	}
	if g := p.InTurns(); len(g) != 1 || strings.Join(g[0], ",") != "translate,speech" {
		t.Errorf("the plan's groups %v", g)
	}

	if err := p.Run(ctx, nil); err != nil {
		t.Fatal(err)
	}
	rt, err := app.Open(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer rt.Close()
	for _, u := range []Use[*fakeModel]{stt, tr} {
		if _, err := u.Get(ctx, rt); err != nil {
			t.Fatal(err)
		}
	}
	closed := f.closed
	if _, err := tts.Get(ctx, rt); err != nil {
		t.Fatal(err)
	}
	var uses []string
	for _, l := range rt.Loaded() {
		uses = append(uses, l.Use)
	}
	if f.closed != closed+1 || strings.Join(uses, ",") != "transcribe,speech" {
		t.Errorf("speech after translate: %v loaded, %d closed; want translate released, transcribe kept", uses, f.closed-closed)
	}
	defer func() {
		if recover() == nil {
			t.Error("a use in two groups of InTurns")
		}
	}()
	NewApp("twice").InTurns(stt, tr).InTurns(tr, tts)
}

// WithLoading tells of each load, with how long the one before took.
func TestWithLoading(t *testing.T) {
	ownDefinitions(t)
	var f fakes
	stt := Define[*fakeModel](f.role("transcribe", "", chat.KindTranscribe), "transcribe")
	ctx := context.Background()
	app := NewApp("test", stt).withStore(testStore(t))
	if err := app.Prepare(ctx, nil); err != nil {
		t.Fatal(err)
	}
	var seen []Progress
	watch := WithLoading(func(p Progress) { seen = append(seen, p) })
	// the first load after the compile: not known how long it takes; the next is
	for i, known := range []bool{false, true} {
		seen = nil
		rt, err := app.Open(ctx, watch)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := stt.Get(ctx, rt); err != nil {
			t.Fatal(err)
		}
		rt.Close()
		if len(seen) != 2 || seen[0].Stage != StageLoad || seen[1].Stage != StageDone || (seen[0].Expected > 0) != known || seen[0].Since.IsZero() {
			t.Errorf("Get %d: %+v, want a load (its time known: %v), then done", i+1, seen, known)
		}
	}
}

// A use compiled by another program (Core AI's cache is the process's) is not compiled for this
// one: it is not run until this program's Plan.Run compiles it (loads it once, warms it).
func TestCompiledByProgram(t *testing.T) {
	ownDefinitions(t)
	var f fakes
	stt := Define[*fakeModel](f.role("transcribe", "", chat.KindTranscribe), "transcribe")
	ctx := context.Background()
	app := NewApp("test", stt).withStore(testStore(t))
	saved := processName
	t.Cleanup(func() { processName = saved })
	processName = func() string { return "layers" }
	if err := app.Prepare(ctx, nil); err != nil {
		t.Fatal(err)
	}
	processName = func() string { return "parallax" }
	if p, err := app.Plan(ctx); err != nil || p.Compiled() {
		t.Errorf("the plan of another program: %v, compiled %v", err, p != nil && p.Compiled())
	}
	rt, err := app.Open(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := stt.Get(ctx, rt); !errors.Is(err, ErrNotPrepared) || !strings.Contains(err.Error(), "not compiled for parallax") {
		t.Errorf("a use compiled by another program: %v, want not compiled", err)
	}
	rt.Close()
	warmed := f.warmed
	if err := app.Prepare(ctx, nil); err != nil {
		t.Fatal(err)
	}
	rt, err = app.Open(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer rt.Close()
	if _, err := stt.Get(ctx, rt); err != nil || f.warmed != warmed+1 || f.prepared != 1 {
		t.Errorf("compiled by this program: %v, warmed %d more, prepared %d (once)", err, f.warmed-warmed, f.prepared)
	}
}
