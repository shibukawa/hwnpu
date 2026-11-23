package aihost

import (
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/airpc"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/daemon"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/modelstore"
)

// counts are what the fake role did in the engine's end.
var counts struct{ prepared, loads, closed, warmed, echoed atomic.Int64 }

// fakeModel is what the fake role loads: the model itself in the engine's end, a stub over
// the session (remote) in the app's.
type fakeModel struct {
	remote *airpc.Handle
	name   string
}

func (f *fakeModel) Name() string {
	if f.remote != nil {
		return airpc.Value[string](f.remote, "Name")
	}
	return f.name
}

// Echo tells fn each word of s and returns s in capitals.
func (f *fakeModel) Echo(ctx context.Context, s string, fn func(string) bool) (string, error) {
	if f.remote != nil {
		return airpc.Call1[string](f.remote, ctx, "Echo", s, fn)
	}
	counts.echoed.Add(1)
	for _, w := range strings.Fields(s) {
		if fn != nil && !fn(w) {
			break
		}
	}
	if s == "" {
		return "", ai.ErrNotPrepared
	}
	return strings.ToUpper(s), nil
}

func (f *fakeModel) Warm(ctx context.Context) error {
	if f.remote != nil {
		return airpc.Call0(f.remote, ctx, "Warm")
	}
	counts.warmed.Add(1)
	return nil
}

func (f *fakeModel) Close() error {
	if f.remote != nil {
		return f.remote.Release()
	}
	counts.closed.Add(1)
	return nil
}

var (
	fakeRole = &airole.Role{Name: "aihost-test", Kinds: []chat.Kind{chat.KindTranscribe},
		Prepare: func(e airole.PrepareEnv) (f airole.Files, err error) {
			counts.prepared.Add(1)
			if err := os.MkdirAll(e.Dir, 0o755); err != nil {
				return f, err
			}
			p := filepath.Join(e.Dir, "program.xml")
			if err := os.WriteFile(p, []byte(e.Model.Name), 0o644); err != nil {
				return f, err
			}
			e.Log.Write([]byte("converting " + e.Model.Name + "\n"))
			f.Set("program", p)
			return f, nil
		},
		Load: func(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
			counts.loads.Add(1)
			if _, err := os.Stat(f.Path("program")); err != nil {
				return nil, err
			}
			if e.Log != nil {
				io.WriteString(e.Log, "loading "+e.Model+"\n")
			}
			return &fakeModel{name: e.Model}, nil
		},
		Remote: func(h *airpc.Handle) io.Closer { return &fakeModel{remote: h} },
	}
	// localRole runs in the app's process alone (no Remote)
	localRole = &airole.Role{Name: "aihost-local", Kinds: []chat.Kind{chat.KindSpeech}}
)

func init() {
	airole.Register(fakeRole)
	airole.Register(localRole)
}

// serve runs an engine's routes of hwnpud for the ai package (hello and the sessions) on a socket
// of the test's, and points the app's end at it (HWNPU_DAEMON=on).
func serve(t *testing.T) *strings.Builder {
	t.Helper()
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)
	t.Setenv(chat.ModelsEnv, "")
	t.Setenv(chat.PolicyEnv, "")
	t.Setenv(chat.MemoryEnv, "")
	t.Setenv(chat.LicenseEnv, "")
	t.Setenv(llm.BackendEnv, llm.BackendIntel) // programs planned for Intel's NPU need no device
	saved := chat.SystemModelsFile
	chat.SystemModelsFile = func() string { return "" }
	t.Cleanup(func() {
		chat.SystemModelsFile = saved
		cache.SetRoot("")
		chat.LoadModels(nil, false, nil)
	})
	sock := filepath.Join(dir, "hwnpud.sock")
	ln, err := net.Listen("unix", sock)
	if err != nil {
		t.Fatal(err)
	}
	log := &strings.Builder{}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /hwnpud/v1/hello", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"protocol":1}`)
	})
	o := Options{}
	mux.Handle("GET "+daemon.AIPath, o.Handler())
	cache.SetInUse(o.UnitInUse)
	t.Cleanup(func() { cache.SetInUse(nil) })
	srv := &http.Server{Handler: mux}
	go srv.Serve(ln)
	t.Cleanup(func() { srv.Close() })
	t.Setenv(daemon.ModeEnv, "on")
	t.Setenv(daemon.SocketEnv, sock)
	return log
}

func testStore(t *testing.T) ai.Store {
	off := false
	return ai.Store{Models: modelstore.Options{Root: t.TempDir(), Sources: []string{}, HuggingFace: &off}}
}

func TestThroughHwnpud(t *testing.T) {
	serve(t)
	ctx := context.Background()
	use := ai.Define[*fakeModel](fakeRole, "transcribe")
	app := ai.NewApp("test", use).Memory(8)
	app.Store = testStore(t)

	p, err := app.Plan(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if p.Ready() || len(p.Items) != 1 || p.Target() != "intel" || p.Items[0].Model != "whisper-large-v3-turbo" || p.MachineGB() != 8 {
		t.Fatalf("plan through hwnpud for %q: %+v", p.Target(), p.Items)
	}
	if ls := p.Licenses(); len(ls) != 1 || ls[0].Model != "whisper-large-v3-turbo" {
		t.Errorf("licenses %+v", ls)
	}
	if counts.prepared.Load() != 0 {
		t.Error("a plan prepared")
	}

	// the run is hwnpud's: its stages and lines are told here, the models' logs written here
	var log strings.Builder
	var stages []ai.Stage
	var lines []string
	p.Log = &log
	if err := p.Run(ctx, func(pr ai.Progress) {
		if pr.Message != "" {
			lines = append(lines, pr.Message)
		} else {
			stages = append(stages, pr.Stage)
		}
	}); err != nil {
		t.Fatal(err)
	}
	if counts.prepared.Load() != 1 || counts.loads.Load() != 1 || counts.warmed.Load() != 1 || counts.closed.Load() != 1 {
		t.Errorf("run: prepared %d, loaded %d, warmed %d, closed %d", counts.prepared.Load(), counts.loads.Load(), counts.warmed.Load(), counts.closed.Load())
	}
	if len(stages) != 3 || stages[0] != ai.StagePrepare || stages[1] != ai.StageCompile || stages[2] != ai.StageDone {
		t.Errorf("stages %v", stages)
	}
	if len(lines) == 0 || lines[0] != "converting whisper-large-v3-turbo" {
		t.Errorf("lines %q", lines)
	}
	if !strings.Contains(log.String(), "converting whisper-large-v3-turbo") || !strings.Contains(log.String(), "compiling for the NPU") {
		t.Errorf("the run's log here: %q", log.String())
	}
	if !p.Ready() || !p.Compiled() {
		t.Errorf("after the run: %+v", p.Items)
	}
	if p, err := app.Plan(ctx); err != nil || !p.Compiled() {
		t.Errorf("the next plan: %v, %+v", err, p)
	}

	// the Runtime's models are hwnpud's; their methods are called from here
	var loadLog strings.Builder
	rt, err := app.Open(ctx, ai.WithLog(&loadLog))
	if err != nil {
		t.Fatal(err)
	}
	if rt.Target() != "intel" {
		t.Errorf("target %q", rt.Target())
	}
	m, err := use.Get(ctx, rt)
	if err != nil {
		t.Fatal(err)
	}
	if m.remote == nil || counts.loads.Load() != 2 {
		t.Fatalf("Get: a model here (%v), loads %d", m.remote == nil, counts.loads.Load())
	}
	if again, _ := use.Get(ctx, rt); again != m {
		t.Error("a second Get loaded again")
	}
	if ls := rt.Loaded(); len(ls) != 1 || ls[0].Model != "whisper-large-v3-turbo" {
		t.Errorf("loaded %+v", ls)
	}
	if !strings.Contains(loadLog.String(), "loading whisper-large-v3-turbo") {
		t.Errorf("the load's log here: %q", loadLog.String())
	}
	var words []string
	out, err := m.Echo(ctx, "a b c", func(w string) bool { words = append(words, w); return w != "b" })
	if err != nil || out != "A B C" || strings.Join(words, "") != "ab" {
		t.Errorf("Echo: %q %q %v", out, words, err)
	}
	if _, err := m.Echo(ctx, "", nil); !errors.Is(err, ai.ErrNotPrepared) {
		t.Errorf("a sentinel through hwnpud: %v", err)
	}
	if m.Name() != "whisper-large-v3-turbo" {
		t.Errorf("Name %q", m.Name())
	}
	// Release unloads it in hwnpud; the next Get loads it again
	if err := use.Release(rt); err != nil {
		t.Fatal(err)
	}
	if counts.closed.Load() != 2 {
		t.Errorf("release: closed %d", counts.closed.Load())
	}
	if _, err := m.Echo(ctx, "x", nil); err == nil {
		t.Error("a model released still answers")
	}
	if _, err := use.Get(ctx, rt); err != nil || counts.loads.Load() != 3 {
		t.Errorf("Get after the release: %v, loads %d", err, counts.loads.Load())
	}
	if err := rt.Close(); err != nil {
		t.Fatal(err)
	}
	if counts.closed.Load() != 3 {
		t.Errorf("close: closed %d", counts.closed.Load())
	}

	// a use not prepared is ErrNotPrepared as it is here; the app's definitions go to hwnpud
	d := ai.Definition{Like: "whisper-large-v3-turbo", Settings: map[string]string{"format": "lut4"}}
	other := ai.Define[*fakeModel](fakeRole, d.ID())
	app2 := ai.NewApp("other", other)
	app2.Store = app.Store
	app2.Define(d)
	if p, err := app2.Plan(ctx); err != nil || p.Ready() || p.Items[0].Model != d.ID() {
		t.Errorf("a plan of the app's definition: %v, %+v", err, p)
	}
	rt2, err := app2.Open(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer rt2.Close()
	if _, err := other.Get(ctx, rt2); !errors.Is(err, ai.ErrNotPrepared) {
		t.Errorf("a use not prepared: %v", err)
	}
	// a session that ends unloads its models
	if _, err := ai.NewApp("gone", use).Memory(8).Open(ctx); err != nil {
		t.Error(err)
	}
}

func TestRoleWithoutRemote(t *testing.T) {
	serve(t)
	ctx := context.Background()
	use := ai.Define[*fakeModel](localRole, "speech")
	app := ai.NewApp("local", use)
	app.Store = testStore(t)
	rt, err := app.Open(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer rt.Close()
	if _, err := use.Get(ctx, rt); err == nil || !strings.Contains(err.Error(), daemon.ModeEnv+"=off") {
		t.Errorf("a role without Remote: %v", err)
	}
}

// fakeEncoding is what a fake model's Encode gives: its state stays in hwnpud (a reference, as
// vision.PictureEncoding's).
type fakeEncoding struct {
	of     string
	Took   time.Duration
	remote *airpc.Handle
}

func init() {
	airpc.RegisterRef(func(e *fakeEncoding, h *airpc.Handle) { e.remote = h }, func(e *fakeEncoding) *airpc.Handle { return e.remote })
}

func (f *fakeModel) Encode(ctx context.Context, s string) (*fakeEncoding, error) {
	if f.remote != nil {
		return airpc.Call1[*fakeEncoding](f.remote, ctx, "Encode", s)
	}
	return &fakeEncoding{of: s, Took: time.Second}, nil
}

func (f *fakeModel) Decode(e *fakeEncoding, suffix string) (string, error) {
	if f.remote != nil {
		return airpc.Call1[string](f.remote, context.Background(), "Decode", e, suffix)
	}
	if e.of == "" {
		return "", errors.New("an encoding of no state")
	}
	return e.of + suffix, nil
}

// TestEngineState: a model's result that holds state of hwnpud's (Segmenter.Encode's) stays
// there behind a handle and is used there again.
func TestEngineState(t *testing.T) {
	serve(t)
	ctx := context.Background()
	use := ai.Define[*fakeModel](fakeRole, "transcribe")
	app := ai.NewApp("state", use).Memory(8)
	app.Store = testStore(t)
	if err := app.Prepare(ctx, nil); err != nil {
		t.Fatal(err)
	}
	rt, err := app.Open(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer rt.Close()
	m, err := use.Get(ctx, rt)
	if err != nil {
		t.Fatal(err)
	}
	e, err := m.Encode(ctx, "picture")
	if err != nil || e.remote == nil || e.of != "" || e.Took != time.Second {
		t.Fatalf("Encode: %+v, %v", e, err)
	}
	for _, s := range []string{" a", " b"} {
		if out, err := m.Decode(e, s); err != nil || out != "picture"+s {
			t.Errorf("Decode: %q, %v", out, err)
		}
	}
	if _, err := m.Decode(&fakeEncoding{}, "x"); err == nil {
		t.Error("an encoding of this process decoded with hwnpud's state")
	}
}
