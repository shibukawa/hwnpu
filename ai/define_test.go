package ai

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
)

func TestCatalogAndNormalize(t *testing.T) {
	ownDefinitions(t)
	var f fakes
	stt := Define[*fakeModel](f.role("transcribe", "", chat.KindTranscribe), "transcribe")
	cat, err := testStore(t).Catalog()
	if err != nil {
		t.Fatal(err)
	}
	if len(cat.OfKind("transcribe")) == 0 || len(cat.Uses) == 0 || len(cat.Files) == 0 {
		t.Fatalf("catalog: %d transcribe models, %d uses, files %q", len(cat.OfKind("transcribe")), len(cat.Uses), cat.Files)
	}
	// a use is made the model's name
	u, def, err := Normalize(cat, stt)
	if err != nil || def != nil || u.Name() != "whisper-large-v3-turbo" {
		t.Errorf("transcribe: %q, %v, %v", u.Name(), def, err)
	}
	// a name of none lists the role's models
	_, _, err = Normalize(cat, Define[*fakeModel](stt.req.role, "nosuch"))
	var none *NoModelError
	if !errors.As(err, &none) || len(none.Models) == 0 {
		t.Errorf("nosuch: %v", err)
	}
	// another kind's model is refused
	if _, _, err := Normalize(cat, Define[*fakeModel](stt.req.role, "speech")); err == nil || errors.As(err, &none) {
		t.Errorf("speech as transcribe: %v", err)
	}
	// a repository not defined is a definition of the role's kind
	u, def, err = Normalize(cat, Define[*fakeModel](stt.req.role, "someone/whisper-tiny@abc"))
	if err != nil || def == nil || def.From != "hf:someone/whisper-tiny@abc" || def.Kind != "transcribe" || u.Name() != def.ID() {
		t.Errorf("repository: %q, %+v, %v", u.Name(), def, err)
	}
}

func TestDefinitionPreparedApart(t *testing.T) {
	ownDefinitions(t)
	var f fakes
	var seen []airole.PrepareEnv
	role := f.role("transcribe", "", chat.KindTranscribe)
	prepare := role.Prepare
	role.Prepare = func(e airole.PrepareEnv) (airole.Files, error) {
		seen = append(seen, e)
		return prepare(e)
	}
	ctx := context.Background()
	st := testStore(t)
	prog := filepath.Join(t.TempDir(), "encoder.xml")
	if err := os.WriteFile(prog, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	d := Definition{Like: "whisper-large-v3-turbo", Device: "cpu", Settings: map[string]string{"format": "lut4"}, Programs: map[string][]string{"encoder": {prog}}}
	if d.ID() != (Definition{Like: d.Like, Device: "cpu", Settings: map[string]string{"format": "lut4"}, Programs: d.Programs}).ID() {
		t.Error("the same definition has two IDs")
	}
	if d.ID() == (Definition{Like: d.Like, Settings: map[string]string{"format": "lut6"}}).ID() {
		t.Error("two definitions have one ID")
	}
	base := Define[*fakeModel](role, "whisper-large-v3-turbo")
	mine := Define[*fakeModel](role, d.ID())
	app := NewApp("test", base, mine).Define(d)
	app.Store.Models = st.Models
	if err := app.Prepare(ctx, nil); err != nil {
		t.Fatal(err)
	}
	if len(seen) != 2 {
		t.Fatalf("prepared %d models, want the base and the definition apart", len(seen))
	}
	m, b := seen[1].Model, seen[1].Base
	if m.Name != d.ID() || b.Name != "whisper-large-v3-turbo" || m.Repo != b.Repo || m.Family != "" || b.Family == "" {
		t.Errorf("the definition %s (family %q) of %s (family %q)", m.Name, m.Family, b.Name, b.Family)
	}
	if m.Settings["format"] != "lut4" || m.Programs["encoder"][0] != prog || m.DeviceFor("amd") != "cpu" || m.DeviceFor("intel") != "cpu" {
		t.Errorf("the definition's settings %v, programs %v, device %v", m.Settings, m.Programs, m.Device)
	}
	// on the CPU there is nothing to compile, and a family never picks it
	p, err := app.Plan(ctx)
	if err != nil || !p.Compiled() {
		t.Fatalf("plan after Prepare: %v, %+v", err, p)
	}
	if got, _, _, _ := chat.Pick("whisper", 64); got.Name == d.ID() {
		t.Error("the family whisper picks the app's definition")
	}
	// another definition under the same use is prepared again
	app2 := NewApp("test", Define[*fakeModel](role, d.ID())).Define(Definition{Name: d.ID(), Like: d.Like, Settings: map[string]string{"format": "lut8"}})
	app2.Store.Models = st.Models
	if p, err := app2.Plan(ctx); err != nil || p.Ready() {
		t.Errorf("the same name with other settings is ready: %v", err)
	}
}

func TestCompileOnLoad(t *testing.T) {
	ownDefinitions(t)
	var f fakes
	stt := Define[*fakeModel](f.role("transcribe", "", chat.KindTranscribe), "transcribe")
	ctx := context.Background()
	app := NewApp("test", stt)
	app.Store = testStore(t)
	p, err := app.Plan(ctx)
	if err != nil {
		t.Fatal(err)
	}
	p.NoCompile, p.Quiet = true, true
	if err := p.Run(ctx, nil); err != nil {
		t.Fatal(err)
	}
	rt, err := app.Open(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := stt.Get(ctx, rt); !errors.Is(err, ErrNotPrepared) {
		t.Errorf("not compiled, without CompileOnLoad: %v", err)
	}
	rt.Close()
	if rt, err = app.Open(ctx, CompileOnLoad()); err != nil {
		t.Fatal(err)
	}
	if _, err := stt.Get(ctx, rt); err != nil {
		t.Fatal(err)
	}
	rt.Close()
	if f.loads != 1 {
		t.Errorf("loads %d, want 1: CompileOnLoad's load compiles it", f.loads)
	}
	if p, err := app.Plan(ctx); err != nil || !p.Compiled() {
		t.Errorf("compiled by the load: %v", err)
	}
}
