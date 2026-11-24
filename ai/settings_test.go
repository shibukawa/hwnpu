package ai

import (
	"context"
	"io"
	"maps"
	"testing"

	"github.com/shibukawa/bonsaiai/internal/airole"
	"github.com/shibukawa/bonsaiai/internal/chat"
)

// LoadSettings reach the loads of the role they are for, the later names over the earlier, and
// no other role's; a plan's compile load has none.
func TestLoadSettings(t *testing.T) {
	ownDefinitions(t)
	var f fakes
	got := map[string]map[string]string{}
	role := func(name string, kind chat.Kind) *airole.Role {
		r := f.role(name, "", kind)
		load := r.Load
		r.Load = func(ctx context.Context, files airole.Files, e airole.LoadEnv) (io.Closer, error) {
			got[name] = maps.Clone(e.Settings)
			return load(ctx, files, e)
		}
		return r
	}
	stt := Define[*fakeModel](role("transcribe", chat.KindTranscribe), "transcribe")
	tts := Define[*fakeModel](role("speak", chat.KindSpeech), "speech")
	ctx := context.Background()
	app := NewApp("test", stt, tts).withStore(testStore(t))
	if err := app.Prepare(ctx, nil); err != nil {
		t.Fatal(err)
	}
	if got["transcribe"] != nil || got["speak"] != nil {
		t.Errorf("the compile loads had settings: %v", got)
	}
	rt, err := app.Open(ctx, LoadSettings("transcribe", map[string]string{"idle": "1m", "key": "a"}),
		LoadSettings("transcribe", map[string]string{"key": "b"}))
	if err != nil {
		t.Fatal(err)
	}
	defer rt.Close()
	if _, err := stt.Get(ctx, rt); err != nil {
		t.Fatal(err)
	}
	if _, err := tts.Get(ctx, rt); err != nil {
		t.Fatal(err)
	}
	if s := got["transcribe"]; s["idle"] != "1m" || s["key"] != "b" || len(s) != 2 {
		t.Errorf("transcribe's settings %v, want idle 1m and key b", s)
	}
	if s := got["speak"]; len(s) != 0 {
		t.Errorf("speak's settings %v, want none", s)
	}
}
