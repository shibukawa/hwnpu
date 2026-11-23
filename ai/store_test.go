package ai

import (
	"context"
	"testing"

	"github.com/shibukawa/hwnpu/internal/chat"
)

// A model prepared for a role under one use's name is ready for another use of the same model
// (hwnpu-audio's whisper-large-v3-turbo, interpreter's transcribe).
func TestPreparedUnderAnotherName(t *testing.T) {
	ownDefinitions(t)
	var f fakes
	role := f.role("transcribe", "", chat.KindTranscribe)
	ctx := context.Background()
	st := testStore(t)
	if err := NewApp("one", Define[*fakeModel](role, "whisper-large-v3-turbo")).withStore(st).Prepare(ctx, nil); err != nil {
		t.Fatal(err)
	}
	p, err := NewApp("two", Define[*fakeModel](role, "transcribe")).withStore(st).Plan(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !p.Ready() || !p.Compiled() || f.prepared != 1 {
		t.Errorf("the same model of another use: %+v (prepared %d)", p.Items, f.prepared)
	}
	// prepared under both names, compiled under one: compiled for the other too
	var g fakes
	r2 := g.role("transcribe", "", chat.KindTranscribe)
	st2 := testStore(t)
	two := NewApp("two", Define[*fakeModel](r2, "transcribe")).withStore(st2)
	if p, err := two.Plan(ctx); err != nil {
		t.Fatal(err)
	} else if p.NoCompile = true; p.Run(ctx, nil) != nil {
		t.Fatal("download")
	}
	if err := NewApp("one", Define[*fakeModel](r2, "whisper-large-v3-turbo")).withStore(st2).Prepare(ctx, nil); err != nil {
		t.Fatal(err)
	}
	if p, err := two.Plan(ctx); err != nil || !p.Compiled() {
		t.Errorf("compiled under the other name: %v, %+v", err, p)
	}
	// another role's record of the model is not
	other := f.role("other", "", chat.KindTranscribe)
	if p, err := NewApp("three", Define[*fakeModel](other, "transcribe")).withStore(st).Plan(ctx); err != nil || p.Ready() {
		t.Errorf("another role: %v, %+v", err, p)
	}
}
