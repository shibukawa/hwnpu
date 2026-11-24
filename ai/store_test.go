package ai

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/bonsaiai/internal/airole"
	"github.com/shibukawa/bonsaiai/internal/chat"
)

// A model prepared for a role under one use's name is ready for another use of the same model
// (bonsaiai-audio's whisper-large-v3-turbo, interpreter's transcribe).
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

// A store its role's Usable refuses (prepared for what this machine no longer has: ONNX
// Runtime for YuNet on the CPU) counts as not prepared, and the next Run prepares it again.
func TestUnusableIsNotPrepared(t *testing.T) {
	ownDefinitions(t)
	var f fakes
	role := f.role("transcribe", "", chat.KindTranscribe)
	ctx := context.Background()
	st := testStore(t)
	app := NewApp("one", Define[*fakeModel](role, "whisper-large-v3-turbo")).withStore(st)
	if err := app.Prepare(ctx, nil); err != nil {
		t.Fatal(err)
	}
	// what the first Prepare made does not run here any more; what Prepare makes now does
	gone := false
	role.Usable = func(airole.Files) error {
		if gone && f.prepared == 1 {
			return errors.New("gone")
		}
		return nil
	}
	if p, err := app.Plan(ctx); err != nil || !p.Ready() {
		t.Fatalf("usable: %v, %+v", err, p)
	}
	gone = true
	if p, err := app.Plan(ctx); err != nil || p.Ready() {
		t.Fatalf("unusable: %v, %+v", err, p)
	}
	if err := app.Prepare(ctx, nil); err != nil || f.prepared != 2 {
		t.Fatalf("prepared again: %v (prepared %d)", err, f.prepared)
	}
	if p, err := app.Plan(ctx); err != nil || !p.Ready() {
		t.Errorf("after preparing again: %v, %+v", err, p)
	}
}

// A tokenizer the sources gave as its binary file alone (tokenizer.tok for tokenizer.json) is
// there: a fresh cache's model prepared from them is not prepared again at every run.
func TestTokenizerBinaryIsThere(t *testing.T) {
	dir := t.TempDir()
	tok := filepath.Join(dir, "tokenizer.json")
	r := &record{}
	r.Files.Set("tokenizer", tok)
	if r.missing() != tok {
		t.Fatalf("missing() = %q with no tokenizer, want %q", r.missing(), tok)
	}
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.tok"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if m := r.missing(); m != "" {
		t.Errorf("missing() = %q with tokenizer.tok there, want none", m)
	}
}
