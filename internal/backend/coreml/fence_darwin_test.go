package coreml

import (
	"context"
	"errors"
	"testing"
	"time"
)

// Acceptance requirement:model-execution-api/ctx-cancel.
func TestFenceWaitHonoursContext(t *testing.T) {
	f := &fence{done: make(chan struct{})}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	if err := f.Wait(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Wait = %v, want DeadlineExceeded", err)
	}
	if done, _ := f.Done(); done {
		t.Fatal("Done reported completion before the submission finished")
	}
	f.err = errors.New("boom")
	close(f.done)
	if err := f.Wait(context.Background()); err == nil || err.Error() != "boom" {
		t.Fatalf("Wait = %v, want boom", err)
	}
}

func TestStrides(t *testing.T) {
	got := contiguousStrides([]int{2, 3, 4})
	want := []int{12, 4, 1}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("strides %v, want %v", got, want)
		}
	}
	if !isContiguous([]int{1, 4}, []int{99, 1}) || isContiguous([]int{2, 4}, []int{1, 2}) {
		t.Fatal("isContiguous")
	}
}
