package syncx

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
)

func TestSyncMap(t *testing.T) {
	var m SyncMap[string, int]
	if _, ok := m.Load("a"); ok {
		t.Fatal("unexpected key")
	}
	m.Store("a", 42)
	if v, ok := m.Load("a"); !ok || v != 42 {
		t.Fatalf("got %v, %v", v, ok)
	}
	if v, loaded := m.LoadOrStore("a", 99); !loaded || v != 42 {
		t.Fatalf("got %v, %v", v, loaded)
	}
	if v, loaded := m.LoadAndDelete("a"); !loaded || v != 42 {
		t.Fatalf("got %v, %v", v, loaded)
	}
	if _, ok := m.Load("a"); ok {
		t.Fatal("expected deleted")
	}
}

func TestPool(t *testing.T) {
	var p Pool[[]byte]
	if p.Get() != nil {
		t.Fatal("expected nil when New is nil")
	}
	s := make([]byte, 16)
	p.Put(&s)
	got := p.Get()
	if got == nil || len(*got) != 16 {
		t.Fatalf("unexpected pooled slice: %v", got)
	}

	p2 := Pool[int]{New: func() *int { v := 7; return &v }}
	if v := p2.Get(); v == nil || *v != 7 {
		t.Fatalf("unexpected New value: %v", v)
	}
}

func TestForEach(t *testing.T) {
	var sum atomic.Int64
	items := []int{1, 2, 3, 4, 5, 6}
	if err := ForEach(context.Background(), 2, items, func(_ context.Context, v int) error {
		sum.Add(int64(v))
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if sum.Load() != 21 {
		t.Fatalf("sum = %d, want 21", sum.Load())
	}

	wantErr := errors.New("boom")
	var ran atomic.Int64
	err := ForEachAll(context.Background(), 2, items, func(_ context.Context, v int) error {
		ran.Add(1)
		if v == 3 {
			return wantErr
		}
		return nil
	})
	if !errors.Is(err, wantErr) || ran.Load() != int64(len(items)) {
		t.Fatalf("err=%v ran=%d", err, ran.Load())
	}

	cctx, cancel := context.WithCancel(context.Background())
	cancel()
	ran.Store(0)
	if err := ForEachAll(cctx, 2, items, func(_ context.Context, _ int) error {
		ran.Add(1)
		return nil
	}); err != nil || ran.Load() != int64(len(items)) {
		t.Fatalf("canceled ForEachAll: err=%v ran=%d", err, ran.Load())
	}
}
