// Package syncx provides generic concurrency helpers: typed sync.Map and sync.Pool wrappers,
// and bounded parallel loops.
package syncx

import (
	"context"
	"sync"
)

// SyncMap is a type-safe wrapper around sync.Map.
type SyncMap[K comparable, V any] struct {
	m sync.Map
}

// Load returns the value stored in the map for a key, or the zero value of V if no value is present.
func (m *SyncMap[K, V]) Load(key K) (value V, ok bool) {
	v, ok := m.m.Load(key)
	if !ok || v == nil {
		return value, ok
	}
	return v.(V), true
}

// Store sets the value for a key.
func (m *SyncMap[K, V]) Store(key K, value V) {
	m.m.Store(key, value)
}

// LoadOrStore returns the existing value for the key if present. Otherwise, it stores and returns
// the given value. The loaded result is true if the value was loaded, false if stored.
func (m *SyncMap[K, V]) LoadOrStore(key K, value V) (actual V, loaded bool) {
	v, loaded := m.m.LoadOrStore(key, value)
	if v != nil {
		actual = v.(V)
	}
	return actual, loaded
}

// LoadAndDelete deletes the value for a key, returning the previous value if any.
func (m *SyncMap[K, V]) LoadAndDelete(key K) (value V, loaded bool) {
	v, loaded := m.m.LoadAndDelete(key)
	if !loaded || v == nil {
		return value, loaded
	}
	return v.(V), true
}

// Delete deletes the value for a key.
func (m *SyncMap[K, V]) Delete(key K) {
	m.m.Delete(key)
}

// Range calls f sequentially for each key and value present in the map. If f returns false,
// Range stops the iteration.
func (m *SyncMap[K, V]) Range(f func(key K, value V) bool) {
	m.m.Range(func(k, v any) bool {
		var val V
		if v != nil {
			val = v.(V)
		}
		return f(k.(K), val)
	})
}

// Pool is a type-safe wrapper around sync.Pool storing pointers *T to avoid interface boxing
// allocations on value types such as slices.
type Pool[T any] struct {
	New func() *T
	p   sync.Pool
}

// Get selects an arbitrary item from the Pool, removes it from the Pool, and returns it to the
// caller. When the pool is empty, Get calls p.New if non-nil, or returns nil otherwise.
func (p *Pool[T]) Get() *T {
	if v := p.p.Get(); v != nil {
		return v.(*T)
	}
	if p.New != nil {
		return p.New()
	}
	return nil
}

// Put adds x to the pool. Nil pointers are ignored.
func (p *Pool[T]) Put(x *T) {
	if x != nil {
		p.p.Put(x)
	}
}

// ForEach runs fn over items with at most limit concurrent goroutines, cancelling the derived
// context on the first non-nil error and returning that error.
func ForEach[T any](parent context.Context, limit int, items []T, fn func(context.Context, T) error) error {
	return ForEachIndex(parent, limit, len(items), func(ctx context.Context, i int) error {
		return fn(ctx, items[i])
	})
}

// ForEachIndex runs fn for indices 0..n-1 with at most limit concurrent goroutines, cancelling the
// derived context on the first non-nil error and returning that error.
func ForEachIndex(parent context.Context, limit, n int, fn func(context.Context, int) error) error {
	return runIndexed(parent, limit, n, true, fn)
}

// ForEachAll runs fn over all items with at most limit concurrent goroutines without cancelling
// siblings when one item fails, returning the first non-nil error encountered.
func ForEachAll[T any](parent context.Context, limit int, items []T, fn func(context.Context, T) error) error {
	return ForEachIndexAll(parent, limit, len(items), func(ctx context.Context, i int) error {
		return fn(ctx, items[i])
	})
}

// ForEachIndexAll runs fn for indices 0..n-1 with at most limit concurrent goroutines without
// cancelling siblings when one fails, returning the first non-nil error encountered.
func ForEachIndexAll(parent context.Context, limit, n int, fn func(context.Context, int) error) error {
	return runIndexed(parent, limit, n, false, fn)
}

func runIndexed(parent context.Context, limit, n int, cancelOnErr bool, fn func(context.Context, int) error) error {
	if n <= 0 {
		return nil
	}
	if limit <= 0 {
		limit = 1
	}
	if limit > n {
		limit = n
	}
	ctx := parent
	var cancel context.CancelFunc
	if cancelOnErr {
		ctx, cancel = context.WithCancel(parent)
		defer cancel()
	}
	sem := make(chan struct{}, limit)
	var (
		wg    sync.WaitGroup
		mu    sync.Mutex
		first error
	)
	recordErr := func(err error) {
		mu.Lock()
		if first == nil {
			first = err
			if cancel != nil {
				cancel()
			}
		}
		mu.Unlock()
	}
loop:
	for i := range n {
		if cancelOnErr {
			select {
			case sem <- struct{}{}:
			case <-ctx.Done():
				recordErr(ctx.Err())
				break loop
			}
			mu.Lock()
			done := first != nil
			mu.Unlock()
			if done {
				<-sem
				break loop
			}
		} else {
			sem <- struct{}{}
		}
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			defer func() { <-sem }()
			if err := fn(ctx, i); err != nil {
				recordErr(err)
			}
		}(i)
	}
	wg.Wait()
	return first
}
