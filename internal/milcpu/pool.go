package milcpu

import (
	"sync"
	"unsafe"
)

// pool keeps the float buffers of a Program's values for its next values, within a run and
// from one run to the next: a run of YuNet made 28 MB of new buffers, and the garbage collector
// took about 40 % of the CPU time (waking every core it may use) and their first touch the
// kernel's zeroing of pages. A buffer goes back when the last value holding it dies (a
// reshape's value shares its input's); one a value holds a part of (a slice into it) never goes
// back, nor the outputs the run returns.
type pool struct {
	free  map[int][][]float32 // by capacity
	refs  map[*float32]int    // the buffers handed out: the live values holding each
	inner map[*float32]bool   // handed out, and a value holds a part of it: not taken back
	// convs are the packed weights of the convolutions whose weights are constants, by their
	// weights' buffer
	convs map[*float32]*packedConv
	// spare are work areas (a padded plane, a packed block), lent to a goroutine of an
	// operation at a time; kept here rather than in a sync.Pool, which the collector empties
	smu   sync.Mutex
	spare []*[]float32
}

// work lends a work area of at least n floats, not cleared; give it back with done.
func (pl *pool) work(n int) *[]float32 {
	var b *[]float32
	if pl != nil {
		pl.smu.Lock()
		if k := len(pl.spare); k > 0 {
			b = pl.spare[k-1]
			pl.spare = pl.spare[:k-1]
		}
		pl.smu.Unlock()
	}
	if b == nil {
		b = new([]float32)
	}
	if cap(*b) < n {
		*b = make([]float32, n)
	}
	*b = (*b)[:n]
	return b
}

// done takes back a work area of work.
func (pl *pool) done(b *[]float32) {
	if pl == nil {
		return
	}
	pl.smu.Lock()
	pl.spare = append(pl.spare, b)
	pl.smu.Unlock()
}

// get is a buffer of n floats, not cleared: the operation writes every one.
func (pl *pool) get(n int) []float32 {
	if pl == nil || n == 0 {
		return make([]float32, n)
	}
	if pl.refs == nil {
		pl.free, pl.refs, pl.inner = map[int][][]float32{}, map[*float32]int{}, map[*float32]bool{}
	}
	var b []float32
	if fs := pl.free[n]; len(fs) > 0 {
		b = fs[len(fs)-1]
		pl.free[n] = fs[:len(fs)-1]
	} else {
		b = make([]float32, n)
	}
	pl.refs[unsafe.SliceData(b)] = 0
	return b
}

// zeros is get cleared.
func (pl *pool) zeros(n int) []float32 {
	b := pl.get(n)
	clear(b)
	return b
}

// hold counts a new value holding f, given the values it was computed from: f is a buffer of
// the pool, or a part of one of theirs, which is then not taken back.
func (pl *pool) hold(f []float32, from []*value) {
	if pl == nil || pl.refs == nil || len(f) == 0 {
		return
	}
	p := unsafe.SliceData(f)
	if _, ok := pl.refs[p]; ok {
		pl.refs[p]++
		return
	}
	lo := uintptr(unsafe.Pointer(p))
	for _, v := range from {
		if v == nil || cap(v.f) == 0 {
			continue
		}
		q := unsafe.SliceData(v.f)
		if _, ok := pl.refs[q]; !ok {
			continue
		}
		base := uintptr(unsafe.Pointer(q))
		if lo >= base && lo < base+uintptr(cap(v.f))*4 {
			pl.inner[q] = true
		}
	}
}

// drop counts a value holding f gone, taking the buffer back when it was the last.
func (pl *pool) drop(f []float32) {
	if pl == nil || pl.refs == nil || cap(f) == 0 {
		return
	}
	p := unsafe.SliceData(f)
	n, ok := pl.refs[p]
	if !ok {
		return
	}
	if n > 1 {
		pl.refs[p] = n - 1
		return
	}
	delete(pl.refs, p)
	if pl.inner[p] {
		delete(pl.inner, p)
		return
	}
	b := f[:cap(f)]
	pl.free[cap(f)] = append(pl.free[cap(f)], b)
}

// give hands f to the caller: no longer the pool's.
func (pl *pool) give(f []float32) {
	if pl == nil || pl.refs == nil || cap(f) == 0 {
		return
	}
	p := unsafe.SliceData(f)
	delete(pl.refs, p)
	delete(pl.inner, p)
}
