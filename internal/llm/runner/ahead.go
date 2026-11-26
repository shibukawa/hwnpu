package runner

import (
	"context"
	"fmt"

	"github.com/shibukawa/hwnpu"
)

// Ahead reports whether the programs that run in turn on dev load one ahead (InTurn, Ring): on
// Core AI, where a load is not a compile once its cache holds the program.
func Ahead(dev *hwnpu.Device) bool {
	return dev != nil && dev.Info().Backend == "coreai"
}

// Swap reports whether the groups of a large DiT's blocks load for each step on dev and close
// after it, rather than all being held: on Core AI, which fails to run more than the Neural Engine
// holds where Core ML evicts (Qwen-Image's 8 groups on an M1, docs/design/image-apple.md), and loads a group
// from its cache in 20 ms. Where they can all be held that is no faster: a group pays for
// following another group either way (docs/design/image-apple.md).
func Swap(dev *hwnpu.Device) bool {
	return dev != nil && dev.Info().Backend == "coreai"
}

// OpenFunc loads the program name (a part of an export) for InTurn and Ring.
type OpenFunc func(ctx context.Context, name string) (Program, error)

// InTurn calls f with each of the programs names in order, each loaded just before its turn and
// closed after it. With ahead the next one loads while f runs this one, so that two are held for
// that time: through Core AI a program its cache holds loads in a fraction of a second, which is
// then not waited for (Qwen-Image's 17 programs of a prompt on an M1: 7.9 -> 4.7 s, docs/design/image-apple.md). Through Core ML a load is a compile, which a run beside it does not shorten (docs/design/image-apple.md): there ahead stays false.
func InTurn(ctx context.Context, names []string, ahead bool, open OpenFunc, f func(i int, p Program) error) error {
	r := NewRing(names, ahead, open)
	defer r.Close()
	for i, name := range names {
		p, err := r.take(ctx, i, i+1 < len(names))
		if err != nil {
			return err
		}
		err = f(i, p)
		p.Close()
		if err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
	}
	return nil
}

// Ring loads programs that run in one order again and again (the groups of a DiT's blocks, each
// step), each for its turn alone: an NPU that cannot hold them all holds one, or with ahead two.
type Ring struct {
	names []string
	ahead bool
	open  OpenFunc
	next  chan loaded // the load under way, of names[at]; nil: none
	at    int
}

type loaded struct {
	p   Program
	err error
}

// NewRing is a Ring of the programs names, which open loads. With ahead Take starts loading the
// program after the one it gives.
func NewRing(names []string, ahead bool, open OpenFunc) *Ring {
	return &Ring{names: names, ahead: ahead, open: open}
}

// Take gives the program i loaded, for the caller to run and close. With ahead the one after it
// (the first, after the last) then loads while the caller runs this one; Close ends a load nobody
// took.
func (r *Ring) Take(ctx context.Context, i int) (Program, error) { return r.take(ctx, i, true) }

func (r *Ring) take(ctx context.Context, i int, more bool) (Program, error) {
	var l loaded
	if r.next != nil && r.at == i {
		l = <-r.next
		r.next = nil
		if l.err != nil && ctx.Err() == nil {
			// a load that failed beside a run is tried once more by itself, so that loading
			// ahead fails nothing that loading in turn would not
			l.p, l.err = r.open(ctx, r.names[i])
		}
	} else {
		r.Close() // a load ahead of another program
		l.p, l.err = r.open(ctx, r.names[i])
	}
	if l.err != nil {
		return nil, l.err
	}
	if r.ahead && more && len(r.names) > 1 {
		ch := make(chan loaded, 1)
		r.next, r.at = ch, (i+1)%len(r.names)
		name := r.names[r.at]
		go func() {
			p, err := r.open(ctx, name)
			ch <- loaded{p, err}
		}()
	}
	return l.p, nil
}

// Close waits for the load under way, if any, and closes its program.
func (r *Ring) Close() error {
	if r.next == nil {
		return nil
	}
	l := <-r.next
	r.next = nil
	if l.p != nil {
		return l.p.Close()
	}
	return nil
}
