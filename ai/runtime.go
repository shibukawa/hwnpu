package ai

import (
	"context"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"
	"sync"
)

// Option is a setting of Open.
type Option func(*options)

type options struct {
	log io.Writer
	cpu bool
}

// WithLog sends what loading the models does (the programs, where they run) to w.
func WithLog(w io.Writer) Option { return func(o *options) { o.log = w } }

// WithCPU runs the models with the CPU interpreter instead of the NPU: slow, for tests and
// machines without an NPU.
func WithCPU() Option { return func(o *options) { o.cpu = true } }

// Runtime runs the models a store has prepared. It touches no network and converts nothing:
// a use the store has not prepared is ErrNotPrepared. Each model is loaded at its first use
// and stays until Close.
type Runtime struct {
	store  Store
	target string
	o      options

	mu     sync.Mutex
	loaded map[string]io.Closer
	closed bool
}

// Open opens the models of store. A bundle or a store prepared for another NPU's programs is
// ErrBundleTarget when one of its uses is asked for.
func Open(ctx context.Context, store Store, opts ...Option) (*Runtime, error) {
	if store.Dir == "" && len(store.Bundles) == 0 {
		return nil, errors.New("ai: a store needs a directory or bundles")
	}
	rt := &Runtime{store: store, loaded: map[string]io.Closer{}}
	for _, o := range opts {
		o(&rt.o)
	}
	if !rt.o.cpu {
		rt.target = hereTarget(ctx)
	}
	return rt, nil
}

// Uses are the uses the store has prepared, in its bundles and its directory.
func (rt *Runtime) Uses() []string {
	var out []string
	for _, root := range append(append([]string(nil), rt.store.Bundles...), rt.store.Dir) {
		if root == "" {
			continue
		}
		m, err := readManifest(root)
		if err != nil {
			continue
		}
		for u := range m.Uses {
			if !slices.Contains(out, u) {
				out = append(out, u)
			}
		}
	}
	slices.Sort(out)
	return out
}

// get loads use as role at its first call.
func (rt *Runtime) get(ctx context.Context, use string, want Role) (io.Closer, error) {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if rt.closed {
		return nil, errors.New("ai: the runtime is closed")
	}
	if l, ok := rt.loaded[use]; ok {
		return l, nil
	}
	loc, err := rt.store.find(use, rt.target)
	if err != nil {
		return nil, err
	}
	if loc.Role != want {
		return nil, fmt.Errorf("ai: %s is prepared as %s (%s), not %s", use, loc.Role, loc.Model, want)
	}
	l, err := roles[want].load(ctx, loc, loadEnv{cache: rt.store.cacheDir(), cpu: rt.o.cpu, log: rt.o.log})
	if err != nil {
		return nil, fmt.Errorf("ai: loading %s (%s): %w", use, loc.Model, err)
	}
	rt.loaded[use] = l
	return l, nil
}

// Transcriber is the transcriber of use (transcribe), loaded at the first call.
func (rt *Runtime) Transcriber(ctx context.Context, use string) (*Transcriber, error) {
	l, err := rt.get(ctx, use, RoleTranscribe)
	if err != nil {
		return nil, err
	}
	return l.(*Transcriber), nil
}

// Translator is the translator of use (translate), loaded at the first call.
func (rt *Runtime) Translator(ctx context.Context, use string) (*Translator, error) {
	l, err := rt.get(ctx, use, RoleTranslate)
	if err != nil {
		return nil, err
	}
	return l.(*Translator), nil
}

// Speaker is the speaker of use (speech), loaded at the first call.
func (rt *Runtime) Speaker(ctx context.Context, use string) (*Speaker, error) {
	l, err := rt.get(ctx, use, RoleSpeech)
	if err != nil {
		return nil, err
	}
	return l.(*Speaker), nil
}

// VAD is a detector of speech for use vad. It has no model: NewVAD gives one with other
// options.
func (rt *Runtime) VAD(use string) (*VAD, error) {
	if !strings.EqualFold(use, string(RoleVAD)) {
		return nil, fmt.Errorf("ai: %s is not a use of a VAD (vad is)", use)
	}
	return NewVAD(VADOptions{}), nil
}

// Close unloads the models.
func (rt *Runtime) Close() error {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if rt.closed {
		return nil
	}
	rt.closed = true
	var errs []error
	for _, l := range rt.loaded {
		errs = append(errs, l.Close())
	}
	rt.loaded = nil
	return errors.Join(errs...)
}
