package chat

import (
	"context"
	"fmt"
	"time"

	"github.com/shibukawa/hwnpu"
)

// An IRON model's device memory is pinned host memory that the process's resident size does
// not show (1.8 GB for Gemma 4 E2B in 4 bits, 9 GB in bf16), held while the model is loaded.
// With Config.Idle the engine gives it back after that long without a request: it sets the
// sequence in the cache aside first (putAside: the rows and states, in host memory, which the
// system may swap), releases the model (iron.Model.Release) and closes the device. The next
// request loads the model again (a second or two from the page cache) and goes on from the
// sequence set aside, or from the prompt store, as after any other sequence.
//
// Only an IRON model is released: the Neural Engine's and Intel's programs take from tens of
// seconds to minutes to load again.

// touch notes a request's end and starts the idle clock (e.mu held).
func (e *Engine) touch() {
	if e.idle <= 0 || e.ir == nil {
		return
	}
	e.lastUsed = time.Now()
	if e.idleTimer == nil {
		e.idleTimer = time.AfterFunc(e.idle, e.idleCheck)
	}
}

// idleCheck releases the model when no request came for e.idle, else waits for the rest.
func (e *Engine) idleCheck() {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.idleTimer = nil
	if e.closed || e.ir == nil || e.ir.Released() {
		return
	}
	if d := time.Since(e.lastUsed); d < e.idle {
		e.idleTimer = time.AfterFunc(e.idle-d, e.idleCheck)
		return
	}
	if err := e.release(); err != nil {
		e.logf("%s: releasing the NPU's memory: %v", e.Name, err)
	}
}

// release sets the sequence aside, releases the model and closes the device (e.mu held).
func (e *Engine) release() error {
	e.putAside()
	e.cur = seq{}
	e.sess.Reset()
	if s, ok := e.sess.(*ironSession); ok {
		clear(s.placed) // the towers' soft tokens are gone with the device memory
	}
	err := e.ir.Release()
	if e.dev != nil {
		e.dev.Close()
		e.dev = nil
	}
	e.logf("%s: released the NPU's memory after %s without a request (%d sequences kept on the host)",
		e.Name, e.idle.Round(time.Second), len(e.saved))
	return err
}

// Release gives the model's device memory back now, as Config.Idle does after a while; the
// next request loads it again. Only an IRON model is released; for the others it does nothing.
func (e *Engine) Release() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed || e.ir == nil || e.ir.Released() {
		return nil
	}
	return e.release()
}

// Released reports whether the model's device memory is released (Config.Idle, Release).
func (e *Engine) Released() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.ir != nil && e.ir.Released()
}

// reload loads a released model again (e.mu held).
func (e *Engine) reload(ctx context.Context) error {
	t0 := time.Now()
	dev, err := hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD))
	if err != nil {
		return fmt.Errorf("chat: %s: opening the NPU again: %w", e.Name, err)
	}
	if err := e.ir.Reload(ctx, dev); err != nil {
		dev.Close()
		return fmt.Errorf("chat: %s: loading again: %w", e.Name, err)
	}
	e.dev = dev
	e.logf("%s: loaded again in %.1f s", e.Name, time.Since(t0).Seconds())
	return nil
}
