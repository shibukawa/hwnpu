package airpc

import (
	"context"
	"io"
)

// Proxy dispatches calls on a client model either to a remote Handle in hwnpud or to a local
// engine implementation L in this process.
type Proxy[L io.Closer] struct {
	remote *Handle
	local  L
}

// LocalProxy wraps a local implementation L.
func LocalProxy[L io.Closer](local L) Proxy[L] { return Proxy[L]{local: local} }

// RemoteProxy wraps a remote Handle.
func RemoteProxy[L io.Closer](h *Handle) Proxy[L] { return Proxy[L]{remote: h} }

// Remote returns the remote Handle, or nil when the model runs in this process.
func (p Proxy[L]) Remote() *Handle { return p.remote }

// Local returns the local implementation when Remote is nil.
func (p Proxy[L]) Local() L { return p.local }

// Close releases the remote handle or closes the local implementation.
func (p Proxy[L]) Close() error {
	if p.remote != nil {
		return p.remote.Release()
	}
	return p.local.Close()
}

// ProxyValue calls method via Value[R] when p is remote, or calls local(p.Local()) otherwise.
func ProxyValue[L io.Closer, R any](p Proxy[L], method string, local func(L) R, args ...any) R {
	if p.remote != nil {
		return Value[R](p.remote, method, args...)
	}
	return local(p.local)
}

// ProxyCall0 calls method via Call0 when p is remote, or calls local(p.Local(), ctx) otherwise.
func ProxyCall0[L io.Closer](p Proxy[L], ctx context.Context, method string, local func(L, context.Context) error, args ...any) error {
	if p.remote != nil {
		return Call0(p.remote, ctx, method, args...)
	}
	return local(p.local, ctx)
}

// ProxyCall1 calls method via Call1[R] when p is remote, or calls local(p.Local()) otherwise.
func ProxyCall1[L io.Closer, R any](p Proxy[L], ctx context.Context, method string, local func(L) (R, error), args ...any) (R, error) {
	if p.remote != nil {
		return Call1[R](p.remote, ctx, method, args...)
	}
	return local(p.local)
}
