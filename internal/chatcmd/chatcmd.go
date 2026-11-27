// Package chatcmd is what hwnpu-chat asks of hwnpud besides running its models, without the
// engine (docs/design/hwnpud.md, "Clients without the engine"): the list of the models, why a use
// or a family picks one, the memory a model is kept to run as, the check of the NPU, and the
// profile of the steps run. hwnpud's engine serves Host on its sessions (internal/aihost); with
// HWNPU_DAEMON=off they are this process's engine's (aieng.Chat), and without one they fail
// with aieng.ErrNone.
package chatcmd

import (
	"context"
	"fmt"
	"io"
	"reflect"
	"strings"
	"sync"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/internal/aieng"
	"github.com/shibukawa/hwnpu/internal/airpc"
	"github.com/shibukawa/hwnpu/internal/daemon"
)

// Host is what hwnpud's engine serves of hwnpu-chat on a session (internal/aihost). store is the
// client's (ai.App.Spec), whose definitions are read first.
type Host interface {
	// ChatList is aieng.ChatCommands.List as text.
	ChatList(ctx context.Context, store ai.StoreSpec, r ListRequest) (string, error)
	// ChatPicked is what aieng.ChatCommands.Picked writes.
	ChatPicked(ctx context.Context, store ai.StoreSpec, name string) (string, error)
	// ChatKeepMemory is what aieng.ChatCommands.KeepMemory writes.
	ChatKeepMemory(ctx context.Context, store ai.StoreSpec, model string, gb int) (string, error)
	// ChatDevice is aieng.ChatCommands.Device in hwnpud.
	ChatDevice(ctx context.Context, backend string) error
	// ChatProfile is the profile of the steps hwnpud ran (aieng.ChatCommands.Profile).
	ChatProfile(ctx context.Context) (string, error)
}

// ListRequest is a list asked for: the kinds of the models (all for none), full for hwnpu-chat
// list's, and the client's HWNPU_BACKEND.
type ListRequest struct {
	Kinds   []string
	Full    bool
	Backend string
}

// List writes the list r asks for to w.
func List(ctx context.Context, st ai.Store, w io.Writer, r ListRequest) error {
	return call(ctx, w, func(e aieng.ChatCommands, w io.Writer) error {
		e.List(ctx, w, r.Kinds, r.Full, r.Backend)
		return nil
	}, func(h *airpc.Handle, spec ai.StoreSpec) (string, error) {
		return airpc.Call1[string](h, ctx, "ChatList", spec, r)
	}, &st)
}

// Picked writes to log why the model name means was picked.
func Picked(ctx context.Context, st ai.Store, name string, log io.Writer) error {
	return call(ctx, log, func(e aieng.ChatCommands, w io.Writer) error {
		e.Picked(name, w)
		return nil
	}, func(h *airpc.Handle, spec ai.StoreSpec) (string, error) {
		return airpc.Call1[string](h, ctx, "ChatPicked", spec, name)
	}, &st)
}

// KeepMemory keeps gb as the memory to run model as (none for 0), and says so to log.
func KeepMemory(ctx context.Context, st ai.Store, model string, gb int, log io.Writer) error {
	if gb == 0 {
		return nil
	}
	return call(ctx, log, func(e aieng.ChatCommands, w io.Writer) error {
		e.KeepMemory(model, gb, w)
		return nil
	}, func(h *airpc.Handle, spec ai.StoreSpec) (string, error) {
		return airpc.Call1[string](h, ctx, "ChatKeepMemory", spec, model, gb)
	}, &st)
}

// Device checks that the NPU of the backend named is there, where the models run.
func Device(ctx context.Context, backend string) error {
	return call(ctx, io.Discard, func(e aieng.ChatCommands, _ io.Writer) error {
		return e.Device(ctx, backend)
	}, func(h *airpc.Handle, _ ai.StoreSpec) (string, error) {
		return "", airpc.Call0(h, ctx, "ChatDevice", backend)
	}, nil)
}

// Profile writes the profile of the steps run where the models run.
func Profile(ctx context.Context, w io.Writer) error {
	return call(ctx, w, func(e aieng.ChatCommands, w io.Writer) error {
		e.Profile(w)
		return nil
	}, func(h *airpc.Handle, _ ai.StoreSpec) (string, error) {
		return airpc.Call1[string](h, ctx, "ChatProfile")
	}, nil)
}

// call runs here in this process's engine (HWNPU_DAEMON=off, or auto when hwnpud does not
// answer), else remote in hwnpud with st's spec (none for nil), writing what it gives to w.
func call(ctx context.Context, w io.Writer, here func(aieng.ChatCommands, io.Writer) error,
	remote func(*airpc.Handle, ai.StoreSpec) (string, error), st *ai.Store) error {
	dc, err := hwnpud(ctx)
	if err != nil {
		return err
	}
	if dc == nil {
		if aieng.Chat == nil {
			return fmt.Errorf("hwnpu-chat: %w", aieng.ErrNone)
		}
		return here(aieng.Chat, w)
	}
	var spec ai.StoreSpec
	if st != nil {
		a := ai.NewApp("hwnpu-chat")
		a.Store = *st
		as, err := a.Spec()
		if err != nil {
			return err
		}
		spec = as.Store
	}
	sess, err := dc.DialAI(ctx)
	if err != nil {
		return err
	}
	defer sess.Close()
	out, err := remote(sess.Object(0).Bind(reflect.TypeFor[Host]()), spec)
	if err != nil {
		return olderHwnpud(dc, err)
	}
	_, err = io.WriteString(w, out)
	return err
}

// client is this process's client of hwnpud, asked once for a mode and a socket (daemon.Use; the
// ai package tells of hwnpud's version and settings, so this says nothing).
var client struct {
	sync.Mutex
	key string // the mode and the socket asked of ("": not asked)
	c   *daemon.Client
}

// hwnpud is the client of hwnpud, nil when the models run in this process.
func hwnpud(ctx context.Context) (*daemon.Client, error) {
	mode := daemon.Mode(daemon.DefaultMode())
	if mode == "off" {
		return nil, nil
	}
	client.Lock()
	defer client.Unlock()
	if key := mode + " " + daemon.SocketPath(); client.key != key {
		c, err := daemon.Use(ctx, daemon.DefaultMode(), io.Discard)
		if err != nil {
			return nil, err
		}
		client.key, client.c = key, c
	}
	return client.c, nil
}

// olderHwnpud is err, told as what to do when hwnpud is of a build before Host's methods.
func olderHwnpud(dc *daemon.Client, err error) error {
	if !strings.Contains(err.Error(), "has no method") {
		return err
	}
	return fmt.Errorf("%w: hwnpud (build %s) is older than this client (go install ./cmd/hwnpud && hwnpu service restart)", err, dc.Server.Build)
}
