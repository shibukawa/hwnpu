// Package cachecmd is bonsaiai's commands of the cache and of the memory a model runs as (bonsaiai
// cache, bonsaiai memory). They run in bonsaiaid's engine unless BONSAIAI_DAEMON is off
// (docs/design/bonsaiaid.md, Models and caches): bonsaiaid is then the only process that removes from
// the cache, it reads the cache of its own settings (BONSAIAI_CACHE, the sources), and it leaves the
// files of the models it holds loaded. The command's output and its question come back to the
// client's terminal.
package cachecmd

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"reflect"
	"strings"

	"github.com/shibukawa/bonsaiai/internal/airpc"
	"github.com/shibukawa/bonsaiai/internal/cache"
	"github.com/shibukawa/bonsaiai/internal/console"
	"github.com/shibukawa/bonsaiai/internal/daemon"
	"github.com/shibukawa/bonsaiai/modelstore"
)

// Env is where a command writes, how it asks, and what it leaves.
type Env struct {
	Ctx      context.Context
	Out, Err io.Writer
	// Ask asks a question of yes or no on the user's terminal; nil when there is none.
	Ask func(prompt string) bool
	// InUse says why a unit's files are in use (a model bonsaiaid holds loaded), "" when they are
	// not: those are listed as in use and not removed. nil for none.
	InUse func(x modelstore.Unit) string
}

// Request is a command as it is sent to bonsaiaid: its name and its flags.
type Request struct {
	Command string // info, ls, prune, clear, clean, memory
	Ls      LsCmd
	Prune   PruneCmd
	Clear   ClearCmd
	Clean   CleanCmd
	Memory  MemoryCmd
}

// Exec runs the command in this process.
func (r Request) Exec(e *Env) error {
	switch r.Command {
	case "info":
		return InfoCmd{}.exec(e)
	case "ls":
		return r.Ls.exec(e)
	case "prune":
		return r.Prune.exec(e)
	case "clear":
		return r.Clear.exec(e)
	case "clean":
		return r.Clean.exec(e)
	case "memory":
		return r.Memory.exec(e)
	}
	return fmt.Errorf("bonsaiai: no command %q of the cache", r.Command)
}

// Host is what bonsaiaid's engine serves of the cache on a session (internal/aihost).
type Host interface {
	// Machine runs a command of the cache or of the memory, its output and its errors' lines to
	// out and errOut, its question to ask (nil: none).
	Machine(ctx context.Context, r Request, out, errOut func([]byte), ask func(string) bool) error
	// CacheState is what bonsaiai status shows of the cache (its units read when scan).
	CacheState(ctx context.Context, scan bool) (State, error)
}

// State is the cache as bonsaiaid sees it.
type State struct {
	Root       string
	Layers     []string
	Free       int64
	Total      int64
	DiskErr    string
	Limits     cache.Limits
	LimitsErr  string
	Usage      cache.Usage
	UsageErr   string
	HeldModels []string // the models bonsaiaid holds loaded
}

// run runs r in bonsaiaid, or here with BONSAIAI_DAEMON=off.
func run(ctx context.Context, r Request) error {
	s, h, err := dial(ctx)
	if err != nil {
		return err
	}
	if s == nil {
		return r.Exec(&Env{Ctx: ctx, Out: os.Stdout, Err: os.Stderr, Ask: terminalAsk()})
	}
	defer s.Close()
	write := func(w io.Writer) func([]byte) { return func(b []byte) { w.Write(b) } }
	return airpc.Call0(h, ctx, "Machine", r, write(os.Stdout), write(os.Stderr), terminalAsk())
}

// dial opens a session with bonsaiaid (nil with BONSAIAI_DAEMON=off, or auto when it does not answer)
// and the handle of its Host.
func dial(ctx context.Context) (*airpc.Session, *airpc.Handle, error) {
	c, err := daemon.Use(ctx, daemon.DefaultMode(), console.Log(os.Stderr))
	if err != nil || c == nil {
		return nil, nil, err
	}
	s, err := c.DialAI(ctx)
	if err != nil {
		return nil, nil, err
	}
	return s, s.Object(0).Bind(reflect.TypeFor[Host]()), nil
}

// terminalAsk asks on standard output and reads the answer from standard input when it is a
// terminal; nil when it is not.
func terminalAsk() func(string) bool {
	if !console.Terminal(os.Stdin) {
		return nil
	}
	return func(prompt string) bool {
		fmt.Print(prompt)
		line, _ := bufio.NewReader(os.Stdin).ReadString('\n')
		a := strings.ToLower(strings.TrimSpace(line))
		return a == "y" || a == "yes"
	}
}

// notHeld are units without those in use (InUse), which it says are kept.
func (e *Env) notHeld(units []modelstore.Unit) []modelstore.Unit {
	if e.InUse == nil {
		return units
	}
	var out []modelstore.Unit
	for _, x := range units {
		if why := e.InUse(x); why != "" {
			fmt.Fprintf(e.Err, "%s: kept: %s (it can be removed once it is let go)\n", x.Name, why)
			continue
		}
		out = append(out, x)
	}
	return out
}

// held says which of the units listed are in use.
func (e *Env) held(units []modelstore.Unit) {
	if e.InUse == nil {
		return
	}
	for _, x := range units {
		if why := e.InUse(x); why != "" {
			fmt.Fprintf(e.Out, "in use: %s (%s)\n", x.Name, why)
		}
	}
}

// RemoteStore is what bonsaiai status reads of the cache, from bonsaiaid's (npustatus.StoreView):
// nil, nil with BONSAIAI_DAEMON=off. close ends its session.
func RemoteStore(ctx context.Context) (view *StoreView, close func(), err error) {
	s, h, err := dial(ctx)
	if err != nil || s == nil {
		return nil, func() {}, err
	}
	return &StoreView{ctx: ctx, h: h}, func() { s.Close() }, nil
}

// StoreView is bonsaiaid's cache as npustatus reads a store.
type StoreView struct {
	ctx context.Context
	h   *airpc.Handle
}

func (v *StoreView) state(scan bool) State {
	st, err := airpc.Call1[State](v.h, v.ctx, "CacheState", scan)
	if err != nil {
		st.DiskErr, st.LimitsErr, st.UsageErr = err.Error(), err.Error(), err.Error()
	}
	return st
}

// Root is the root of bonsaiaid's cache.
func (v *StoreView) Root() string { return v.state(false).Root }

func (v *StoreView) DiskSpace() (int64, int64, error) {
	st := v.state(false)
	if st.DiskErr != "" {
		return 0, 0, fmt.Errorf("bonsaiaid: %s", st.DiskErr)
	}
	return st.Free, st.Total, nil
}

func (v *StoreView) Limits() (cache.Limits, error) {
	st := v.state(false)
	if st.LimitsErr != "" {
		return cache.Limits{}, fmt.Errorf("bonsaiaid: %s", st.LimitsErr)
	}
	return st.Limits, nil
}

func (v *StoreView) Usage() (cache.Usage, error) {
	st := v.state(true)
	if st.UsageErr != "" {
		return cache.Usage{}, fmt.Errorf("bonsaiaid: %s", st.UsageErr)
	}
	return st.Usage, nil
}

func (v *StoreView) LayerNames() []string { return v.state(false).Layers }

// LocalState is State read in this process (bonsaiaid's engine): s's, with the models held.
func LocalState(s *modelstore.Store, layers []string, scan bool, held []string) State {
	st := State{Root: cache.Root(), Layers: layers, HeldModels: held}
	var err error
	if st.Free, st.Total, err = s.DiskSpace(); err != nil {
		st.DiskErr = err.Error()
	}
	if st.Limits, err = s.Limits(); err != nil {
		st.LimitsErr = err.Error()
	}
	if scan {
		if st.Usage, err = s.Usage(); err != nil {
			st.UsageErr = err.Error()
		}
	}
	return st
}
