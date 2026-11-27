package cachecmd

import (
	"context"
	"io"
	"os"
	"reflect"

	"github.com/shibukawa/hwnpu/internal/aieng"
	"github.com/shibukawa/hwnpu/internal/airpc"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/daemon"
	"github.com/shibukawa/hwnpu/modelstore"
)

// ListHost is what hwnpud's engine serves of the commands' list of models (hwnpu-audio list,
// and the list a command shows when --model names none) on a session (internal/aihost): the
// definitions, whether each model is here, and what the uses pick are the engine's.
type ListHost interface {
	// ListModels writes the list r asks for to out, the definitions' log to errOut.
	ListModels(ctx context.Context, r ListRequest, out, errOut func([]byte)) error
}

// ListRequest is a list of the models as it is sent to hwnpud.
type ListRequest struct {
	// Kinds are the kinds of models listed (catalog.Kind's values), all for none.
	Kinds []string
	// Files are the definition files read over the built-in ones (a command's --models).
	Files []string
}

// Exec writes the list in this process (hwnpud's engine, or a process of the engine with
// HWNPU_DAEMON=off), the log of reading the definitions to log.
func (r ListRequest) Exec(w, log io.Writer) error {
	if log == nil {
		log = io.Discard
	}
	if _, err := modelstore.Open(modelstore.Options{Files: r.Files, Log: log}); err != nil {
		return err
	}
	m, err := aieng.GetMachine()
	if err != nil {
		return err
	}
	m.List(w, r.Kinds...)
	return nil
}

// List writes the models of kinds (all for none) that the definitions name, files read over
// the built-in ones, to w: hwnpud's list, or with HWNPU_DAEMON=off this process's.
func List(ctx context.Context, w io.Writer, files []string, kinds ...string) error {
	r := ListRequest{Kinds: kinds, Files: files}
	c, err := daemon.Use(ctx, daemon.DefaultMode(), console.Log(os.Stderr))
	if err != nil {
		return err
	}
	if c == nil {
		return r.Exec(w, os.Stderr)
	}
	s, err := c.DialAI(ctx)
	if err != nil {
		return err
	}
	defer s.Close()
	h := s.Object(0).Bind(reflect.TypeFor[ListHost]())
	return airpc.Call0(h, ctx, "ListModels", r, func(b []byte) { w.Write(b) }, func(b []byte) { os.Stderr.Write(b) })
}
