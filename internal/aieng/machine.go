package aieng

import (
	"context"
	"io"

	"github.com/shibukawa/hwnpu/internal/types"
)

// Machine is what the engine of this process tells of this machine beyond Engine, for hwnpu's
// commands that run in it (internal/cachecmd's, internal/npustatus' devices): hwnpud's engine, or
// a process that imports the engine with HWNPU_DAEMON=off.
type Machine interface {
	// RunFile is the file that keeps the memory chosen for each model ("" for none).
	RunFile() string
	// Enumerate opens the NPUs the engine finds (hwnpu.Enumerate): it wakes a sleeping one.
	Enumerate(ctx context.Context) ([]types.DeviceInfo, error)
	// List writes the commands' list of the models of kinds (all for none) of the definitions
	// read last: whether each is here, what the uses pick by memory, the policy, the sources.
	List(w io.Writer, kinds ...string)
}

// M is the machine of this process's engine, nil without one.
var M Machine

// GetMachine is M, or ErrNone without the engine.
func GetMachine() (Machine, error) {
	if M == nil {
		return nil, ErrNone
	}
	return M, nil
}
