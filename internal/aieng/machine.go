package aieng

import (
	"context"

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
