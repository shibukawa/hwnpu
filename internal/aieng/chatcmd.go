package aieng

import (
	"context"
	"io"
)

// ChatCommands is what hwnpu-chat asks of the engine besides running its models
// (internal/chatcmd): the list of the models, why a use or a family picks one, the memory a model
// is kept to run as, the check of the NPU it runs on, and the profile of the steps run. The
// engine (internal/aiengine) sets Chat; hwnpud serves it to its clients (internal/aihost).
type ChatCommands interface {
	// List writes hwnpu-chat's list of the models of kinds (all for none); full adds what
	// hwnpu-chat list shows besides (the export each runs from on AMD's NPU, the features,
	// where the models are kept). backend is the client's HWNPU_BACKEND.
	List(ctx context.Context, w io.Writer, kinds []string, full bool, backend string)
	// Picked writes to log why the model name means was picked (a use or a family), nothing
	// for a model named.
	Picked(name string, log io.Writer)
	// KeepMemory keeps gb as the memory to run model as, and says so to log; nothing for 0 or
	// for the memory kept already. It fails no run.
	KeepMemory(model string, gb int, log io.Writer)
	// Device checks that the NPU of the backend named is here (Core AI: macOS 27).
	Device(ctx context.Context, backend string) error
	// Profile writes the profile of the steps run (HWNPU_PROFILE).
	Profile(w io.Writer)
}

// Chat is the engine's ChatCommands, nil without the engine.
var Chat ChatCommands
