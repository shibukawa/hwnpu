package aihost

import (
	"context"
	"strings"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/internal/aieng"
	"github.com/shibukawa/hwnpu/internal/chatcmd"
)

var _ chatcmd.Host = (*Host)(nil)

// chatEngine is the engine's commands of hwnpu-chat, after the definitions of the client's
// store are read (Store.Catalog's), when st is given.
func chatEngine(st *ai.StoreSpec) (aieng.ChatCommands, error) {
	if aieng.Chat == nil {
		return nil, aieng.ErrNone
	}
	if st != nil {
		if _, err := ai.HostCatalog(*st); err != nil {
			return nil, err
		}
	}
	return aieng.Chat, nil
}

// ChatList is hwnpu-chat's list of the models (hwnpu-chat list, and the models of a kind when
// --model names none) in the engine's definitions.
func (h *Host) ChatList(ctx context.Context, st ai.StoreSpec, r chatcmd.ListRequest) (string, error) {
	e, err := chatEngine(&st)
	if err != nil {
		return "", err
	}
	var b strings.Builder
	e.List(ctx, &b, r.Kinds, r.Full, r.Backend)
	return b.String(), nil
}

// ChatPicked is why a use or a family picks the model it does here.
func (h *Host) ChatPicked(ctx context.Context, st ai.StoreSpec, name string) (string, error) {
	e, err := chatEngine(&st)
	if err != nil {
		return "", err
	}
	var b strings.Builder
	e.Picked(name, &b)
	return b.String(), nil
}

// ChatKeepMemory keeps the memory a model was run as for its next runs, and says so.
func (h *Host) ChatKeepMemory(ctx context.Context, st ai.StoreSpec, model string, gb int) (string, error) {
	e, err := chatEngine(&st)
	if err != nil {
		return "", err
	}
	var b strings.Builder
	e.KeepMemory(model, gb, &b)
	return b.String(), nil
}

// ChatDevice checks that the engine has the NPU of the backend named.
func (h *Host) ChatDevice(ctx context.Context, backend string) error {
	e, err := chatEngine(nil)
	if err != nil {
		return err
	}
	return e.Device(ctx, backend)
}

// ChatProfile is the profile of the steps the engine ran.
func (h *Host) ChatProfile(ctx context.Context) (string, error) {
	e, err := chatEngine(nil)
	if err != nil {
		return "", err
	}
	var b strings.Builder
	e.Profile(&b)
	return b.String(), nil
}
