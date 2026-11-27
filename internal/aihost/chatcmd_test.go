package aihost

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/aieng"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/chatcmd"
	"github.com/shibukawa/hwnpu/internal/daemon"
)

// What hwnpu-chat asks of hwnpud besides its models comes from hwnpud's engine (chatcmd.Host):
// the list, why a use picks a model, the memory kept, the check of the NPU and the profile.
func TestChatThroughHwnpud(t *testing.T) {
	serve(t)
	ctx := context.Background()
	st := testStore(t)
	var list bytes.Buffer
	if err := chatcmd.List(ctx, st, &list, chatcmd.ListRequest{Full: true, Backend: "intel"}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(list.String(), "LFM3-1.7B") || !strings.Contains(list.String(), "\nFeatures:") {
		t.Errorf("hwnpu-chat list:\n%s", list.String())
	}
	var picked bytes.Buffer
	if err := chatcmd.Picked(ctx, st, "translate", &picked); err != nil || !strings.HasPrefix(picked.String(), "translate: ") {
		t.Errorf("why translate picks its model: %q %v", picked.String(), err)
	}
	m, _ := chat.Lookup("LFM3-1.7B")
	var kept bytes.Buffer
	if err := chatcmd.KeepMemory(ctx, st, m.Name, 8, &kept); err != nil || chat.ChosenMemory(m) != 8 ||
		!strings.Contains(kept.String(), "runs as on a machine of 8 GB from now on") {
		t.Errorf("kept: %d %q %v", chat.ChosenMemory(m), kept.String(), err)
	}
	if err := chatcmd.Device(ctx, "intel"); err != nil {
		t.Errorf("no check of Intel's NPU: %v", err)
	}
	if err := chatcmd.Profile(ctx, &bytes.Buffer{}); err != nil {
		t.Errorf("profile: %v", err)
	}
	// with HWNPU_DAEMON=off they are this process's engine's, and without one there is none
	t.Setenv(daemon.ModeEnv, "off")
	picked.Reset()
	if err := chatcmd.Picked(ctx, st, "translate", &picked); err != nil || !strings.HasPrefix(picked.String(), "translate: ") {
		t.Errorf("here: %q %v", picked.String(), err)
	}
	saved := aieng.Chat
	aieng.Chat = nil
	defer func() { aieng.Chat = saved }()
	if err := chatcmd.Profile(ctx, &bytes.Buffer{}); !errors.Is(err, aieng.ErrNone) {
		t.Errorf("no engine: %v", err)
	}
}
