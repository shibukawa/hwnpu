package aiengine

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"text/tabwriter"

	"github.com/shibukawa/hwnpu/internal/aieng"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
)

func init() { aieng.Chat = chatCommands{} }

// chatCommands is hwnpu-chat's aieng.ChatCommands, of internal/chat.
type chatCommands struct{}

func (chatCommands) List(ctx context.Context, w io.Writer, kinds []string, full bool, backend string) {
	if full {
		showModels(w, chat.OnlyAMD(ctx, backend))
		return
	}
	var ks []chat.Kind
	for _, k := range kinds {
		ks = append(ks, chat.Kind(k))
	}
	chat.List(w, ks...)
}

func (chatCommands) Picked(name string, log io.Writer) { chat.LookupOn(name, 0, log) }

func (chatCommands) KeepMemory(model string, gb int, log io.Writer) {
	if m, ok := chat.Lookup(model); ok {
		keepMemory(m, gb, log)
	}
}

func (chatCommands) Device(ctx context.Context, backend string) error {
	if backend == llm.BackendCoreAI {
		if _, _, err := llm.FindDevice(ctx, backend); err != nil {
			return fmt.Errorf("hwnpu-chat runs on Apple's Neural Engine through Core AI, which needs macOS 27: %w", err)
		}
	}
	return nil
}

func (chatCommands) Profile(w io.Writer) { decode.WriteProfile(w) }

// keepMemory keeps gb, the memory given, as the one to run model m as (chat.ChooseMemory), now
// that it is ready on it. Not keeping it fails no run.
func keepMemory(m chat.Model, gb int, log io.Writer) {
	if gb == 0 || chat.ChosenMemory(m) == gb {
		return
	}
	if err := chat.ChooseMemory(m, gb); err != nil {
		fmt.Fprintf(log, "%s: %d GB not kept for the next runs: %v\n", m.Name, gb, err)
		return
	}
	fmt.Fprintf(log, "%s: runs as on a machine of %d GB from now on, in every hwnpu command (%s; hwnpu memory %s --forget forgets it)\n", m.Name, gb, chat.RunFile(), m.Name)
}

// showModels writes the models of the definitions hwnpu-chat runs (chat.List) and what the
// features mean (hwnpu-chat list); amd: the export each runs from on AMD's NPU.
func showModels(w io.Writer, amd bool) {
	if amd { // what runs: the export here, or the one the first run makes
		chat.ListHave(w, "RUNS FROM", func(m chat.Model) string {
			if d := chat.FindAMDExport(m); d != "" {
				return shortPath(d)
			}
			if m.Kind == chat.KindEmbed {
				if d := chat.FindEmbedAMDExport(m); d != "" {
					return shortPath(d)
				}
				return "exported on first run"
			}
			if len(m.AMDExport) > 0 {
				return "exported on first run"
			}
			return "- (no form for AMD)"
		}, chat.KindChat, chat.KindEmbed, chat.KindDecision)
	} else {
		chat.List(w, chat.KindChat, chat.KindEmbed, chat.KindDecision)
	}
	fmt.Fprintln(w, "\nFeatures:")
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	for _, f := range chat.Features {
		fmt.Fprintf(tw, "  %s\t%s\n", f.Name, f.About)
	}
	tw.Flush()
	fmt.Fprintf(w, "\n--model (-m) takes a model's name, a use or a family (translate, plamo3translate), either with a\n"+
		"tag (plamo3translate:2b, chat:e4b, chat:16g), or a Hugging Face repository OWNER/NAME of a supported\n"+
		"architecture; serve takes several, comma separated. The default is LFM3-1.7B (search: embedding).\n"+
		"Requests may name a model the same ways. Only the model picked is downloaded.\n"+
		"On AMD's NPU a name runs its export, NAME.amd-npu2-q4 (-q6, or none for bf16) in %s or ~/models;\n"+
		"the first run takes it from the sources (hwnpu-dev sources), else downloads the checkpoint\n"+
		"and makes it with IRON (tools/amd/iron_export_lm.py, %s, INSTALL_AMD.md).\n"+
		"Models are kept in %s (HWNPU_CACHE moves it).\n", shortPath(chat.AMDRoot()), chat.IRONEnv, shortPath(cache.Root()))
}

// shortPath is path with the home directory written ~.
func shortPath(path string) string {
	home, err := os.UserHomeDir()
	if err == nil && home != "" && strings.HasPrefix(path, home+string(filepath.Separator)) {
		return "~" + path[len(home):]
	}
	return path
}
