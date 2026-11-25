package sourcescmd

import (
	"context"
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"

	"github.com/shibukawa/bonsaiai/cmd/bonsaiai-dev/devopt"
	"github.com/shibukawa/bonsaiai/internal/chat"
	"github.com/shibukawa/bonsaiai/internal/console"
	"github.com/shibukawa/bonsaiai/internal/llm"
	shared "github.com/shibukawa/bonsaiai/internal/store"
)

// TokenizersCmd is bonsaiai-dev sources tokenizers.
type TokenizersCmd struct {
	Models []string `arg:"" optional:"" help:"Models of the definitions (all that run from their checkpoint's tokenizer when none)."`
	DryRun bool     `name:"dry-run" help:"Only show each model's tokenizer file."`
}

// tokenizerFile is the tokenizer file (a path in its repository) of a model that runs from its
// checkpoint's tokenizer (chat.PrepareTokenizer), "" for one that does not.
func tokenizerFile(ctx context.Context, m chat.Model) (string, error) {
	switch m.Kind {
	case chat.KindTranscribe, chat.KindSpeech:
		return "tokenizer.json", nil
	case chat.KindSFX:
		return "tokenizer/tokenizer.json", nil
	case chat.KindChat, chat.KindDecision:
		cfg, err := chat.Fetch(ctx, m.Source(), "config.json", console.Log(os.Stderr))
		if err != nil {
			return "", err
		}
		a, _, err := llm.LoadConfig(cfg, "")
		if err != nil {
			return "", err
		}
		return chat.TokenizerFiles(a.Name)[0], nil
	}
	return "", nil
}

func (tc TokenizersCmd) Run(d *devopt.Defs) error {
	if err := load(d); err != nil {
		return err
	}
	ctx, stop := signalContext()
	defer stop()
	if len(shared.Writebacks()) == 0 && !tc.DryRun {
		return errors.New("no source is written back to: give one ?writeback=true (" + shared.SourceEnv + ")")
	}
	seen := map[string]bool{}
	failed := 0
	for _, m := range chat.Catalog {
		if m.Repo == "" || len(tc.Models) > 0 && !slices.Contains(tc.Models, m.Name) {
			continue
		}
		f, err := tokenizerFile(ctx, m)
		if err != nil {
			fmt.Printf("%s: %v\n", m.Name, err)
			failed++
			continue
		}
		repo, _, _ := strings.Cut(m.Source(), "@")
		if f == "" || seen[repo+"|"+f] {
			continue
		}
		seen[repo+"|"+f] = true
		if tc.DryRun {
			fmt.Printf("%s: %s %s\n", m.Name, repo, f)
			continue
		}
		p, err := chat.PrepareTokenizer(ctx, m.Source(), f, console.Log(os.Stderr))
		if err != nil {
			if chat.NotFound(err) {
				fmt.Printf("%s: %s has no %s\n", m.Name, repo, f)
				continue
			}
			fmt.Printf("%s: %v\n", m.Name, err)
			failed++
			continue
		}
		fmt.Printf("%s: %s %s: %s\n", m.Name, repo, f, p)
	}
	shared.Flush(ctx)
	if failed > 0 {
		return fmt.Errorf("%d models' tokenizers were not put", failed)
	}
	return nil
}
