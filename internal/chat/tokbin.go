package chat

import (
	"context"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/shibukawa/bonsaiai/internal/cache"
	"github.com/shibukawa/bonsaiai/internal/tokenizer"
)

// TokbinKind is the kind of entry of a checkpoint's tokenizer as its binary file
// (tokenizer.WriteBinary), which the sources hold as they hold a conversion: a machine that runs
// a model from its checkpoint (Apple's Neural Engine, Intel's NPU, the CPU) fetches it in place
// of the tokenizer's text and maps it, with no parsing at its first start either. An export for
// AMD's NPU holds its own (the recipe's finish).
const TokbinKind = "tokenizer"

// TokenizerOut is the folder the binary file of the tokenizer file (a path in repo:
// tokenizer.json, tokenizer/tokenizer.json, tokenizer.jsonl) is kept in:
// tokenizers/OWNER/NAME/STEM under the cache's root, STEM the file's path without its extension,
// "/" as "-".
func TokenizerOut(repo, file string) string {
	base, _, _ := strings.Cut(repo, "@")
	stem := strings.ReplaceAll(strings.TrimSuffix(file, path.Ext(file)), "/", "-")
	return cache.Sub("tokenizers", filepath.FromSlash(base), stem)
}

// PrepareTokenizer is the path to load the tokenizer file (a path in repo, a model's
// checkpoint) with: the binary file of the entry of the sources (TokbinKind), else made here
// from the file downloaded from repo (and put in the sources written back to), beside the path
// it gives (tokenizer.Load takes the binary file for it). PLaMo's tokenizer.jsonl is made with
// the tokenizer_config.json beside it. The entry is the repository's, whichever model of it
// asks (tokbinModel).
func PrepareTokenizer(ctx context.Context, repo, file string, log io.Writer) (string, error) {
	m := tokbinModel(repo)
	out := TokenizerOut(repo, file)
	want := filepath.Join(out, path.Base(file))
	bin := tokenizer.BinaryPath(want)
	if tokenizer.BinaryCurrent(bin) {
		return want, nil // made or fetched before (an entry.json only when it went in a source)
	}
	build := func() error {
		src, err := fetch(ctx, repo, file, log)
		if err != nil {
			return err
		}
		if strings.HasSuffix(file, ".jsonl") {
			cfg := path.Join(path.Dir(file), "tokenizer_config.json")
			if _, err := fetch(ctx, repo, cfg, log); err != nil && !NotFound(err) {
				return err
			}
		}
		if err := os.MkdirAll(out, 0o755); err != nil {
			return err
		}
		return tokenizer.WriteBinary(src, bin)
	}
	if err := SharedExport(ctx, TokbinKind, tokbinRecipe(ctx, m, repo, file), out, log, build); err != nil {
		return "", err
	}
	if _, err := os.Stat(bin); err != nil {
		return "", fmt.Errorf("chat: %s: the tokenizer's binary file is not there: %w", m.Name, err)
	}
	return want, nil
}

// tokbinRecipe is the key of the binary file of repo's tokenizer file: the model, the
// repository and its commit (and the SHA-256 the definitions pin of the file), the file and the
// binary file's version. nil when the commit is not known (it is made here, not shared).
func tokbinRecipe(ctx context.Context, m Model, repo, file string) map[string]string {
	commit := sourceCommit(ctx, m)
	if commit == "" {
		return nil
	}
	base, _, _ := strings.Cut(repo, "@")
	r := map[string]string{"model": m.Name, "source": base, "commit": commit, "file": file, "tokbin": fmt.Sprint(tokenizer.BinaryVersion)}
	if d, ok := m.SHA256[file]; ok {
		r["sha256:"+file] = strings.ToLower(d)
	}
	return r
}

// tokbinModel is the model the entries of repo's tokenizer are under: named by the repository
// (OWNER/NAME's NAME), so that every model of one checkpoint (LFM3-Audio-2B's chat and speech)
// and every caller finds the same; with the SHA-256 the definitions pin of its files.
func tokbinModel(repo string) Model {
	base, _, _ := strings.Cut(repo, "@")
	m := Model{Name: repoName(base), Repo: repo}
	for _, c := range Catalog {
		if r, _, _ := strings.Cut(c.Repo, "@"); c.Repo != "" && strings.EqualFold(r, base) && len(c.SHA256) > 0 {
			m.SHA256 = c.SHA256
			break
		}
	}
	return m
}
