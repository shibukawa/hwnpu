package chat

import (
	"cmp"
	"context"
	"fmt"
	"os"
	"regexp"
	"slices"
	"strings"

	shared "github.com/shibukawa/hwnpu/internal/store"
)

// The converted models on Hugging Face (docs/prebuilt.md): models.yaml's hub names, for a
// model, the repositories its conversions are in and the commit each target reads there, and at
// its top the namespace hwnpu-dev hf push puts them in (hub: {push: hwnpu}).

// HubRepo is a repository of a model's conversions on Hugging Face (models.yaml's hub).
type HubRepo struct {
	Repo string // OWNER/NAME
	// Pins are the commit each target reads (amd-xdna2, apple, coreai-ARCH): 40 hex digits, or
	// main (to try what is there now).
	Pins map[string]string
	// File is the definition file that wrote the repository last (Builtin for the built-in one).
	File string
}

// HubPush is the namespace on Hugging Face that hwnpu-dev hf push puts the conversions in
// (models.yaml's hub.push, the last file's that gives it), and HubPushFile that file.
var HubPush, HubPushFile string

var (
	hubNamespace = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*$`)
	// HubTarget matches the targets a repository of hub pins a commit for: amd-xdna2 (AMD's
	// NPU), apple (every Mac), coreai-ARCH (a Mac of that Neural Engine, before apple), and the
	// generations to come (amd-xdna1, intel-npu4).
	HubTarget = regexp.MustCompile(`^(amd-xdna[0-9]+|apple|coreai-h[0-9]+[a-z]+|intel-npu[0-9]+)$`)
)

// toHub checks a model's hub and orders its repositories: those of the files read last first,
// then by name.
func toHub(h map[string]map[string]string, files map[string]hubOrigin) ([]HubRepo, error) {
	var out []HubRepo
	for repo, pins := range h {
		owner, name, ok := strings.Cut(repo, "/")
		if !ok || !hubNamespace.MatchString(owner) || !hubNamespace.MatchString(name) {
			return nil, fmt.Errorf("%q is not a repository of Hugging Face (OWNER/NAME)", repo)
		}
		for t, c := range pins {
			if !HubTarget.MatchString(t) {
				return nil, fmt.Errorf("%s: %q is not a target (amd-xdna2, apple, coreai-ARCH)", repo, t)
			}
			if c != "main" && !commitRE.MatchString(c) {
				return nil, fmt.Errorf("%s: %s: %q is not a commit (40 hex digits) or main", repo, t, c)
			}
		}
		out = append(out, HubRepo{Repo: repo, Pins: pins, File: files[repo].file})
	}
	slices.SortFunc(out, func(a, b HubRepo) int {
		return cmp.Or(cmp.Compare(files[b.Repo].at, files[a.Repo].at), strings.Compare(a.Repo, b.Repo))
	})
	return out, nil
}

// setHub gives the store the repositories of the catalog's conversions on Hugging Face.
func setHub(catalog []Model) {
	models := map[string][]shared.HubRepo{}
	for _, m := range catalog {
		for _, r := range m.Hub {
			models[m.Name] = append(models[m.Name], shared.HubRepo{Repo: r.Repo, Pins: r.Pins})
		}
	}
	shared.SetHub(models, hubArch, hubAuth)
}

// hubAuth is what a repository of conversions that refuses a request does: a gated one signs in
// or sends the person to agree to its terms as the models' own repositories do (gate); a
// private one takes the token of HF_TOKEN or hf auth login.
func hubAuth(ctx context.Context, repo string, status int, sent string) (string, error) {
	if isGated(ctx, repo) {
		if err := gate(ctx, repo, status, sent, os.Stderr); err != nil {
			return "", err
		}
		return hfToken(), nil
	}
	if t := hfToken(); t != "" && t != sent {
		return t, nil
	}
	return "", fmt.Errorf("chat: %s on Hugging Face refuses to be read (%d): a private repository needs a token that reads it (HF_TOKEN, or hf auth login); %s=false converts here instead",
		repo, status, shared.HubEnv)
}
