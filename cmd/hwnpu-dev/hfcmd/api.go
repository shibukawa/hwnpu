package hfcmd

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	shared "github.com/shibukawa/hwnpu/internal/store"
)

// api calls Hugging Face's API (shared.HubEndpoint) with tok ("" for none): method on path
// (/api/...), body as JSON when not nil, the answer's JSON into out when not nil. An answer not
// 2xx is an *apiError.
func api(ctx context.Context, tok, method, p string, body, out any) error {
	var r io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return err
		}
		r = bytes.NewReader(b)
	}
	ctx, cancel := context.WithTimeout(ctx, time.Minute)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, method, strings.TrimSuffix(shared.HubEndpoint, "/")+p, r)
	if err != nil {
		return err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if resp.StatusCode/100 != 2 {
		return &apiError{status: resp.StatusCode, msg: strings.TrimSpace(string(b)), what: method + " " + p}
	}
	if out != nil {
		return json.Unmarshal(b, out)
	}
	return nil
}

type apiError struct {
	status    int
	msg, what string
}

func (e *apiError) Error() string {
	return fmt.Sprintf("Hugging Face: %s: %d %s", e.what, e.status, e.msg)
}

func apiStatus(err error) int {
	var ae *apiError
	if errors.As(err, &ae) {
		return ae.status
	}
	return 0
}

// whoami is the account of a token and what it may write to (/api/whoami-v2).
type whoami struct {
	Name string `json:"name"`
	Orgs []struct {
		Name string `json:"name"`
		Role string `json:"roleInOrg"`
	} `json:"orgs"`
	Auth struct {
		AccessToken struct {
			Role        string `json:"role"` // read, write, fineGrained
			FineGrained struct {
				Scoped []struct {
					Entity struct {
						Type string `json:"type"`
						Name string `json:"name"`
					} `json:"entity"`
					Permissions []string `json:"permissions"`
				} `json:"scoped"`
			} `json:"fineGrained"`
		} `json:"accessToken"`
	} `json:"auth"`
}

// canWrite reports whether the token may make and write repositories in namespace ns, and why
// not.
func (w whoami) canWrite(ns string) (bool, string) {
	member := ns == w.Name
	for _, o := range w.Orgs {
		if o.Name == ns {
			member = true
			if o.Role == "read" {
				return false, fmt.Sprintf("%s is in %s with the role read", w.Name, ns)
			}
		}
	}
	if !member {
		return false, fmt.Sprintf("%s is not %s nor in the organization %s", w.Name, ns, ns)
	}
	tok := w.Auth.AccessToken
	switch tok.Role {
	case "write", "admin":
		return true, ""
	case "fineGrained":
		for _, s := range tok.FineGrained.Scoped {
			if s.Entity.Name == ns && slices.ContainsFunc(s.Permissions, func(p string) bool {
				return p == "repo.write" || p == "repo.content.write"
			}) {
				return true, ""
			}
		}
		return false, fmt.Sprintf("the fine-grained token of %s may not write to the repositories of %s", w.Name, ns)
	}
	return false, fmt.Sprintf("the token of %s reads only", w.Name)
}

// hfTokenFile is where huggingface_hub keeps its token (hf auth login).
func hfTokenFile() string {
	if p := os.Getenv("HF_TOKEN_PATH"); p != "" {
		return p
	}
	home := os.Getenv("HF_HOME")
	if home == "" {
		if xdg := os.Getenv("XDG_CACHE_HOME"); xdg != "" {
			home = filepath.Join(xdg, "huggingface")
		} else if h, err := os.UserHomeDir(); err == nil {
			home = filepath.Join(h, ".cache", "huggingface")
		}
	}
	return filepath.Join(home, "token")
}

// knownToken is HF_TOKEN, else hf auth login's, and where it came from; "" for none.
func knownToken() (string, string) {
	if t := strings.TrimSpace(os.Getenv("HF_TOKEN")); t != "" {
		return t, "HF_TOKEN"
	}
	if b, err := os.ReadFile(hfTokenFile()); err == nil {
		if t := strings.TrimSpace(string(b)); t != "" {
			return t, hfTokenFile()
		}
	}
	return "", ""
}

// repoInfo is what /api/models/REPO says of a repository.
type repoInfo struct {
	SHA   string `json:"sha"`
	Gated any    `json:"gated"` // false, or "auto" / "manual"
}

func (r repoInfo) gated() string { s, _ := r.Gated.(string); return s }

// modelInfo reads /api/models/REPO (at rev, when not ""); ok false when there is no such
// repository.
func modelInfo(ctx context.Context, tok, repo, rev string) (repoInfo, bool, error) {
	p := "/api/models/" + repo
	if rev != "" {
		p += "/revision/" + rev
	}
	var info repoInfo
	err := api(ctx, tok, http.MethodGet, p, nil, &info)
	if s := apiStatus(err); s == http.StatusNotFound || s == http.StatusUnauthorized && tok == "" {
		return info, false, nil
	}
	return info, err == nil, err
}

// createRepo makes the public model repository ns/name.
func createRepo(ctx context.Context, tok string, who whoami, ns, name string) error {
	body := map[string]any{"type": "model", "name": name, "private": false}
	if ns != who.Name {
		body["organization"] = ns
	}
	return api(ctx, tok, http.MethodPost, "/api/repos/create", body, nil)
}

// setGated makes the repository gated as the model's own is (auto, manual).
func setGated(ctx context.Context, tok, repo, gated string) error {
	return api(ctx, tok, http.MethodPut, "/api/models/"+repo+"/settings", map[string]any{"gated": gated}, nil)
}
