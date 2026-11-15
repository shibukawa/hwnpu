package chat

import (
	"cmp"
	"context"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"

	shared "github.com/shibukawa/hwnpu/internal/store"
)

// SystemModelsFile is the definition file of the machine, which its administrator puts there
// (the organization's models): read after the built-in one and before the user's
// (docs/cache-sharing.md 7.4).
var SystemModelsFile = func() string {
	switch runtime.GOOS {
	case "windows":
		return filepath.Join(cmp.Or(os.Getenv("ProgramData"), `C:\ProgramData`), "hwnpu", "models.yaml")
	case "darwin":
		return "/Library/Application Support/hwnpu/models.yaml"
	}
	return "/etc/hwnpu/models.yaml"
}

var sha256RE = regexp.MustCompile(`^[0-9a-f]{64}$`)

// hubFile asks the hub (a HEAD, no download) for a file's SHA-256 (X-Linked-Etag, a file in
// LFS: the large ones) and the commit it is of. Either is "" when it does not say, and both when
// hwnpu keeps off Hugging Face.
func hubFile(ctx context.Context, repo, name string) (sum, commit string) {
	if !shared.HuggingFace() {
		return "", ""
	}
	base, rev, _ := strings.Cut(repo, "@")
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, hubURL+base+"/resolve/"+cmp.Or(rev, "main")+"/"+name, nil)
	if err != nil {
		return "", ""
	}
	if tok := hfToken(); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	resp, err := headClient.Do(req)
	if err != nil {
		return "", ""
	}
	resp.Body.Close()
	if resp.StatusCode >= 400 {
		return "", ""
	}
	if e := strings.Trim(resp.Header.Get("X-Linked-Etag"), `"`); sha256RE.MatchString(e) {
		sum = e
	}
	if c := resp.Header.Get("X-Repo-Commit"); commitRE.MatchString(c) {
		commit = c
	}
	return sum, commit
}
