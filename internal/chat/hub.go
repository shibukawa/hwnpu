package chat

import (
	"cmp"
	"context"
	"github.com/shibukawa/configdir"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"

	shared "github.com/shibukawa/bonsaiai/internal/store"
)

// SystemModelsFile is the definition file of the machine, which its administrator puts there
// (the organization's models): read after the built-in one and before the user's
// (docs/cache-sharing-design.md 7.4).
var SystemModelsFile = func() string {
	switch runtime.GOOS {
	case "windows":
		return filepath.Join(cmp.Or(os.Getenv("ProgramData"), `C:\ProgramData`), "bonsaiai", "models.yaml")
	case "darwin":
		return "/Library/Application Support/bonsaiai/models.yaml"
	}
	return "/etc/bonsaiai/models.yaml"
}

var sha256RE = regexp.MustCompile(`^[0-9a-f]{64}$`)

// hubFile asks the hub (a HEAD, no download) for a file's SHA-256 (X-Linked-Etag, a file in
// LFS: the large ones) and the commit it is of. Either is "" when it does not say, and both when
// bonsaiai keeps off Hugging Face.
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

// BundleFolder is the folder of an application's resource folders that holds the models it
// installs with it, in the layout of a source (original/, converted/).
const BundleFolder = "models"

// BundleDirs are the folders of models app installs with it: folder (BundleFolder when "") of
// each of its resource folders that has it (configdir's QueryResourceFolders: next to the
// executable, X.app/Contents/Resources, PREFIX/share/APP, $XDG_DATA_DIRS/APP). bonsaiai's commands
// read bonsaiai's (LoadModels); an application that embeds bonsaiai says its own (modelstore).
func BundleDirs(app configdir.ConfigDir, folder string) []string {
	var dirs []string
	for _, c := range app.QueryResourceFolders() {
		d := filepath.Join(c.Path, cmp.Or(folder, BundleFolder))
		if fi, err := os.Stat(d); err == nil && fi.IsDir() {
			dirs = append(dirs, d)
		}
	}
	return dirs
}
