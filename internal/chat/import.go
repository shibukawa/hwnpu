package chat

import (
	"cmp"
	"context"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	shared "github.com/shibukawa/hwnpu/internal/store"
)

// HubFile is a file of a repository's tree on Hugging Face at a commit.
type HubFile struct {
	Path string
	Size int64
	// SHA256 is the file's SHA-256 when it is in LFS (the large ones); GitSHA1 the git blob's
	// (every file's).
	SHA256, GitSHA1 string
}

// HubTree is the commit revision of repo is at on Hugging Face and the files of its tree.
func HubTree(ctx context.Context, repo, revision string) (string, map[string]HubFile, error) {
	var rev struct {
		SHA string `json:"sha"`
	}
	if err := hubJSON(ctx, hubURL+"api/models/"+repo+"/revision/"+cmp.Or(revision, "main"), &rev); err != nil {
		return "", nil, err
	}
	if !commitRE.MatchString(rev.SHA) {
		return "", nil, fmt.Errorf("chat: %s@%s: Hugging Face gave no commit", repo, revision)
	}
	files := map[string]HubFile{}
	next := hubURL + "api/models/" + repo + "/tree/" + rev.SHA + "?recursive=true&expand=false"
	for next != "" {
		var page []struct {
			Type string `json:"type"`
			Path string `json:"path"`
			Size int64  `json:"size"`
			OID  string `json:"oid"`
			LFS  *struct {
				OID string `json:"oid"`
			} `json:"lfs"`
		}
		link, err := hubJSONPage(ctx, next, &page)
		if err != nil {
			return "", nil, err
		}
		for _, f := range page {
			if f.Type != "file" {
				continue
			}
			h := HubFile{Path: f.Path, Size: f.Size, GitSHA1: f.OID}
			if f.LFS != nil {
				h.SHA256 = f.LFS.OID
			}
			files[f.Path] = h
		}
		next = link
	}
	return rev.SHA, files, nil
}

var nextLink = regexp.MustCompile(`<([^>]+)>;\s*rel="next"`)

func hubJSON(ctx context.Context, u string, v any) error {
	_, err := hubJSONPage(ctx, u, v)
	return err
}

// hubJSONPage reads a JSON answer of the hub's API and the URL of its next page ("" for none).
func hubJSONPage(ctx context.Context, u string, v any) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return "", err
	}
	if tok := hfToken(); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return "", fmt.Errorf("chat: %s: %s %s", u, resp.Status, strings.TrimSpace(string(b)))
	}
	if err := json.NewDecoder(resp.Body).Decode(v); err != nil {
		return "", fmt.Errorf("chat: %s: %w", u, err)
	}
	next := ""
	if m := nextLink.FindStringSubmatch(resp.Header.Get("Link")); m != nil {
		next = m[1]
	}
	return next, nil
}

// Import is what ImportOriginal did.
type Import struct {
	Commit                 string
	Put, Had, Skipped, Bad int   // files put, already there, not of the repository, differing
	Bytes                  int64 // of the files put
	Files                  []string
}

// ImportOriginal puts the files of a folder dir that are files of Hugging Face repository repo
// (OWNER/NAME[@REVISION]) at its commit, with the same contents, in s as the repository's
// (original/OWNER/NAME/ and its hwnpu-lock.json). name maps a file's path in dir to its path in
// the repository ("" leaves it out; nil: the same). A file s already names is not read again.
// Files that are not the repository's (programs, logs) are left out.
func ImportOriginal(ctx context.Context, s shared.Source, repo, dir string, name func(string) string, log io.Writer) (Import, error) {
	base, rev, _ := strings.Cut(repo, "@")
	commit, tree, err := HubTree(ctx, base, rev)
	if err != nil {
		return Import{}, err
	}
	res := Import{Commit: commit}
	have := shared.Lock{}
	if l, err := shared.ReadLock(ctx, s, base); err == nil {
		if l.Commit != commit {
			return res, fmt.Errorf("%s holds %s of %.12s, not %.12s (hwnpu-dev cache update moves it)", s, base, l.Commit, commit)
		}
		have = l
	}
	err = filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		rel, _ := filepath.Rel(dir, p)
		rel = filepath.ToSlash(rel)
		if name != nil {
			rel = name(rel)
		}
		h, ok := tree[rel]
		fi, ferr := d.Info()
		if rel == "" || !ok || ferr != nil || !fi.Mode().IsRegular() || fi.Size() != h.Size {
			res.Skipped++
			return nil
		}
		if f, ok := have.Files[rel]; ok && f.Size == h.Size && (h.SHA256 == "" || f.SHA256 == h.SHA256) {
			res.Had++
			return nil
		}
		sum, gitSum, err := sums(p, fi.Size())
		if err != nil {
			return err
		}
		if h.SHA256 != "" && sum != h.SHA256 || h.SHA256 == "" && gitSum != h.GitSHA1 {
			res.Bad++
			if log != nil {
				fmt.Fprintf(log, "%s: %s differs from %s's at %.12s: left out\n", p, rel, base, commit)
			}
			return nil
		}
		if log != nil {
			fmt.Fprintf(log, "%s: putting %s (%.2f GB) in %s\n", base, rel, float64(fi.Size())/1e9, s)
		}
		if err := shared.PutOriginal(ctx, s, base, commit, rel, p, sum); err != nil {
			return fmt.Errorf("%s: %s: %w", base, rel, err)
		}
		res.Put++
		res.Bytes += fi.Size()
		res.Files = append(res.Files, rel)
		return nil
	})
	return res, err
}

// sums are a file's SHA-256 and the SHA-1 of it as a git blob.
func sums(path string, size int64) (string, string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", "", err
	}
	defer f.Close()
	h, g := sha256.New(), sha1.New()
	fmt.Fprintf(g, "blob %d\x00", size)
	if _, err := io.Copy(io.MultiWriter(h, g), f); err != nil {
		return "", "", err
	}
	return hex.EncodeToString(h.Sum(nil)), hex.EncodeToString(g.Sum(nil)), nil
}
