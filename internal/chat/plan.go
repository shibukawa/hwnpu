package chat

import (
	"cmp"
	"context"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu/internal/console"
)

// File is a file of a model's repository and where fetch puts it.
type File struct {
	Repo string // as fetch takes it: OWNER/NAME, OWNER/NAME@REVISION or SOURCE:NAME
	Name string // in the repository
	Path string
}

// plan is the files Expect looked up that are not fetched yet (their sizes and the bytes
// written so far), and the bytes of those fetched since, for a download's bar to show the time
// left of all of them.
var plan struct {
	mu       sync.Mutex
	files    map[string]*plannedFile // by the cleaned path
	finished int64
}

type plannedFile struct{ size, written int64 }

// Expect looks up the sizes of the files that are not here yet (all at once), so that the bars
// of their downloads, one after another, show the bytes and the time left of all of them. A file
// of no size known (an s3 source, a repository that asks to sign in) is left out of the sum.
func Expect(ctx context.Context, files []File) {
	t0 := time.Now()
	var mu sync.Mutex
	sizes := map[string]int64{}
	var wg sync.WaitGroup
	sem := make(chan struct{}, 8)
	for _, f := range files {
		if _, err := os.Stat(f.Path); err == nil {
			continue
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			if n, ok := fileSize(ctx, f.Repo, f.Name); ok {
				mu.Lock()
				sizes[filepath.Clean(f.Path)] = n
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	if len(sizes) > 0 {
		console.Looked(len(sizes), time.Since(t0))
	}
	plan.mu.Lock()
	defer plan.mu.Unlock()
	if plan.files == nil {
		plan.files = map[string]*plannedFile{}
	}
	for p, n := range sizes {
		f := &plannedFile{size: n}
		if fi, err := os.Stat(p + ".part"); err == nil && fi.Size() <= n {
			f.written = fi.Size() // a download cut off goes on from there
		}
		plan.files[p] = f
	}
}

// planned is the planned file at path, if Expect looked it up and it is not fetched yet.
func planned(path string) *plannedFile {
	plan.mu.Lock()
	defer plan.mu.Unlock()
	return plan.files[filepath.Clean(path)]
}

// planWrote sets the bytes of the planned file f written so far.
func planWrote(f *plannedFile, n int64) {
	plan.mu.Lock()
	f.written = n
	plan.mu.Unlock()
}

// planNow is the bytes of the planned files done and in all.
func planNow() (done, total int64) {
	plan.mu.Lock()
	defer plan.mu.Unlock()
	done, total = plan.finished, plan.finished
	for _, f := range plan.files {
		done += min(f.written, f.size)
		total += f.size
	}
	return done, total
}

// planFinish counts the file at path, now here, as done; when no planned file is left, the next
// Expect starts a new sum.
func planFinish(path string) {
	plan.mu.Lock()
	defer plan.mu.Unlock()
	p := filepath.Clean(path)
	f, ok := plan.files[p]
	if !ok {
		return
	}
	delete(plan.files, p)
	plan.finished += f.size
	if len(plan.files) == 0 {
		plan.finished = 0
	}
}

// fileSize is the size of a file of repo in the first source that has it: a path source's file,
// or a Hugging Face repository's from a HEAD request (X-Linked-Size, which a redirect to the CDN
// carries).
func fileSize(ctx context.Context, repo, name string) (int64, bool) {
	order, key := sourceOrder(repo)
	for _, sn := range order {
		s := Sources[sn]
		switch s.Type {
		case "path":
			if n, ok := pathSize(s, key, name); ok {
				return n, true
			}
		case "hf":
			if n, ok := hfSize(ctx, key, name); ok {
				return n, true
			}
		}
	}
	return 0, false
}

func pathSize(s Source, key, name string) (int64, bool) {
	base, rev, _ := strings.Cut(key, "@")
	if !safeName(base) || !safeName(name) {
		return 0, false
	}
	dirs := []string{base}
	if rev != "" {
		dirs = []string{base + "@" + rev, base}
	}
	for _, d := range dirs {
		if fi, err := os.Stat(filepath.Join(s.Path, filepath.FromSlash(d), filepath.FromSlash(name))); err == nil && fi.Mode().IsRegular() {
			return fi.Size(), true
		}
	}
	return 0, false
}

// headClient asks without following redirects: the size is on the Hub's answer.
var headClient = &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}

func hfSize(ctx context.Context, repo, name string) (int64, bool) {
	repo, rev, _ := strings.Cut(repo, "@")
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, hubURL+repo+"/resolve/"+cmp.Or(rev, "main")+"/"+name, nil)
	if err != nil {
		return 0, false
	}
	if tok := hfToken(); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	resp, err := headClient.Do(req)
	if err != nil {
		return 0, false
	}
	resp.Body.Close()
	if resp.StatusCode >= 400 {
		return 0, false
	}
	if n, err := strconv.ParseInt(resp.Header.Get("X-Linked-Size"), 10, 64); err == nil {
		return n, true
	}
	if resp.StatusCode == http.StatusOK && resp.ContentLength >= 0 {
		return resp.ContentLength, true
	}
	return 0, false
}
