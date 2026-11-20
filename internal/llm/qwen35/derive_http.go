package qwen35

import (
	"cmp"
	"context"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"strings"
	"time"
)

// HTTPSource is a checkpoint of a Hugging Face repository read over HTTP, ranges of its shards
// with Range requests (DeriveMXFP4 reads Qwen3.6-35B-A3B's 72 GB without a copy on the disk).
type HTTPSource struct {
	Ctx      context.Context
	Endpoint string // HF_ENDPOINT, else https://huggingface.co
	Repo     string // OWNER/NAME
	Revision string // main when empty
	Token    string // a token for gated repositories (HF_TOKEN)
	Client   *http.Client
}

// NewHTTPSource reads repo from Hugging Face (HF_ENDPOINT, HF_TOKEN).
func NewHTTPSource(ctx context.Context, repo string) *HTTPSource {
	return &HTTPSource{Ctx: ctx, Endpoint: os.Getenv("HF_ENDPOINT"), Repo: repo, Token: strings.TrimSpace(os.Getenv("HF_TOKEN"))}
}

func (s *HTTPSource) url(name string) string {
	ep := strings.TrimSuffix(cmp.Or(s.Endpoint, "https://huggingface.co"), "/")
	return fmt.Sprintf("%s/%s/resolve/%s/%s", ep, s.Repo, cmp.Or(s.Revision, "main"), name)
}

func (s *HTTPSource) ReadFile(name string) ([]byte, error) { return s.get(name, -1, 0) }

func (s *HTTPSource) ReadRange(name string, off, n int64) ([]byte, error) { return s.get(name, off, n) }

// get reads name whole (off < 0) or n bytes from off, trying again a few times on errors of the
// network or the server's.
func (s *HTTPSource) get(name string, off, n int64) ([]byte, error) {
	ctx := cmp.Or[context.Context](s.Ctx, context.Background())
	client := cmp.Or(s.Client, http.DefaultClient)
	var last error
	for try := range 6 {
		if try > 0 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(time.Duration(1<<try) * time.Second):
			}
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, s.url(name), nil)
		if err != nil {
			return nil, err
		}
		if s.Token != "" {
			req.Header.Set("Authorization", "Bearer "+s.Token)
		}
		if off >= 0 {
			req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", off, off+n-1))
		}
		resp, err := client.Do(req)
		if err != nil {
			last = err
			continue
		}
		b, err := func() ([]byte, error) {
			defer resp.Body.Close()
			switch {
			case resp.StatusCode == http.StatusNotFound:
				return nil, fmt.Errorf("%s: %w", s.url(name), fs.ErrNotExist)
			case off >= 0 && resp.StatusCode != http.StatusPartialContent:
				return nil, fmt.Errorf("%s: %s for a range", s.url(name), resp.Status)
			case off < 0 && resp.StatusCode != http.StatusOK:
				return nil, fmt.Errorf("%s: %s", s.url(name), resp.Status)
			}
			b, err := io.ReadAll(resp.Body)
			if err == nil && off >= 0 && int64(len(b)) != n {
				err = fmt.Errorf("%s: %d bytes of the %d asked", s.url(name), len(b), n)
			}
			return b, err
		}()
		if err == nil {
			return b, nil
		}
		if strings.Contains(err.Error(), fs.ErrNotExist.Error()) || resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
			return nil, err
		}
		last = err
	}
	return nil, last
}
