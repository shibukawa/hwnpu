package store

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/shibukawa/tinygodriver/storage/s3"
)

// S3Config is where an S3 store is and how it is signed for (docs/cache-sharing.md 7.2).
type S3Config struct {
	Name           string
	Bucket, Prefix string
	// Endpoint is the server's URL (rustfs, MinIO: http://host:9000); "" for AWS's of Region
	// (or AWS_ENDPOINT_URL_S3, AWS_ENDPOINT_URL).
	Endpoint string
	// Region signs the requests: Region, else the profile's, AWS_REGION, AWS_DEFAULT_REGION,
	// else us-east-1 (what S3-compatible servers take).
	Region string
	// Profile is the profile of ~/.aws/credentials and ~/.aws/config whose static keys sign the
	// requests ("" for AWS_PROFILE, else the environment's keys, else default's).
	Profile string
	// PathStyle: nil picks by the endpoint (virtual hosts on AWS's, paths elsewhere).
	PathStyle *bool
	// NoSign reads a public bucket with plain GETs, unsigned.
	NoSign bool
}

// S3 is a Store of an S3 bucket: the object of key is Prefix+key.
type S3 struct {
	cfg    S3Config
	client *s3.Client // nil for NoSign
	base   *url.URL   // NoSign: the URL of the bucket's objects
}

// partSize is the size of the parts a large file goes up in, and the size above which it does.
var partSize int64 = 64 << 20

// tries is how many times a request that failed in a way that may pass (5xx, SlowDown, a cut
// connection) is made.
const tries = 3

// OpenS3 makes an S3 store. A store that signs needs keys: the profile's or the environment's.
func OpenS3(c S3Config) (*S3, error) {
	if c.Prefix != "" && !strings.HasSuffix(c.Prefix, "/") {
		c.Prefix += "/"
	}
	prof := readProfile(c.Profile)
	if c.Region == "" {
		c.Region = firstOf(prof.region, os.Getenv("AWS_REGION"), os.Getenv("AWS_DEFAULT_REGION"), "us-east-1")
	}
	if c.Endpoint == "" {
		c.Endpoint = firstOf(prof.endpoint, os.Getenv("AWS_ENDPOINT_URL_S3"), os.Getenv("AWS_ENDPOINT_URL"))
	}
	st := &S3{cfg: c}
	if c.NoSign {
		ep := c.Endpoint
		if ep == "" {
			ep = "https://s3." + c.Region + ".amazonaws.com"
		}
		if !strings.Contains(ep, "://") {
			ep = "https://" + ep
		}
		u, err := url.Parse(strings.TrimSuffix(ep, "/") + "/" + url.PathEscape(c.Bucket) + "/")
		if err != nil {
			return nil, fmt.Errorf("store: %s: %w", c.Name, err)
		}
		st.base = u
		return st, nil
	}
	creds := prof.creds
	if !creds.Valid() {
		return nil, fmt.Errorf("store: %s: no keys for S3: set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, or give the source a profile of ~/.aws/credentials with static keys (for SSO: eval \"$(aws configure export-credentials --profile NAME --format env)\")", c.Name)
	}
	opts := []s3.Option{s3.WithRegion(c.Region), s3.WithCredentials(creds), s3.WithUnsignedPayload(false)}
	if c.Endpoint != "" {
		opts = append(opts, s3.WithEndpoint(c.Endpoint))
	}
	if c.PathStyle != nil {
		opts = append(opts, s3.WithPathStyle(*c.PathStyle))
	}
	// the default timeout bounds a whole operation: too short for a large object
	opts = append(opts, s3.WithTimeout(6*time.Hour))
	client, err := s3.New(opts...)
	if err != nil {
		return nil, fmt.Errorf("store: %s: %w", c.Name, err)
	}
	st.client = client
	return st, nil
}

func (s *S3) String() string {
	d := fmt.Sprintf("%s (s3://%s/%s", s.cfg.Name, s.cfg.Bucket, s.cfg.Prefix)
	if s.cfg.Endpoint != "" {
		d += " at " + s.cfg.Endpoint
	}
	return d + ")"
}

func (s *S3) key(key string) (string, error) {
	if !SafeKey(key) {
		return "", fmt.Errorf("store: %q is not a key inside %s", key, s)
	}
	return s.cfg.Prefix + key, nil
}

// retry runs f until it succeeds, fails in a way that does not pass, or has run tries times,
// waiting longer each time.
func retry(ctx context.Context, f func() error) error {
	var err error
	for i := 0; i < tries; i++ {
		if err = f(); err == nil || !transient(err) || ctx.Err() != nil {
			return err
		}
		wait := time.Duration(1<<i)*time.Second + time.Duration(rand.Int64N(int64(500*time.Millisecond)))
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(wait):
		}
	}
	return err
}

func transient(err error) bool {
	var e *s3.Error
	if errors.As(err, &e) {
		return e.StatusCode >= 500 || e.Code == "SlowDown" || e.StatusCode == http.StatusTooManyRequests
	}
	var ne net.Error
	return errors.As(err, &ne) || errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, io.EOF)
}

func notFound(err error) bool {
	if errors.Is(err, s3.ErrNoSuchKey) {
		return true
	}
	var e *s3.Error
	return errors.As(err, &e) && e.StatusCode == http.StatusNotFound
}

// open reads the object from offset: its body, the bytes from offset on, and the whole size.
func (s *S3) open(ctx context.Context, k string, offset int64) (io.ReadCloser, int64, int64, error) {
	if s.client == nil {
		u := *s.base
		u.Path += k
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
		if err != nil {
			return nil, 0, 0, err
		}
		if offset > 0 {
			req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return nil, 0, 0, err
		}
		switch resp.StatusCode {
		case http.StatusOK:
			if offset > 0 { // the server ignored the range
				resp.Body.Close()
				return s.open(ctx, k, 0)
			}
			return resp.Body, resp.ContentLength, resp.ContentLength, nil
		case http.StatusPartialContent:
			return resp.Body, resp.ContentLength, offset + resp.ContentLength, nil
		case http.StatusNotFound, http.StatusForbidden: // a public bucket answers 403 for a key it does not have
			resp.Body.Close()
			return nil, 0, 0, ErrNotFound
		}
		resp.Body.Close()
		return nil, 0, 0, fmt.Errorf("store: GET %s: %s", u.String(), resp.Status)
	}
	var obj *s3.Object
	var err error
	if offset > 0 {
		obj, err = s.client.GetRange(ctx, s.cfg.Bucket, k, offset, 0)
	} else {
		obj, err = s.client.Get(ctx, s.cfg.Bucket, k)
	}
	if notFound(err) {
		return nil, 0, 0, ErrNotFound
	}
	if errors.Is(err, s3.ErrInvalidRange) && offset > 0 {
		return s.open(ctx, k, 0)
	}
	if err != nil {
		return nil, 0, 0, err
	}
	n := obj.Size
	return obj.Body, n, offset + n, nil
}

// Get downloads the object, going on from dst.part when a Get was cut off.
func (s *S3) Get(ctx context.Context, key, dst string, bar func(done, total int64)) (int64, error) {
	k, err := s.key(key)
	if err != nil {
		return 0, err
	}
	if err := os.MkdirAll(dirOf(dst), 0o755); err != nil {
		return 0, err
	}
	var size int64
	err = retry(ctx, func() error {
		// the objects are never changed in place (blobs by their SHA-256, the checkpoints' files
		// under their commit): a part goes on from where it stopped
		var have int64
		if fi, err := os.Stat(dst + ".part"); err == nil {
			have = fi.Size()
		}
		if have > 0 {
			n, ok, err := s.Has(ctx, key)
			switch {
			case err != nil:
				return err
			case !ok:
				return ErrNotFound
			case have == n:
				size = n
				return os.Rename(dst+".part", dst)
			case have > n:
				os.Remove(dst + ".part")
				have = 0
			}
		}
		body, _, total, err := s.open(ctx, k, have)
		if err != nil {
			return err
		}
		defer body.Close()
		size, err = copyTo(ctx, body, dst, have, total, bar)
		return err
	})
	if errors.Is(err, ErrNotFound) {
		return 0, fmt.Errorf("store: %s: %s: %w", s.cfg.Name, key, ErrNotFound)
	}
	return size, err
}

func (s *S3) Read(ctx context.Context, key string) ([]byte, error) {
	k, err := s.key(key)
	if err != nil {
		return nil, err
	}
	var b []byte
	err = retry(ctx, func() error {
		body, _, _, err := s.open(ctx, k, 0)
		if err != nil {
			return err
		}
		defer body.Close()
		b, err = io.ReadAll(io.LimitReader(body, 64<<20))
		return err
	})
	if errors.Is(err, ErrNotFound) {
		return nil, fmt.Errorf("store: %s: %s: %w", s.cfg.Name, key, ErrNotFound)
	}
	return b, err
}

func (s *S3) Has(ctx context.Context, key string) (int64, bool, error) {
	k, err := s.key(key)
	if err != nil {
		return 0, false, err
	}
	if s.client == nil {
		u := *s.base
		u.Path += k
		req, err := http.NewRequestWithContext(ctx, http.MethodHead, u.String(), nil)
		if err != nil {
			return 0, false, err
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return 0, false, err
		}
		resp.Body.Close()
		if resp.StatusCode == http.StatusOK {
			return resp.ContentLength, true, nil
		}
		if resp.StatusCode == http.StatusNotFound || resp.StatusCode == http.StatusForbidden {
			return 0, false, nil
		}
		return 0, false, fmt.Errorf("store: HEAD %s: %s", u.String(), resp.Status)
	}
	var info *s3.ObjectInfo
	err = retry(ctx, func() error {
		info, err = s.client.Head(ctx, s.cfg.Bucket, k)
		return err
	})
	if notFound(err) {
		return 0, false, nil
	}
	if err != nil {
		return 0, false, err
	}
	return info.Size, true, nil
}

func (s *S3) writable() error {
	if s.client == nil {
		return fmt.Errorf("store: %s is read without signing (no_sign): it is not written", s.cfg.Name)
	}
	return nil
}

// Put uploads the file: in one request up to partSize, above it in parts of partSize (S3's one
// request stops at 5 GiB; IRON's banks are larger). An upload that fails is aborted.
func (s *S3) Put(ctx context.Context, key, src string) error {
	if err := s.writable(); err != nil {
		return err
	}
	k, err := s.key(key)
	if err != nil {
		return err
	}
	f, err := os.Open(src)
	if err != nil {
		return err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return err
	}
	size := fi.Size()
	if size <= partSize {
		return retry(ctx, func() error {
			_, err := s.client.Put(ctx, s.cfg.Bucket, k, io.NewSectionReader(f, 0, size), s3.WithContentLength(size))
			return err
		})
	}
	var up *s3.MultipartUpload
	if err := retry(ctx, func() error {
		up, err = s.client.CreateMultipartUpload(ctx, s.cfg.Bucket, k)
		return err
	}); err != nil {
		return err
	}
	var parts []s3.CompletedPart
	for n, off := 1, int64(0); off < size; n, off = n+1, off+partSize {
		l := min(partSize, size-off)
		var p *s3.CompletedPart
		err := retry(ctx, func() error {
			p, err = s.client.UploadPart(ctx, *up, n, io.NewSectionReader(f, off, l), s3.WithContentLength(l))
			return err
		})
		if err != nil {
			s.client.AbortMultipartUpload(context.WithoutCancel(ctx), *up)
			return err
		}
		parts = append(parts, *p)
	}
	if err := retry(ctx, func() error {
		_, err := s.client.CompleteMultipartUpload(ctx, *up, parts)
		return err
	}); err != nil {
		s.client.AbortMultipartUpload(context.WithoutCancel(ctx), *up)
		return err
	}
	return nil
}

func (s *S3) Write(ctx context.Context, key string, b []byte) error {
	if err := s.writable(); err != nil {
		return err
	}
	k, err := s.key(key)
	if err != nil {
		return err
	}
	return retry(ctx, func() error {
		_, err := s.client.Put(ctx, s.cfg.Bucket, k, bytes.NewReader(b), s3.WithContentLength(int64(len(b))))
		return err
	})
}

func (s *S3) List(ctx context.Context, prefix string) ([]string, error) {
	if err := s.writable(); err != nil {
		return nil, err // a public bucket's listing is not read
	}
	var keys []string
	token := ""
	for {
		var page *s3.ListResult
		err := retry(ctx, func() error {
			var err error
			page, err = s.client.List(ctx, s.cfg.Bucket, s3.WithPrefix(s.cfg.Prefix+prefix), s3.WithContinuationToken(token))
			return err
		})
		if err != nil {
			return nil, err
		}
		for _, o := range page.Objects {
			keys = append(keys, strings.TrimPrefix(o.Key, s.cfg.Prefix))
		}
		if !page.IsTruncated {
			return keys, nil
		}
		token = page.NextToken
	}
}

func (s *S3) Delete(ctx context.Context, key string) error {
	if err := s.writable(); err != nil {
		return err
	}
	k, err := s.key(key)
	if err != nil {
		return err
	}
	err = retry(ctx, func() error { return s.client.Delete(ctx, s.cfg.Bucket, k) })
	if notFound(err) {
		return nil
	}
	return err
}

func dirOf(p string) string {
	i := strings.LastIndexAny(p, `/\`)
	if i < 0 {
		return "."
	}
	return p[:i]
}

func firstOf(v ...string) string {
	for _, s := range v {
		if s != "" {
			return s
		}
	}
	return ""
}
