package store

import (
	"bytes"
	"context"
	"crypto/rand"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shibukawa/tinygodriver/storage/s3"
)

// TestS3Server runs the S3 store against a real server (RustFS: docs/cache-sharing.md 7.2), when
// HWNPU_S3_TEST_ENDPOINT names it (with AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY); the bucket
// hwnpu-test is made there if missing.
func TestS3Server(t *testing.T) {
	ep := os.Getenv("HWNPU_S3_TEST_ENDPOINT")
	if ep == "" {
		t.Skip("HWNPU_S3_TEST_ENDPOINT is not set")
	}
	ctx := context.Background()
	c, err := s3.New(s3.WithEndpoint(ep), s3.WithRegion("us-east-1"))
	if err != nil {
		t.Fatal(err)
	}
	if err := c.CreateBucket(ctx, "hwnpu-test"); err != nil && !errors.Is(err, s3.ErrBucketExists) && !strings.Contains(err.Error(), "BucketAlreadyOwnedByYou") {
		t.Fatal(err)
	}
	old := partSize
	partSize = 5 << 20 // S3's least part
	t.Cleanup(func() { partSize = old })
	st, err := OpenS3(S3Config{Name: "rustfs", Bucket: "hwnpu-test", Prefix: "it", Endpoint: ep})
	if err != nil {
		t.Fatal(err)
	}
	big := make([]byte, 12<<20+123) // 3 parts
	rand.Read(big)
	src := filepath.Join(t.TempDir(), "big")
	os.WriteFile(src, big, 0o644)
	if err := st.Put(ctx, "a/big", src); err != nil {
		t.Fatal(err)
	}
	small := filepath.Join(t.TempDir(), "small")
	os.WriteFile(small, []byte("small"), 0o644)
	if err := st.Put(ctx, "a/small", small); err != nil {
		t.Fatal(err)
	}
	empty := filepath.Join(t.TempDir(), "empty")
	os.WriteFile(empty, nil, 0o644)
	if err := st.Put(ctx, "a/empty", empty); err != nil {
		t.Fatal(err)
	}
	if n, ok, err := st.Has(ctx, "a/big"); err != nil || !ok || n != int64(len(big)) {
		t.Fatalf("Has: %d %v %v", n, ok, err)
	}
	if _, ok, err := st.Has(ctx, "a/none"); err != nil || ok {
		t.Errorf("Has of none: %v %v", ok, err)
	}
	dst := filepath.Join(t.TempDir(), "big")
	os.WriteFile(dst+".part", big[:7<<20], 0o644) // cut off: goes on
	if _, err := st.Get(ctx, "a/big", dst, nil); err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(dst); !bytes.Equal(b, big) {
		t.Fatalf("got %d bytes", len(b))
	}
	if _, err := st.Get(ctx, "a/none", dst+"2", nil); !errors.Is(err, ErrNotFound) {
		t.Errorf("Get of none: %v", err)
	}
	keys, err := st.List(ctx, "a/")
	if err != nil || strings.Join(keys, " ") != "a/big a/empty a/small" {
		t.Errorf("List: %v %v", keys, err)
	}
	for _, k := range keys {
		if err := st.Delete(ctx, k); err != nil {
			t.Fatal(err)
		}
	}
	if keys, _ := st.List(ctx, "a/"); len(keys) != 0 {
		t.Errorf("left %v", keys)
	}
}
