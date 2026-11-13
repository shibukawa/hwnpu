// Package s3fake is an S3 server in memory for tests: the path-style objects, ranges, HEAD,
// DELETE, ListObjectsV2 and multipart uploads, without checking signatures.
package s3fake

import (
	"encoding/xml"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
)

// Server is the fake.
type Server struct {
	*httptest.Server
	mu      sync.Mutex
	Objects map[string][]byte // by "BUCKET/KEY"
	uploads map[string]map[int][]byte
	// Requests counts the requests by method.
	Requests map[string]int
	// Signed counts the requests that carried a signature.
	Signed int
}

// New starts a fake that ends with the test.
func New(t *testing.T) *Server {
	s := &Server{Objects: map[string][]byte{}, uploads: map[string]map[int][]byte{}, Requests: map[string]int{}}
	s.Server = httptest.NewServer(http.HandlerFunc(s.serve))
	t.Cleanup(s.Close)
	return s
}

// Put puts an object.
func (s *Server) Put(bucket, key string, b []byte) {
	s.mu.Lock()
	s.Objects[bucket+"/"+key] = b
	s.mu.Unlock()
}

// Get returns an object.
func (s *Server) Get(bucket, key string) ([]byte, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	b, ok := s.Objects[bucket+"/"+key]
	return b, ok
}

func (s *Server) serve(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Requests[r.Method]++
	if r.Header.Get("Authorization") != "" {
		s.Signed++
	}
	path := strings.TrimPrefix(r.URL.Path, "/")
	bucket, key, _ := strings.Cut(path, "/")
	q := r.URL.Query()
	id := bucket + "/" + key
	switch {
	case r.Method == http.MethodGet && key == "" && q.Get("list-type") == "2":
		type content struct {
			Key  string
			Size int64
		}
		var res struct {
			XMLName     xml.Name `xml:"ListBucketResult"`
			Contents    []content
			IsTruncated bool
		}
		var keys []string
		for k := range s.Objects {
			if b, rest, _ := strings.Cut(k, "/"); b == bucket && strings.HasPrefix(rest, q.Get("prefix")) {
				keys = append(keys, rest)
			}
		}
		sort.Strings(keys)
		for _, k := range keys {
			res.Contents = append(res.Contents, content{Key: k, Size: int64(len(s.Objects[bucket+"/"+k]))})
		}
		w.Header().Set("Content-Type", "application/xml")
		xml.NewEncoder(w).Encode(res)
	case r.Method == http.MethodPost && q.Has("uploads"):
		uid := fmt.Sprintf("u%d", len(s.uploads)+1)
		s.uploads[uid] = map[int][]byte{}
		fmt.Fprintf(w, `<InitiateMultipartUploadResult><Bucket>%s</Bucket><Key>%s</Key><UploadId>%s</UploadId></InitiateMultipartUploadResult>`, bucket, key, uid)
	case r.Method == http.MethodPut && q.Get("uploadId") != "":
		n, _ := strconv.Atoi(q.Get("partNumber"))
		b, _ := io.ReadAll(r.Body)
		s.uploads[q.Get("uploadId")][n] = b
		w.Header().Set("ETag", fmt.Sprintf(`"p%d"`, n))
	case r.Method == http.MethodPost && q.Get("uploadId") != "":
		parts := s.uploads[q.Get("uploadId")]
		var all []byte
		for i := 1; i <= len(parts); i++ {
			all = append(all, parts[i]...)
		}
		s.Objects[id] = all
		delete(s.uploads, q.Get("uploadId"))
		fmt.Fprintf(w, `<CompleteMultipartUploadResult><ETag>"done"</ETag></CompleteMultipartUploadResult>`)
	case r.Method == http.MethodDelete && q.Get("uploadId") != "":
		delete(s.uploads, q.Get("uploadId"))
		w.WriteHeader(http.StatusNoContent)
	case r.Method == http.MethodPut:
		b, _ := io.ReadAll(r.Body)
		s.Objects[id] = b
		w.Header().Set("ETag", `"x"`)
	case r.Method == http.MethodDelete:
		delete(s.Objects, id)
		w.WriteHeader(http.StatusNoContent)
	case r.Method == http.MethodGet || r.Method == http.MethodHead:
		b, ok := s.Objects[id]
		if !ok {
			w.Header().Set("Content-Type", "application/xml")
			w.WriteHeader(http.StatusNotFound)
			if r.Method == http.MethodGet {
				fmt.Fprint(w, `<Error><Code>NoSuchKey</Code><Message>no</Message></Error>`)
			}
			return
		}
		w.Header().Set("ETag", `"x"`)
		if rg := r.Header.Get("Range"); rg != "" && r.Method == http.MethodGet {
			from, _ := strconv.Atoi(strings.TrimSuffix(strings.TrimPrefix(rg, "bytes="), "-"))
			if from >= len(b) {
				w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
				return
			}
			w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", from, len(b)-1, len(b)))
			w.Header().Set("Content-Length", strconv.Itoa(len(b)-from))
			w.WriteHeader(http.StatusPartialContent)
			w.Write(b[from:])
			return
		}
		w.Header().Set("Content-Length", strconv.Itoa(len(b)))
		if r.Method == http.MethodGet {
			w.Write(b)
		}
	default:
		http.Error(w, "not faked", http.StatusNotImplemented)
	}
}
