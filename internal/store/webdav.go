package store

import (
	"bufio"
	"bytes"
	"cmp"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"
)

// WebDAVConfig is where a WebDAV folder is and who signs in to it (docs/cache-sharing.md 7).
type WebDAVConfig struct {
	Name string
	// URL is the folder: https://nas:5006/hwnpu (Synology), https://cloud/remote.php/dav/files/USER/hwnpu
	// (Nextcloud). Its user name and password, if any, sign in.
	URL string
	// User and Password sign in with Basic authentication (over https). With no user: the
	// URL's, else the netrc's login for the host, else HWNPU_WEBDAV_USER; with no password: the
	// URL's, else the netrc's, else HWNPU_WEBDAV_PASSWORD. With neither, nothing signs in.
	User, Password string
	// ReadOnly reads with GET and HEAD only, as any HTTP server serves them; nothing is written
	// or listed.
	ReadOnly bool
}

// WebDAV is a Store of a WebDAV folder: the object of key is the file URL/key. A file is
// written to a temporary name in its folder and moved into place, so that a reader never sees
// part of one.
type WebDAV struct {
	cfg        WebDAVConfig
	base       *url.URL // the folder, its path ending in /
	user, pass string
	client     *http.Client

	mu   sync.Mutex
	made map[string]bool // the folders known to be there
}

// WebDAVPasswordEnv and WebDAVUserEnv sign in to a WebDAV store the netrc does not name.
const (
	WebDAVUserEnv     = "HWNPU_WEBDAV_USER"
	WebDAVPasswordEnv = "HWNPU_WEBDAV_PASSWORD"
)

// OpenWebDAV makes a WebDAV store.
func OpenWebDAV(c WebDAVConfig) (*WebDAV, error) {
	u, err := url.Parse(c.URL)
	if err != nil {
		return nil, fmt.Errorf("store: %s: %w", c.Name, err)
	}
	if u.Scheme != "http" && u.Scheme != "https" || u.Host == "" {
		return nil, fmt.Errorf("store: %s: %q is not an http or https URL", c.Name, c.URL)
	}
	user, pass := c.User, c.Password
	if u.User != nil {
		user = cmp.Or(user, u.User.Username())
		if p, ok := u.User.Password(); ok {
			pass = cmp.Or(pass, p)
		}
	}
	u.User = nil
	u.RawQuery, u.Fragment, u.RawPath = "", "", ""
	if !strings.HasSuffix(u.Path, "/") {
		u.Path += "/"
	}
	c.URL = u.String()
	if pass == "" {
		login, p := netrcLogin(u.Hostname(), user)
		user, pass = cmp.Or(user, login), p
	}
	user = cmp.Or(user, os.Getenv(WebDAVUserEnv))
	pass = cmp.Or(pass, os.Getenv(WebDAVPasswordEnv))
	// the default client: the proxy of the environment, HTTP/2; no timeout, which would bound
	// a whole large download, but one for the connection and the answer's header
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.ResponseHeaderTimeout = 5 * time.Minute // a MOVE of a large file can take a while
	return &WebDAV{cfg: c, base: u, user: user, pass: pass, client: &http.Client{Transport: tr}, made: map[string]bool{}}, nil
}

func (w *WebDAV) String() string {
	if w.cfg.ReadOnly {
		return fmt.Sprintf("%s (%s, read only)", w.cfg.Name, w.cfg.URL)
	}
	return fmt.Sprintf("%s (webdav %s)", w.cfg.Name, w.cfg.URL)
}

func (w *WebDAV) writable() error {
	if w.cfg.ReadOnly {
		return fmt.Errorf("store: %s is read only (?writeback=true writes to it with WebDAV)", w)
	}
	return nil
}

// url is the URL of key (a key, or a folder of keys ending in /).
func (w *WebDAV) url(key string) string {
	u := *w.base
	u.Path += key
	return u.String()
}

func (w *WebDAV) checkKey(key string) error {
	if !SafeKey(key) {
		return fmt.Errorf("store: %q is not a key inside %s", key, w)
	}
	return nil
}

// davError is a request the server did not do.
type davError struct {
	Method, URL string
	StatusCode  int
	Status      string
}

func (e *davError) Error() string { return fmt.Sprintf("store: %s %s: %s", e.Method, e.URL, e.Status) }

// do makes a request (body made by body for each try, nil for none), retried as S3's are (5xx,
// 429, a cut connection). An answer of a status not in ok is a davError, its body closed.
func (w *WebDAV) do(ctx context.Context, method, u string, hdr http.Header, body func() (io.Reader, int64), ok ...int) (*http.Response, error) {
	var resp *http.Response
	err := retryIf(ctx, davTransient, func() error {
		var r io.Reader
		n := int64(0)
		if body != nil {
			r, n = body()
		}
		req, err := http.NewRequestWithContext(ctx, method, u, r)
		if err != nil {
			return err
		}
		if body != nil {
			req.ContentLength = n
			if n == 0 {
				req.Body = http.NoBody
			}
			req.GetBody = func() (io.ReadCloser, error) { r, _ := body(); return io.NopCloser(r), nil }
		}
		for k, v := range hdr {
			req.Header[k] = v
		}
		if w.user != "" || w.pass != "" {
			req.SetBasicAuth(w.user, w.pass)
		}
		resp, err = w.client.Do(req)
		if err != nil {
			return err
		}
		for _, s := range ok {
			if resp.StatusCode == s {
				return nil
			}
		}
		io.Copy(io.Discard, io.LimitReader(resp.Body, 64<<10))
		resp.Body.Close()
		return &davError{Method: method, URL: u, StatusCode: resp.StatusCode, Status: resp.Status}
	})
	if err != nil {
		var de *davError
		if errors.As(err, &de) && de.StatusCode == http.StatusUnauthorized {
			who := "no user"
			if w.user != "" {
				who = "the user " + w.user
			}
			return nil, fmt.Errorf("%w (signed in as %s: give the password in the URL, ~/.netrc (machine %s) or %s)", err, who, w.base.Hostname(), WebDAVPasswordEnv)
		}
		return nil, err
	}
	return resp, nil
}

func davTransient(err error) bool {
	var de *davError
	if errors.As(err, &de) {
		return de.StatusCode >= 500 && de.StatusCode != http.StatusNotImplemented || de.StatusCode == http.StatusTooManyRequests
	}
	var ne net.Error
	return errors.As(err, &ne) || errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, io.EOF)
}

func davStatus(err error, code int) bool {
	var de *davError
	return errors.As(err, &de) && de.StatusCode == code
}

// open reads key from offset: its body, the offset it starts at (0 when the server sent the
// whole file), and the whole size (-1 if not known).
func (w *WebDAV) open(ctx context.Context, key string, offset int64) (io.ReadCloser, int64, int64, error) {
	var hdr http.Header
	if offset > 0 {
		hdr = http.Header{"Range": {fmt.Sprintf("bytes=%d-", offset)}}
	}
	resp, err := w.do(ctx, http.MethodGet, w.url(key), hdr, nil, http.StatusOK, http.StatusPartialContent)
	if davStatus(err, http.StatusNotFound) {
		return nil, 0, 0, ErrNotFound
	}
	if davStatus(err, http.StatusRequestedRangeNotSatisfiable) && offset > 0 {
		return w.open(ctx, key, 0)
	}
	if err != nil {
		return nil, 0, 0, err
	}
	if resp.StatusCode == http.StatusPartialContent {
		total := int64(-1)
		if _, all, ok := strings.Cut(resp.Header.Get("Content-Range"), "/"); ok {
			if n, err := strconv.ParseInt(all, 10, 64); err == nil {
				total = n
			}
		}
		if total < 0 && resp.ContentLength >= 0 {
			total = offset + resp.ContentLength
		}
		return resp.Body, offset, total, nil
	}
	return resp.Body, 0, resp.ContentLength, nil // the server ignored the range
}

// Get downloads the file, going on from dst.part when a Get was cut off.
func (w *WebDAV) Get(ctx context.Context, key, dst string, bar func(done, total int64)) (int64, error) {
	if err := w.checkKey(key); err != nil {
		return 0, err
	}
	if err := os.MkdirAll(dirOf(dst), 0o755); err != nil {
		return 0, err
	}
	var size int64
	err := retryIf(ctx, davTransient, func() error {
		// the files are never changed in place: a part goes on from where it stopped
		var have int64
		if fi, err := os.Stat(dst + ".part"); err == nil {
			have = fi.Size()
		}
		if have > 0 {
			n, ok, err := w.Has(ctx, key)
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
		body, start, total, err := w.open(ctx, key, have)
		if err != nil {
			return err
		}
		defer body.Close()
		size, err = copyTo(ctx, body, dst, start, total, bar)
		return err
	})
	if errors.Is(err, ErrNotFound) {
		return 0, fmt.Errorf("store: %s: %s: %w", w.cfg.Name, key, ErrNotFound)
	}
	return size, err
}

func (w *WebDAV) Read(ctx context.Context, key string) ([]byte, error) {
	if err := w.checkKey(key); err != nil {
		return nil, err
	}
	var b []byte
	err := retryIf(ctx, davTransient, func() error {
		body, _, _, err := w.open(ctx, key, 0)
		if err != nil {
			return err
		}
		defer body.Close()
		b, err = io.ReadAll(io.LimitReader(body, 64<<20))
		return err
	})
	if errors.Is(err, ErrNotFound) {
		return nil, fmt.Errorf("store: %s: %s: %w", w.cfg.Name, key, ErrNotFound)
	}
	return b, err
}

// Has asks for the file's size with a PROPFIND of depth 0 (a HEAD's Content-Length is left out
// by some servers, and a HEAD of a folder looks like one of a file).
func (w *WebDAV) Has(ctx context.Context, key string) (int64, bool, error) {
	if err := w.checkKey(key); err != nil {
		return 0, false, err
	}
	if w.cfg.ReadOnly {
		resp, err := w.do(ctx, http.MethodHead, w.url(key), nil, nil, http.StatusOK)
		if davStatus(err, http.StatusNotFound) || davStatus(err, http.StatusForbidden) {
			return 0, false, nil
		}
		if err != nil {
			return 0, false, err
		}
		resp.Body.Close()
		return resp.ContentLength, true, nil
	}
	rs, err := w.propfind(ctx, key, "0")
	if errors.Is(err, ErrNotFound) {
		return 0, false, nil
	}
	if err != nil {
		return 0, false, err
	}
	for _, r := range rs {
		if !r.collection {
			return r.size, true, nil
		}
	}
	return 0, false, nil
}

// mkdirs makes the folders of dir (a slash path inside the store, "" for its root) that are
// not there, from the store's folder down.
func (w *WebDAV) mkdirs(ctx context.Context, dir string) error {
	if dir == "." {
		dir = ""
	}
	w.mu.Lock()
	done := w.made[dir]
	w.mu.Unlock()
	if done {
		return nil
	}
	u := w.url("")
	if dir != "" {
		if err := w.mkdirs(ctx, path.Dir(dir)); err != nil {
			return err
		}
		u = w.url(dir + "/")
	}
	// 201 made; 405 (Method Not Allowed) is there already; some servers answer 301 or 200 then
	_, err := w.do(ctx, "MKCOL", u, nil, nil, http.StatusCreated, http.StatusMethodNotAllowed, http.StatusOK, http.StatusMovedPermanently, http.StatusNoContent)
	if err != nil {
		return err
	}
	w.mu.Lock()
	w.made[dir] = true
	w.mu.Unlock()
	return nil
}

// put writes body (of size n, made again for each try) as key: to a temporary name in key's
// folder, moved into place.
func (w *WebDAV) put(ctx context.Context, key string, body func() (io.Reader, int64)) error {
	if err := w.writable(); err != nil {
		return err
	}
	if err := w.checkKey(key); err != nil {
		return err
	}
	dir := path.Dir(key)
	if dir == "." {
		dir = ""
	}
	var rnd [8]byte
	rand.Read(rnd[:])
	tmp := path.Join(dir, ".put-"+hex.EncodeToString(rnd[:]))
	ok := []int{http.StatusCreated, http.StatusNoContent, http.StatusOK}
	_, err := w.do(ctx, http.MethodPut, w.url(tmp), nil, body, ok...)
	if davStatus(err, http.StatusConflict) || davStatus(err, http.StatusNotFound) { // its folder is not there
		if err = w.mkdirs(ctx, dir); err == nil {
			_, err = w.do(ctx, http.MethodPut, w.url(tmp), nil, body, ok...)
		}
	}
	if err != nil {
		return err
	}
	hdr := http.Header{"Destination": {w.url(key)}, "Overwrite": {"T"}}
	if _, err := w.do(ctx, "MOVE", w.url(tmp), hdr, nil, http.StatusCreated, http.StatusNoContent, http.StatusOK); err != nil {
		w.do(context.WithoutCancel(ctx), http.MethodDelete, w.url(tmp), nil, nil, http.StatusNoContent, http.StatusOK, http.StatusNotFound)
		return err
	}
	return nil
}

// Put uploads the file in one request (a server may bound its size: nginx's
// client_max_body_size, IIS's maxAllowedContentLength).
func (w *WebDAV) Put(ctx context.Context, key, src string) error {
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
	return w.put(ctx, key, func() (io.Reader, int64) { return io.NewSectionReader(f, 0, size), size })
}

func (w *WebDAV) Write(ctx context.Context, key string, b []byte) error {
	return w.put(ctx, key, func() (io.Reader, int64) { return bytes.NewReader(b), int64(len(b)) })
}

// davEntry is a file or folder of a PROPFIND's answer: its path below the store's folder.
type davEntry struct {
	rel        string
	collection bool
	size       int64
}

type multistatus struct {
	Responses []struct {
		Href     string `xml:"href"`
		Propstat []struct {
			Status string `xml:"status"`
			Prop   struct {
				ResourceType struct {
					Collection *struct{} `xml:"collection"`
				} `xml:"resourcetype"`
				Length string `xml:"getcontentlength"`
			} `xml:"prop"`
		} `xml:"propstat"`
	} `xml:"response"`
}

const propfindBody = `<?xml version="1.0" encoding="utf-8"?>
<d:propfind xmlns:d="DAV:"><d:prop><d:resourcetype/><d:getcontentlength/></d:prop></d:propfind>`

// propfind lists what is at rel (a key, or a folder ending in /) at depth "0" or "1".
func (w *WebDAV) propfind(ctx context.Context, rel, depth string) ([]davEntry, error) {
	hdr := http.Header{"Depth": {depth}, "Content-Type": {`application/xml; charset="utf-8"`}}
	resp, err := w.do(ctx, "PROPFIND", w.url(rel), hdr, func() (io.Reader, int64) {
		return strings.NewReader(propfindBody), int64(len(propfindBody))
	}, http.StatusMultiStatus)
	if davStatus(err, http.StatusNotFound) || davStatus(err, http.StatusConflict) { // 409: a folder above is not there
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var ms multistatus
	if err := xml.NewDecoder(bufio.NewReader(resp.Body)).Decode(&ms); err != nil {
		return nil, fmt.Errorf("store: PROPFIND %s: %w", w.url(rel), err)
	}
	var out []davEntry
	for _, r := range ms.Responses {
		h, err := url.Parse(strings.TrimSpace(r.Href))
		if err != nil {
			continue
		}
		p, found := strings.CutPrefix(h.Path, w.base.Path)
		if !found {
			if h.Path+"/" == w.base.Path {
				p = ""
			} else {
				continue
			}
		}
		e := davEntry{rel: p, size: -1}
		for _, ps := range r.Propstat {
			if !strings.Contains(ps.Status, " 200") {
				continue
			}
			if ps.Prop.ResourceType.Collection != nil {
				e.collection = true
			}
			if n, err := strconv.ParseInt(strings.TrimSpace(ps.Prop.Length), 10, 64); err == nil {
				e.size = n
			}
		}
		if strings.HasSuffix(e.rel, "/") {
			e.collection = true
		}
		e.rel = strings.TrimSuffix(e.rel, "/")
		out = append(out, e)
	}
	return out, nil
}

// List walks the folders under prefix with PROPFINDs of depth 1 (depth infinity is often turned
// off).
func (w *WebDAV) List(ctx context.Context, prefix string) ([]string, error) {
	if err := w.writable(); err != nil {
		return nil, err // a server read with GETs has no listing
	}
	dir := strings.TrimSuffix(prefix, "/")
	if dir != "" && !SafeKey(dir) {
		return nil, fmt.Errorf("store: %q is not a prefix inside %s", prefix, w)
	}
	var keys []string
	var walk func(dir string) error
	walk = func(dir string) error {
		rel := ""
		if dir != "" {
			rel = dir + "/"
		}
		rs, err := w.propfind(ctx, rel, "1")
		if errors.Is(err, ErrNotFound) {
			return nil
		}
		if err != nil {
			return err
		}
		for _, r := range rs {
			if r.rel == dir || r.rel == "" || !strings.HasPrefix(r.rel, rel) {
				continue // the folder itself
			}
			name := path.Base(r.rel)
			switch {
			case r.collection:
				if err := walk(r.rel); err != nil {
					return err
				}
			case !strings.HasPrefix(name, ".put-"):
				keys = append(keys, r.rel)
			}
		}
		return nil
	}
	err := walk(dir)
	slices.Sort(keys)
	return keys, err
}

func (w *WebDAV) Delete(ctx context.Context, key string) error {
	if err := w.writable(); err != nil {
		return err
	}
	if err := w.checkKey(key); err != nil {
		return err
	}
	_, err := w.do(ctx, http.MethodDelete, w.url(key), nil, nil, http.StatusNoContent, http.StatusOK, http.StatusAccepted, http.StatusNotFound)
	return err
}

// netrcLogin reads the login and the password of host in the netrc (NETRC, else ~/.netrc;
// %USERPROFILE%\_netrc on Windows), as curl and git do: the first machine host whose login is
// user (any, for ""), else default's.
func netrcLogin(host, user string) (login, password string) {
	file := os.Getenv("NETRC")
	if file == "" {
		home, _ := os.UserHomeDir()
		file = filepath.Join(home, ".netrc")
		if runtime.GOOS == "windows" {
			if _, err := os.Stat(file); err != nil {
				file = filepath.Join(home, "_netrc")
			}
		}
	}
	b, err := os.ReadFile(file)
	if err != nil {
		return "", ""
	}
	return parseNetrc(string(b), host, user)
}

func parseNetrc(text, host, user string) (login, password string) {
	type machine struct {
		name, login, password string
		isDefault             bool
	}
	var ms []machine
	f := strings.Fields(text)
read:
	for i := 0; i < len(f); i++ {
		switch f[i] {
		case "machine":
			if i+1 < len(f) {
				i++
				ms = append(ms, machine{name: f[i]})
			}
		case "default":
			ms = append(ms, machine{isDefault: true})
		case "login", "password", "account":
			if i+1 < len(f) && len(ms) > 0 {
				i++
				switch f[i-1] {
				case "login":
					ms[len(ms)-1].login = f[i]
				case "password":
					ms[len(ms)-1].password = f[i]
				}
			}
		case "macdef": // a macro runs to an empty line: what follows is not read
			break read
		}
	}
	for _, def := range []bool{false, true} {
		for _, m := range ms {
			if m.isDefault == def && (def || m.name == host) && (user == "" || m.login == user) {
				return m.login, m.password
			}
		}
	}
	return "", ""
}
