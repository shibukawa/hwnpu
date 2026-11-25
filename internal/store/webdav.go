package store

import (
	"bufio"
	"bytes"
	"cmp"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
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

// WebDAVConfig is where a WebDAV folder is and who signs in to it (docs/cache-sharing-design.md 7).
type WebDAVConfig struct {
	Name string
	// URL is the folder: https://nas:5006/hwnpu (Synology), https://cloud/remote.php/dav/files/USER/hwnpu
	// (Nextcloud). Its user name and password, if any, sign in.
	URL string
	// User and Password sign in with Basic authentication (over https). With no user: the
	// URL's, else the netrc's login for the host, else HWNPU_WEBDAV_USER; with no password: the
	// URL's, else the netrc's, else HWNPU_WEBDAV_PASSWORD. With neither, nothing signs in.
	User, Password string
	// TLSSHA256 is the SHA-256 of the server's certificate (hex, colons allowed) to trust instead
	// of the system's roots: a NAS's own certificate, which no authority signed and which names
	// no host (?tls_sha256= of the URL). The certificate is checked, the host name is not.
	TLSSHA256 string
	// Insecure trusts any certificate (?insecure=true of the URL): neither it nor the host name
	// is checked. TLSSHA256 is the safer way to trust a NAS's own certificate.
	Insecure bool
	// ReadOnly reads with GET and HEAD only, as any HTTP server serves them; nothing is written
	// or listed.
	ReadOnly bool
	// Bearer, when set, signs in with the token it returns ("": none yet) instead of a user and
	// password (no netrc is read): Hugging Face (Hub).
	Bearer func() string
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
	mk   sync.Mutex      // held while folders are made: the puts side by side make the same ones
	down error           // the server could not be reached: every request fails at once with it
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
	if c.Bearer == nil {
		if pass == "" {
			login, p := netrcLogin(u.Hostname(), user)
			user, pass = cmp.Or(user, login), p
		}
		user = cmp.Or(user, os.Getenv(WebDAVUserEnv))
		pass = cmp.Or(pass, os.Getenv(WebDAVPasswordEnv))
	}
	// the default client: the proxy of the environment, HTTP/2; no timeout, which would bound
	// a whole large download, but one for the connection and the answer's header
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.ResponseHeaderTimeout = 5 * time.Minute // a MOVE of a large file can take a while
	// a NAS that is off is found out soon (not after the default 30 seconds, for each try), and
	// its name is looked up once: a .local name takes 5 seconds a lookup on macOS (mDNS)
	tr.DialContext = (&cachedDialer{d: net.Dialer{Timeout: davDialTimeout, KeepAlive: 30 * time.Second}}).DialContext
	tr.TLSHandshakeTimeout = davDialTimeout
	tr.MaxIdleConnsPerHost = 8 // the queue's jobs go side by side
	switch {
	case c.TLSSHA256 != "":
		want, err := hex.DecodeString(strings.ReplaceAll(strings.ToLower(c.TLSSHA256), ":", ""))
		if err != nil || len(want) != sha256.Size {
			return nil, fmt.Errorf("store: %s: tls_sha256 %q is not a SHA-256 in hex", c.Name, c.TLSSHA256)
		}
		tr.TLSClientConfig = &tls.Config{
			InsecureSkipVerify: true, // the pinned certificate is checked below instead
			VerifyConnection: func(cs tls.ConnectionState) error {
				if len(cs.PeerCertificates) == 0 {
					return fmt.Errorf("store: %s: the server gave no certificate", c.Name)
				}
				if got := sha256.Sum256(cs.PeerCertificates[0].Raw); !bytes.Equal(got[:], want) {
					return &pinError{fmt.Errorf("store: %s: the server's certificate has the SHA-256 %x, not the tls_sha256 %x", c.Name, got, want)}
				}
				return nil
			},
		}
	case c.Insecure:
		tr.TLSClientConfig = &tls.Config{InsecureSkipVerify: true}
	}
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

// davStall is how long a download may go without a byte, its answer's headers too, before it is
// cut and tried again from where it stopped.
var davStall = 60 * time.Second

// A download that goes on but has slowed to a trickle (a NAS once sent 0.3 GB in 48 min of a
// pack it had sent at 79 MB/s, and a new connection got the rest in a minute) is cut too, and
// gone on from where it stopped: when a davStall brings less than 1/davSlowRatio of the most
// one has brought in it, once one has brought davSlowFloor. A line that is slow all along is
// never cut.
var (
	davSlowRatio int64 = 16
	davSlowFloor int64 = 64 << 20
)

// stallError is a download cut for davStall without a byte, or slowed to a trickle: a
// timeout, tried again.
type stallError struct {
	err error
	// slow is the bytes of the davStall it was cut in, and best the most of one before it, when
	// it was cut as slow
	slow, best int64
	isSlow     bool
}

func (e *stallError) Error() string {
	if e.isSlow {
		return fmt.Sprintf("%.2f MB in %s, after %.2f MB in one before: %v", float64(e.slow)/1e6, davStall, float64(e.best)/1e6, e.err)
	}
	return fmt.Sprintf("no data for %s: %v", davStall, e.err)
}
func (e *stallError) Unwrap() error   { return e.err }
func (e *stallError) Timeout() bool   { return true }
func (e *stallError) Temporary() bool { return true }

// stalled is err as a stallError when it is of rctx cut by its watch (w), not of ctx.
func stalled(ctx, rctx context.Context, w *watched, err error) error {
	if err != nil && ctx.Err() == nil && rctx.Err() != nil {
		e := &stallError{err: err}
		if w != nil {
			w.mu.Lock()
			e.isSlow, e.slow, e.best = w.cutSlow, w.cutAt, w.best
			w.mu.Unlock()
		}
		return e
	}
	return err
}

// watched is a body that puts its watch off by davStall at each byte read, and cuts it (cancel)
// at the end of a davStall that brought too little (davSlowRatio).
type watched struct {
	r      io.Reader
	idle   *time.Timer
	cancel func()

	mu       sync.Mutex
	window   int64 // the bytes read in this davStall
	best     int64 // the most of one before
	cutSlow  bool
	cutAt    int64
	interval *time.Timer
	stopped  bool
}

// watch starts w's measure of each davStall; stop ends it.
func (w *watched) watch() {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.interval = time.AfterFunc(davStall, w.tick)
}

func (w *watched) stop() {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.stopped = true
	if w.interval != nil {
		w.interval.Stop()
	}
}

func (w *watched) tick() {
	w.mu.Lock()
	if w.stopped {
		w.mu.Unlock()
		return
	}
	n := w.window
	w.window = 0
	slow := w.best >= davSlowFloor && n*davSlowRatio < w.best
	if slow {
		w.cutSlow, w.cutAt = true, n
	} else {
		w.best = max(w.best, n)
		w.interval.Reset(davStall)
	}
	w.mu.Unlock()
	if slow {
		w.cancel()
	}
}

func (w *watched) Read(p []byte) (int, error) {
	n, err := w.r.Read(p)
	if n > 0 {
		w.idle.Reset(davStall)
		w.mu.Lock()
		w.window += int64(n)
		w.mu.Unlock()
	}
	return n, err
}

// davDialTimeout bounds connecting to a server (its name's lookup too) and its TLS handshake.
var davDialTimeout = 15 * time.Second

// cachedDialer dials with the addresses of a host looked up once.
type cachedDialer struct {
	d     net.Dialer
	mu    sync.Mutex
	addrs map[string][]string // host: its addresses
}

func (c *cachedDialer) DialContext(ctx context.Context, network, addr string) (net.Conn, error) {
	host, port, err := net.SplitHostPort(addr)
	if err != nil || net.ParseIP(host) != nil {
		return c.d.DialContext(ctx, network, addr)
	}
	c.mu.Lock()
	ips, ok := c.addrs[host]
	c.mu.Unlock()
	if !ok {
		lctx, cancel := context.WithTimeout(ctx, davDialTimeout)
		ips, err = net.DefaultResolver.LookupHost(lctx, host)
		cancel()
		if err != nil {
			return nil, &net.OpError{Op: "dial", Net: network, Err: err}
		}
		c.mu.Lock()
		if c.addrs == nil {
			c.addrs = map[string][]string{}
		}
		c.addrs[host] = ips
		c.mu.Unlock()
	}
	var first error
	for _, ip := range ips {
		conn, err := c.d.DialContext(ctx, network, net.JoinHostPort(ip, port))
		if err == nil {
			return conn, nil
		}
		first = cmp.Or(first, err)
	}
	return nil, first
}

// do makes a request (body made by body for each try, nil for none), retried as S3's are (5xx,
// 429, a cut connection). An answer of a status not in ok is a davError, its body closed.
//
// A server that cannot be reached (no such host, the connection refused, a certificate not
// trusted, or no answer in any of the tries) is not tried again in this process: a NAS that is off is a source
// that is not there, and every request after fails at once (the sources after it are read, and
// what goes in it stays in the queue for the next run).
func (w *WebDAV) do(ctx context.Context, method, u string, hdr http.Header, body func() (io.Reader, int64), ok ...int) (*http.Response, error) {
	w.mu.Lock()
	down := w.down
	w.mu.Unlock()
	if down != nil {
		return nil, down
	}
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
		switch {
		case w.cfg.Bearer != nil:
			if t := w.cfg.Bearer(); t != "" {
				req.Header.Set("Authorization", "Bearer "+t)
			}
		case w.user != "" || w.pass != "":
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
		if errors.As(err, &de) && de.StatusCode == http.StatusUnauthorized && w.cfg.Bearer == nil {
			who := "no user"
			if w.user != "" {
				who = "the user " + w.user
			}
			return nil, fmt.Errorf("%w (signed in as %s: give the password in the URL, ~/.netrc (machine %s) or %s)", err, who, w.base.Hostname(), WebDAVPasswordEnv)
		}
		if unreachable(err) && ctx.Err() == nil {
			err = &unreachableError{name: w.cfg.Name, host: w.base.Host, err: err}
			w.mu.Lock()
			if w.down == nil {
				w.down = err
				w.mu.Unlock()
				fmt.Fprintf(os.Stderr, "store: %s (%s) cannot be reached; read on as if it were not set this run: %v\n", w.cfg.Name, w.base.Host, errors.Unwrap(err))
			} else {
				err = w.down
				w.mu.Unlock()
			}
		}
		return nil, err
	}
	return resp, nil
}

// unreachableError is a server do could not reach.
type unreachableError struct {
	name, host string
	err        error
}

func (e *unreachableError) Error() string {
	return fmt.Sprintf("store: %s (%s) cannot be reached: %v", e.name, e.host, e.err)
}
func (e *unreachableError) Unwrap() error { return e.err }

// pinError is a certificate that is not the one of tls_sha256.
type pinError struct{ error }

func (e *pinError) Unwrap() error { return e.error }

// unreachable reports an error of reaching the server at all: its name, a connection, the TLS
// handshake (not one cut off later, which a try again goes on from).
func unreachable(err error) bool {
	var dns *net.DNSError
	var op *net.OpError
	var cert *tls.CertificateVerificationError
	var rec tls.RecordHeaderError
	var ue *unreachableError
	var pin *pinError
	return errors.As(err, &ue) || errors.As(err, &pin) || errors.As(err, &dns) || errors.As(err, &op) && op.Op == "dial" ||
		errors.As(err, &cert) || errors.As(err, &rec) || tlsHandshakeTimeout(err)
}

func tlsHandshakeTimeout(err error) bool {
	return err != nil && strings.Contains(err.Error(), "TLS handshake timeout")
}

// call is do for an answer whose body is not wanted: read to its end and closed, so that its
// connection is used again (a connection left open is not, and each new one asks for the host's
// address again: 5 seconds a request for a NAS's .local name, docs/design/distribution.md).
func (w *WebDAV) call(ctx context.Context, method, u string, hdr http.Header, body func() (io.Reader, int64), ok ...int) error {
	resp, err := w.do(ctx, method, u, hdr, body, ok...)
	if err != nil {
		return err
	}
	io.Copy(io.Discard, io.LimitReader(resp.Body, 64<<10))
	return resp.Body.Close()
}

func davTransient(err error) bool {
	var ue *unreachableError
	if errors.As(err, &ue) {
		return false
	}
	if unreachable(err) {
		// a lookup or a connection that only timed out (a .local name over mDNS does now and
		// then) is tried again before the server is taken for off; one refused is not
		var ne net.Error
		return errors.As(err, &ne) && ne.Timeout()
	}
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
	partSize := func() int64 {
		if fi, err := os.Stat(dst + ".part"); err == nil {
			return fi.Size()
		}
		return 0
	}
	// the tries are of a download that does not get on: one cut (stalled, slowed) after it
	// brought more of the file starts them again, so that a long download that slows now and
	// then is not given up
	for {
		before := partSize()
		err := w.get(ctx, key, dst, bar, &size)
		var se *stallError
		if err == nil || !errors.As(err, &se) || ctx.Err() != nil || partSize() <= before {
			if errors.Is(err, ErrNotFound) {
				return 0, fmt.Errorf("store: %s: %s: %w", w.cfg.Name, key, ErrNotFound)
			}
			return size, err
		}
	}
}

// get is one round of Get's tries.
func (w *WebDAV) get(ctx context.Context, key, dst string, bar func(done, total int64), size *int64) error {
	return retryIf(ctx, davTransient, func() error {
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
				*size = n
				return os.Rename(dst+".part", dst)
			case have > n:
				os.Remove(dst + ".part")
				have = 0
			}
		}
		// a server that stops sending (or a connection that died quietly) is cut after
		// davStall, and one that slows to a trickle at the end of a davStall (watched), and gone
		// on from where it stopped, not waited on for ever
		rctx, cancel := context.WithCancel(ctx)
		defer cancel()
		idle := time.AfterFunc(davStall, cancel)
		defer idle.Stop()
		body, start, total, err := w.open(rctx, key, have)
		if err != nil {
			return stalled(ctx, rctx, nil, err)
		}
		defer body.Close()
		wb := &watched{r: body, idle: idle, cancel: cancel}
		wb.watch()
		defer wb.stop()
		var n int64
		n, err = copyTo(ctx, wb, dst, start, total, bar)
		*size = n
		return stalled(ctx, rctx, wb, err)
	})
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
// not there, from the store's folder down, one put at a time (two MKCOLs of one folder at once
// are refused: 423 Locked).
func (w *WebDAV) mkdirs(ctx context.Context, dir string) error {
	w.mk.Lock()
	defer w.mk.Unlock()
	return w.mkdirsLocked(ctx, dir)
}

func (w *WebDAV) mkdirsLocked(ctx context.Context, dir string) error {
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
		if err := w.mkdirsLocked(ctx, path.Dir(dir)); err != nil {
			return err
		}
		u = w.url(dir + "/")
	}
	// 201 made; 405 (Method Not Allowed) is there already (or, on a UGREEN NAS, its parent is
	// not: the parents are made first); some servers answer 301 or 200 then
	err := w.call(ctx, "MKCOL", u, nil, nil, http.StatusCreated, http.StatusMethodNotAllowed, http.StatusOK, http.StatusMovedPermanently, http.StatusNoContent)
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
	// the folder first, once a process, so that a large file is not sent twice; and again when
	// a PUT says it is not there (409, 404; 405 on a UGREEN NAS), made since by another
	if err := w.mkdirs(ctx, dir); err != nil {
		return err
	}
	err := w.call(ctx, http.MethodPut, w.url(tmp), nil, body, ok...)
	if davStatus(err, http.StatusConflict) || davStatus(err, http.StatusNotFound) || davStatus(err, http.StatusMethodNotAllowed) {
		w.mu.Lock()
		delete(w.made, dir)
		w.mu.Unlock()
		if err = w.mkdirs(ctx, dir); err == nil {
			err = w.call(ctx, http.MethodPut, w.url(tmp), nil, body, ok...)
		}
	}
	if err != nil {
		return err
	}
	hdr := http.Header{"Destination": {w.url(key)}, "Overwrite": {"T"}}
	if err := w.call(ctx, "MOVE", w.url(tmp), hdr, nil, http.StatusCreated, http.StatusNoContent, http.StatusOK); err != nil {
		w.call(context.WithoutCancel(ctx), http.MethodDelete, w.url(tmp), nil, nil, http.StatusNoContent, http.StatusOK, http.StatusNotFound)
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
	mod        time.Time // zero when the server did not say
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
				Length   string `xml:"getcontentlength"`
				Modified string `xml:"getlastmodified"`
			} `xml:"prop"`
		} `xml:"propstat"`
	} `xml:"response"`
}

const propfindBody = `<?xml version="1.0" encoding="utf-8"?>
<d:propfind xmlns:d="DAV:"><d:prop><d:resourcetype/><d:getcontentlength/><d:getlastmodified/></d:prop></d:propfind>`

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
			if t, err := http.ParseTime(strings.TrimSpace(ps.Prop.Modified)); err == nil {
				e.mod = t
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
	ls, err := w.list(ctx, prefix, false)
	return keysOf(ls), err
}

// listAll is List with the temporary files of uploads (.put-) too, and when each was last
// changed (Clean).
func (w *WebDAV) listAll(ctx context.Context, prefix string) ([]listed, error) {
	return w.list(ctx, prefix, true)
}

func (w *WebDAV) list(ctx context.Context, prefix string, all bool) ([]listed, error) {
	if err := w.writable(); err != nil {
		return nil, err // a server read with GETs has no listing
	}
	dir := strings.TrimSuffix(prefix, "/")
	if dir != "" && !SafeKey(dir) {
		return nil, fmt.Errorf("store: %q is not a prefix inside %s", prefix, w)
	}
	var keys []listed
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
			case all || !strings.HasPrefix(name, ".put-"):
				keys = append(keys, listed{r.rel, r.mod})
			}
		}
		return nil
	}
	err := walk(dir)
	slices.SortFunc(keys, func(a, b listed) int { return strings.Compare(a.key, b.key) })
	return keys, err
}

func (w *WebDAV) Delete(ctx context.Context, key string) error {
	if err := w.writable(); err != nil {
		return err
	}
	if err := w.checkKey(key); err != nil {
		return err
	}
	return w.call(ctx, http.MethodDelete, w.url(key), nil, nil, http.StatusNoContent, http.StatusOK, http.StatusAccepted, http.StatusNotFound)
}

// removeEmpty removes the folder dir if no file is under it (empty folders may be: a DELETE of
// a folder removes what is in it too, so it is listed first, to its depth).
func (w *WebDAV) removeEmpty(ctx context.Context, dir string) bool {
	if w.writable() != nil || w.checkKey(dir) != nil {
		return false
	}
	if files, err := w.list(ctx, dir+"/", true); err != nil || len(files) > 0 {
		return false
	}
	_, err := w.do(ctx, http.MethodDelete, w.url(dir+"/"), nil, nil, http.StatusNoContent, http.StatusOK, http.StatusAccepted, http.StatusNotFound)
	return err == nil
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
