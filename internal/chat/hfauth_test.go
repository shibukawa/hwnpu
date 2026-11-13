package chat

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// gatedHub serves owner/gated as a gated repository does: 401 without a token, 403 until
// agreed is set, the file after; with OAuth's device and token endpoints (a code approved on the
// second poll, access tokens "acc1", "acc2"... that refresh tokens renew).
func gatedHub(t *testing.T, agreed *atomic.Bool, valid *atomic.Value) *atomic.Int32 {
	t.Helper()
	var polls, issued atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/models/owner/gated":
			fmt.Fprint(w, `{"id":"owner/gated","gated":"auto"}`)
		case "/api/models/owner/missing":
			http.Error(w, `{"error":"Repository not found"}`, http.StatusUnauthorized)
		case "/owner/missing/resolve/main/weights.bin":
			http.Error(w, "unauthorized", http.StatusUnauthorized)
		case "/oauth/device":
			r.ParseForm()
			if r.Form.Get("client_id") != hfClientID || r.Form.Get("scope") != "openid gated-repos" {
				http.Error(w, `{"error":"invalid_client"}`, 400)
				return
			}
			fmt.Fprint(w, `{"device_code":"dev","user_code":"ABCD-1234","verification_uri":"https://hf.co/oauth/device","expires_in":300}`)
		case "/oauth/token":
			r.ParseForm()
			switch r.Form.Get("grant_type") {
			case "urn:ietf:params:oauth:grant-type:device_code":
				if polls.Add(1) < 2 {
					w.WriteHeader(400)
					fmt.Fprint(w, `{"error":"authorization_pending"}`)
					return
				}
			case "refresh_token":
				if r.Form.Get("refresh_token") != "ref" {
					http.Error(w, `{"error":"invalid_grant"}`, 400)
					return
				}
			}
			tok := fmt.Sprintf("acc%d", issued.Add(1))
			valid.Store(tok)
			fmt.Fprintf(w, `{"access_token":%q,"refresh_token":"ref","token_type":"bearer","expires_in":900}`, tok)
		case "/owner/gated/resolve/main/weights.bin":
			switch {
			case r.Header.Get("Authorization") != "Bearer "+valid.Load().(string):
				http.Error(w, "unauthorized", http.StatusUnauthorized)
			case !agreed.Load():
				http.Error(w, "not agreed", http.StatusForbidden)
			default:
				fmt.Fprint(w, "weights")
			}
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	old := hubURL
	hubURL = srv.URL + "/"
	t.Cleanup(func() { hubURL = old })
	return &issued
}

// signedOut makes the test see no token of the person's and no earlier sign-in.
func signedOut(t *testing.T, terminal bool, in string) {
	t.Setenv("HF_TOKEN", "")
	t.Setenv("HF_TOKEN_PATH", filepath.Join(t.TempDir(), "none"))
	t.Setenv("HWNPU_HF_CLIENT_ID", "")
	oi, os_, op := interactive, stdin, pollEvery
	interactive, stdin, pollEvery = func() bool { return terminal }, strings.NewReader(in), time.Millisecond
	oauthTok, oauthRefresh = "", ""
	t.Cleanup(func() { interactive, stdin, pollEvery = oi, os_, op; oauthTok, oauthRefresh = "", "" })
}

func TestGatedSignInAndAgree(t *testing.T) {
	var agreed atomic.Bool
	var valid atomic.Value
	valid.Store("-")
	gatedHub(t, &agreed, &valid)
	r, w, _ := os.Pipe()
	signedOut(t, true, "")
	stdin = r
	go func() { // the person agrees on the page, then presses Enter
		time.Sleep(50 * time.Millisecond)
		agreed.Store(true)
		w.Write([]byte("\n"))
	}()
	var log strings.Builder
	path := filepath.Join(t.TempDir(), "weights.bin")
	if err := fetchHF(context.Background(), "owner/gated", "weights.bin", path, &log); err != nil {
		t.Fatalf("fetchHF: %v\n%s", err, log.String())
	}
	if b, _ := os.ReadFile(path); string(b) != "weights" {
		t.Fatalf("got %q", b)
	}
	for _, want := range []string{"ABCD-1234", "signed in", "agree there"} {
		if !strings.Contains(log.String(), want) {
			t.Errorf("log lacks %q:\n%s", want, log.String())
		}
	}
	if oauthTok != "acc1" {
		t.Errorf("token %q", oauthTok)
	}
}

func TestGatedRefresh(t *testing.T) {
	var agreed atomic.Bool
	agreed.Store(true)
	var valid atomic.Value
	issued := gatedHub(t, &agreed, &valid)
	signedOut(t, true, "")
	oauthTok, oauthRefresh = "expired", "ref" // signed in earlier; the token ran out
	valid.Store("-")
	var log strings.Builder
	path := filepath.Join(t.TempDir(), "weights.bin")
	if err := fetchHF(context.Background(), "owner/gated", "weights.bin", path, &log); err != nil {
		t.Fatalf("fetchHF: %v\n%s", err, log.String())
	}
	if issued.Load() != 1 || strings.Contains(log.String(), "enter the code") {
		t.Errorf("refresh should renew without a new sign-in (%d tokens):\n%s", issued.Load(), log.String())
	}
}

func TestGatedNoTerminal(t *testing.T) {
	var agreed atomic.Bool
	var valid atomic.Value
	valid.Store("-")
	gatedHub(t, &agreed, &valid)
	signedOut(t, false, "")
	path := filepath.Join(t.TempDir(), "weights.bin")
	err := fetchHF(context.Background(), "owner/gated", "weights.bin", path, nil)
	if !errors.Is(err, errGated) || !strings.Contains(err.Error(), "HF_TOKEN") || !strings.Contains(err.Error(), "/owner/gated") {
		t.Fatalf("err %v", err)
	}
	// with a token of an account that has not agreed: a 403 that names the page
	t.Setenv("HF_TOKEN", "tok")
	valid.Store("tok")
	err = fetchHF(context.Background(), "owner/gated", "weights.bin", path, nil)
	if !errors.Is(err, errGated) || !strings.Contains(err.Error(), "agree at") {
		t.Fatalf("err %v", err)
	}
	agreed.Store(true)
	if err := fetchHF(context.Background(), "owner/gated", "weights.bin", path, nil); err != nil {
		t.Fatal(err)
	}
}

func TestNotGated401(t *testing.T) {
	// a repository that does not exist (or is private) answers 401 too: no sign-in for it
	var agreed atomic.Bool
	var valid atomic.Value
	valid.Store("-")
	gatedHub(t, &agreed, &valid)
	signedOut(t, true, "")
	var log strings.Builder
	err := fetchHF(context.Background(), "owner/missing", "weights.bin", filepath.Join(t.TempDir(), "w"), &log)
	if err == nil || errors.Is(err, errGated) || !strings.Contains(err.Error(), "401") || log.Len() != 0 {
		t.Fatalf("err %v, log %q", err, log.String())
	}
}
