package chat

import (
	"bufio"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu/internal/console"
)

// Gated repositories: a model whose repository on Hugging Face asks people to agree to its terms
// first (FLUX.1-schnell) answers 401 to a download without a token and 403 to one of an account
// that has not agreed. hfToken finds a token as huggingface_hub does (HF_TOKEN, then the token
// file `hf auth login` writes), else the one this process signed in for. With none, and a person
// at the terminal, signIn signs in to Hugging Face by OAuth's device flow (the app hwnpu,
// hfClientID, asks only for openid and gated-repos: reading public gated repositories the
// account has agreed to, nothing private) and keeps the token in memory only: hwnpu writes no
// token anywhere. A 403 sends the person to the repository's page to agree, then tries again.
// Without a terminal (hwnpu-chat serve, scripts), HF_TOKEN; the error says how to do it by hand.
// A 401 of a repository the hub's API does not call gated (a name that does not exist or is
// private) stays the download's error: no sign-in for it.

// hfClientID is hwnpu's public OAuth app on Hugging Face (no secret: device flow);
// HWNPU_HF_CLIENT_ID replaces it.
const hfClientID = "0b76f16c-f622-43f3-8b2c-b8ec136b77f4"

var (
	authMu       sync.Mutex
	oauthTok     string // the access token signIn got, for this process only
	oauthRefresh string // its refresh token, if Hugging Face gave one (also in memory only)

	// interactive reports whether a person can answer at the terminal; tests replace it, and
	// stdin and pollEvery with it.
	interactive           = func() bool { return console.Terminal(os.Stdin) && console.Terminal(os.Stderr) }
	stdin       io.Reader = os.Stdin
	pollEvery             = 5 * time.Second
)

// hfToken is the token to send to Hugging Face: HF_TOKEN; else the file HF_TOKEN_PATH names, or
// HF_HOME/token (~/.cache/huggingface/token), as `hf auth login` writes it; else the one signIn
// got; "" for none.
func hfToken() string {
	if t := strings.TrimSpace(os.Getenv("HF_TOKEN")); t != "" {
		return t
	}
	if b, err := os.ReadFile(hfTokenPath()); err == nil {
		if t := strings.TrimSpace(string(b)); t != "" {
			return t
		}
	}
	authMu.Lock()
	defer authMu.Unlock()
	return oauthTok
}

// hfTokenPath is where huggingface_hub keeps its token.
func hfTokenPath() string {
	if p := os.Getenv("HF_TOKEN_PATH"); p != "" {
		return p
	}
	home := os.Getenv("HF_HOME")
	if home == "" {
		if xdg := os.Getenv("XDG_CACHE_HOME"); xdg != "" {
			home = filepath.Join(xdg, "huggingface")
		} else if h, err := os.UserHomeDir(); err == nil {
			home = filepath.Join(h, ".cache", "huggingface")
		}
	}
	return filepath.Join(home, "token")
}

// isGated asks the hub's API, without a token, whether repo is a public gated repository.
func isGated(ctx context.Context, repo string) bool {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, hubURL+"api/models/"+repo, nil)
	if err != nil {
		return false
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return false
	}
	var info struct {
		Gated any `json:"gated"` // false, or "auto" / "manual"
	}
	if json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&info) != nil {
		return false
	}
	g, ok := info.Gated.(string)
	return ok && g != ""
}

// errGated is fetchHF's for a gated repository that the person could not, or did not, get access
// to here.
var errGated = errors.New("gated")

// gate is what fetchHF does on a 401 or 403 from repo (OWNER/NAME) of a gated repository: sign
// in, or send the person to agree to the terms; nil to try again. sent is the token the refused
// request carried. It returns an error that says what to do when no one is at the terminal.
func gate(ctx context.Context, repo string, status int, sent string, log io.Writer) error {
	page := strings.TrimSuffix(hubURL, "/") + "/" + repo
	if !interactive() {
		if status == http.StatusUnauthorized {
			return fmt.Errorf("chat: %s on Hugging Face asks you to agree to its terms before downloading: "+
				"agree at %s, then run this again at a terminal to sign in, or set HF_TOKEN to a read token "+
				"(https://huggingface.co/settings/tokens): %w", repo, page, errGated)
		}
		return fmt.Errorf("chat: the Hugging Face account of your token has not agreed to the terms of %s: "+
			"agree at %s and run this again: %w", repo, page, errGated)
	}
	authMu.Lock()
	defer authMu.Unlock()
	if status == http.StatusUnauthorized {
		if oauthTok != "" && oauthTok != sent {
			return nil // another download signed in meanwhile
		}
		if sent != "" && sent == oauthTok && oauthRefresh != "" {
			// hwnpu's app gives tokens of 15 minutes, less than a big download takes
			if tok, refresh, err := refreshToken(ctx); err == nil {
				oauthTok, oauthRefresh = tok, cmp.Or(refresh, oauthRefresh)
				return nil
			}
		}
		if sent != "" && sent != oauthTok {
			return fmt.Errorf("chat: Hugging Face does not take the token of HF_TOKEN or %s (401); "+
				"fix or remove it, or unset it to sign in here: %w", hfTokenPath(), errGated)
		}
		fmt.Fprintf(log, "%s on Hugging Face asks you to agree to its terms before downloading; "+
			"sign in to Hugging Face to download it (the token stays in this process, nothing is saved)\n", repo)
		tok, refresh, err := signIn(ctx, log)
		if err != nil {
			return err
		}
		oauthTok, oauthRefresh = tok, refresh
		return nil
	}
	fmt.Fprintf(log, "Your Hugging Face account has not agreed to the terms of %s yet.\n"+
		"Open %s , agree there, then press Enter here (Ctrl-C to stop): ", repo, page)
	line := make(chan error, 1)
	go func() {
		_, err := bufio.NewReader(stdin).ReadString('\n')
		line <- err
	}()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case err := <-line:
		if err != nil {
			return fmt.Errorf("chat: agreeing to the terms of %s: %w (%w)", repo, err, errGated)
		}
	}
	return nil
}

// signIn runs OAuth's device flow with Hugging Face: shows the code to enter at the page it
// names, waits for the person to approve, and returns the access token and the refresh token
// ("" when Hugging Face gives none).
func signIn(ctx context.Context, log io.Writer) (string, string, error) {
	client := clientID()
	var dev struct {
		DeviceCode      string `json:"device_code"`
		UserCode        string `json:"user_code"`
		VerificationURI string `json:"verification_uri"`
		Complete        string `json:"verification_uri_complete"`
		ExpiresIn       int    `json:"expires_in"`
		Interval        int    `json:"interval"`
	}
	if err := postForm(ctx, "oauth/device", url.Values{"client_id": {client}, "scope": {"openid gated-repos"}}, &dev); err != nil {
		return "", "", fmt.Errorf("chat: signing in to Hugging Face: %w", err)
	}
	if dev.DeviceCode == "" || dev.UserCode == "" {
		return "", "", fmt.Errorf("chat: signing in to Hugging Face: no device code in the answer")
	}
	where := cmp.Or(dev.Complete, dev.VerificationURI)
	fmt.Fprintf(log, "Open %s and enter the code %s (Hugging Face shows hwnpu asking to read gated repositories you agreed to)\n",
		where, console.Highlight(dev.UserCode))
	every := pollEvery
	if dev.Interval > 0 && time.Duration(dev.Interval)*time.Second > every {
		every = time.Duration(dev.Interval) * time.Second
	}
	deadline := time.Now().Add(time.Duration(max(dev.ExpiresIn, 60)) * time.Second)
	for time.Now().Before(deadline) {
		select {
		case <-ctx.Done():
			return "", "", ctx.Err()
		case <-time.After(every):
		}
		var tok tokenAnswer
		err := postForm(ctx, "oauth/token", url.Values{
			"grant_type":  {"urn:ietf:params:oauth:grant-type:device_code"},
			"device_code": {dev.DeviceCode},
			"client_id":   {client},
		}, &tok)
		var he *httpError
		if errors.As(err, &he) {
			json.Unmarshal(he.body, &tok)
		} else if err != nil {
			return "", "", fmt.Errorf("chat: signing in to Hugging Face: %w", err)
		}
		switch {
		case tok.AccessToken != "":
			fmt.Fprintln(log, "signed in to Hugging Face")
			return tok.AccessToken, tok.RefreshToken, nil
		case tok.Error == "authorization_pending":
		case tok.Error == "slow_down":
			every += 5 * time.Second
		case tok.Error != "":
			return "", "", fmt.Errorf("chat: signing in to Hugging Face: %s %s (%w)", tok.Error, tok.Description, errGated)
		default:
			return "", "", fmt.Errorf("chat: signing in to Hugging Face: %w", err)
		}
	}
	return "", "", fmt.Errorf("chat: signing in to Hugging Face: the code expired; run this again (%w)", errGated)
}

// tokenAnswer is the token endpoint's answer.
type tokenAnswer struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	Error        string `json:"error"`
	Description  string `json:"error_description"`
}

// refreshToken trades oauthRefresh for a new access token (and refresh token); authMu is held.
func refreshToken(ctx context.Context) (string, string, error) {
	var tok tokenAnswer
	err := postForm(ctx, "oauth/token", url.Values{
		"grant_type":    {"refresh_token"},
		"refresh_token": {oauthRefresh},
		"client_id":     {clientID()},
	}, &tok)
	if err != nil {
		return "", "", err
	}
	if tok.AccessToken == "" {
		return "", "", fmt.Errorf("no access token in the answer")
	}
	return tok.AccessToken, tok.RefreshToken, nil
}

// clientID is the OAuth app to sign in with: HWNPU_HF_CLIENT_ID, else hwnpu's.
func clientID() string { return cmp.Or(os.Getenv("HWNPU_HF_CLIENT_ID"), hfClientID) }

// httpError is postForm's for an answer that is not 2xx, with its body.
type httpError struct {
	status string
	body   []byte
}

func (e *httpError) Error() string { return e.status + ": " + strings.TrimSpace(string(e.body)) }

// postForm posts form to the hub's path and decodes the JSON answer into out.
func postForm(ctx context.Context, path string, form url.Values, out any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, hubURL+path, strings.NewReader(form.Encode()))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<16))
	if err != nil {
		return err
	}
	if resp.StatusCode/100 != 2 {
		return &httpError{resp.Status, body}
	}
	return json.Unmarshal(body, out)
}
