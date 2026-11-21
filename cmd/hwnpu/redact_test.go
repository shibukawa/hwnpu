package main

import (
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/npustatus"
)

func TestRedact(t *testing.T) {
	home, _ := os.UserHomeDir()
	t.Setenv("HWNPU_MODEL_SOURCE", "https://develop:Develop@nas.example.lan:5006/llm?writeback=true")
	t.Setenv("HWNPU_MODEL_SOURCE_1", "file:///srv/models")
	r := newRedactor()
	in := strings.Join([]string{
		filepath.Join(home, ".cache", "hwnpu"),
		"webdav https://nas.example.lan:5006/llm/",
		"GET https://develop:Develop@nas.example.lan:5006/llm/x",
		"http://192.168.1.20:9000/bucket",
		"https://huggingface.co/LiquidAI/LFM3-1.7B/resolve/main/config.json",
		"token=abc123 HF_TOKEN: hf_abcdefghijklmnop Bearer xyz.abc",
		"AKIAABCDEFGHIJKLMNOP mail me@example.com",
		"firmware 1.1.2.64 at /srv/models/original",
	}, "\n")
	out := r.hide(in)
	for _, bad := range []string{home, "nas.example.lan", "Develop@", "192.168.1.20", "abc123", "hf_abcdefghijklmnop", "xyz.abc", "AKIAABCDEFGHIJKLMNOP", "me@example.com", "/srv/models"} {
		if strings.Contains(out, bad) {
			t.Errorf("%q left in:\n%s", bad, out)
		}
	}
	for _, good := range []string{"~/.cache/hwnpu", "https://huggingface.co/LiquidAI/LFM3-1.7B", "firmware 1.1.2.64", "source-host-"} {
		if !strings.Contains(out, good) {
			t.Errorf("%q not in:\n%s", good, out)
		}
	}
	a := r.args([]string{"hwnpu-chat", "-m", "LFM3-1.7B", "please translate this long sentence into Japanese for me", "x y"})
	if a != "hwnpu-chat -m LFM3-1.7B <text of 56 characters> 'x y'" {
		t.Errorf("args %q", a)
	}
}

// TestIssueURL fills the title and the body of GitHub's new issue page.
func TestIssueURL(t *testing.T) {
	u, err := url.Parse(issueURL("a & b", "line 1\n## x"))
	if err != nil {
		t.Fatal(err)
	}
	q := u.Query()
	if u.Host != "github.com" || u.Path != "/"+Repo+"/issues/new" || q.Get("title") != "a & b" || q.Get("body") != "line 1\n## x" {
		t.Errorf("%s", u)
	}
}

// TestDraft writes the command, its error and log, and the machine, without the command lines
// of the processes on the NPU.
func TestDraft(t *testing.T) {
	st := npustatus.Status{Version: "v1", OS: "linux", Arch: "amd64", Vendor: "amd",
		Processes: []npustatus.Process{{PID: 7, Name: "hwnpu-chat", Args: []string{"hwnpu-chat", "a secret prompt"}, NPUMemory: 2e9}}}
	run := &console.Run{Args: []string{"hwnpu-image", "image", "-m", "anima"}, Error: "step 3: boom",
		Start: time.Now().Add(-time.Minute), End: time.Now(), Lines: []string{"one", "two", "three"}}
	title, body := draft(st, run, "", 2, newRedactor())
	if title != "hwnpu-image image: step 3: boom" {
		t.Errorf("title %q", title)
	}
	for _, want := range []string{"hwnpu-image image -m anima", "step 3: boom", "Log (2 of 3 lines)", "two\nthree", "      7  hwnpu-chat             -         -      -         -      -    2.0 GB\n"} {
		if !strings.Contains(body, want) {
			t.Errorf("%q not in the body:\n%s", want, body)
		}
	}
	if strings.Contains(body, "secret prompt") || strings.Contains(body, "one\n") {
		t.Errorf("body:\n%s", body)
	}
}
