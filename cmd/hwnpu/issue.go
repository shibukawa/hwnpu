package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/npustatus"
	"github.com/shibukawa/hwnpu/modelstore"
)

// Repo is where hwnpu issue opens a new issue.
const Repo = "shibukawa/hwnpu"

// maxURL is the longest URL of a new issue hwnpu issue opens: GitHub's pages take about 8 KB.
const maxURL = 7500

// IssueCmd is hwnpu issue.
type IssueCmd struct {
	Title string `help:"The issue's title (else made from the command and its error)."`
	Which int    `name:"run" help:"Which run to write about: its number in --list (1: the latest; 0: the latest that failed, else the latest)." default:"0"`
	List  bool   `help:"List the runs kept (hwnpu-chat, hwnpu-image, hwnpu-audio) and write nothing."`
	Lines int    `help:"The last lines of the run's log to put in." default:"80"`
	Print bool   `help:"Write the draft to standard output, and open nothing."`
	Yes   bool   `short:"y" help:"Open the browser without asking."`
	Wake  bool   `help:"Open the NPU and ask it too (as hwnpu status --wake): on Linux this wakes a sleeping NPU for a few seconds."`
}

func (c IssueCmd) Run(ctx context.Context) error {
	s, err := modelstore.Open(modelstore.Options{Log: os.Stderr})
	if err != nil {
		return err
	}
	runs := console.Runs()
	if c.List {
		if len(runs) == 0 {
			fmt.Printf("no runs kept in %s\n", console.RunsDir())
		}
		for i, r := range runs {
			end := "ok"
			switch {
			case r.Error != "":
				end = "error: " + r.Error
			case r.End.IsZero():
				end = "did not end (still running, or killed)"
			}
			fmt.Printf("%2d  %s  %s\n    %s\n", i+1, r.Start.Format(time.DateTime), strings.Join(r.Args, " "), truncate(end, 100))
		}
		return nil
	}
	var run *console.Run
	switch {
	case c.Which > 0 && c.Which <= len(runs):
		run = &runs[c.Which-1]
	case c.Which > 0:
		return fmt.Errorf("no run %d: hwnpu issue --list shows %d", c.Which, len(runs))
	default:
		for i := range runs {
			if runs[i].Failed() && time.Since(runs[i].Start) < 7*24*time.Hour {
				run = &runs[i]
				break
			}
		}
		if run == nil && len(runs) > 0 {
			run = &runs[0]
		}
	}
	st := npustatus.Collect(ctx, npustatus.View{Store: s}, true, c.Wake, new(npustatus.Samplers))
	r := newRedactor()
	title, body := draft(st, run, c.Title, c.Lines, r)
	u := issueURL(title, body)
	for c.Lines > 0 && len(u) > maxURL && run != nil { // the log is cut from its start
		c.Lines = c.Lines * 2 / 3
		title, body = draft(st, run, c.Title, c.Lines, r)
		u = issueURL(title, body)
	}
	file := filepath.Join(console.RunsDir(), "issue-"+time.Now().Format("20060102-150405")+".md")
	if err := os.MkdirAll(filepath.Dir(file), 0o755); err == nil {
		os.WriteFile(file, []byte("# "+title+"\n\n"+body), 0o644)
	}
	if c.Print {
		fmt.Printf("# %s\n\n%s", title, body)
		return nil
	}
	fmt.Printf("# %s\n\n%s\n", title, body)
	fmt.Printf("(the draft is in %s too)\n", file)
	if len(u) > maxURL {
		return fmt.Errorf("the draft is too long for a link (%d bytes): open https://github.com/%s/issues/new and paste %s", len(u), Repo, file)
	}
	fmt.Println("Paths, the user's and the machine's names, the sources' addresses and tokens are hidden. Read it before you send it: the browser opens the issue's page with it, and nothing is sent until you press the button there.")
	if !c.Yes {
		if !console.Terminal(os.Stdin) {
			return errors.New("-y opens the browser (standard input is not a terminal to ask on)")
		}
		fmt.Print("open it in the browser? [Y/n] ")
		line, _ := bufio.NewReader(os.Stdin).ReadString('\n')
		if a := strings.ToLower(strings.TrimSpace(line)); a != "" && a != "y" && a != "yes" {
			return nil
		}
	}
	if err := openBrowser(u); err != nil {
		return fmt.Errorf("opening the browser: %w (open https://github.com/%s/issues/new and paste %s)", err, Repo, file)
	}
	return nil
}

func truncate(s string, n int) string {
	if r := []rune(s); len(r) > n {
		return string(r[:n]) + "…"
	}
	return s
}

// draft is the issue's title and body about st and run (nil for none), with the last lines of
// its log, as r hides them.
func draft(st npustatus.Status, run *console.Run, title string, lines int, r redactor) (string, string) {
	var b strings.Builder
	b.WriteString("<!-- Written by hwnpu issue: paths, the user's and the machine's names, the sources' addresses and tokens are hidden. Check it before you submit. -->\n\n")
	b.WriteString("## What happened\n\n<!-- What you did, what you expected, and what happened instead. -->\n\n\n")
	if run != nil {
		b.WriteString("## Command\n\n```\n" + r.args(run.Args) + "\n```\n\n")
		took := ""
		if !run.End.IsZero() {
			took = ", " + console.Duration(run.End.Sub(run.Start))
		}
		switch {
		case run.Error != "":
			fmt.Fprintf(&b, "Ended with an error%s:\n\n```\n%s\n```\n\n", took, r.hide(run.Error))
		case run.End.IsZero():
			b.WriteString("It did not end (it was killed, or it crashed).\n\n")
		default:
			fmt.Fprintf(&b, "It ended without an error%s.\n\n", took)
		}
		if n := len(run.Lines); n > 0 && lines > 0 {
			from := max(0, n-lines)
			fmt.Fprintf(&b, "<details><summary>Log (%d of %d lines)</summary>\n\n```\n", n-from, n)
			for _, l := range run.Lines[from:] {
				b.WriteString(r.hide(l) + "\n")
			}
			b.WriteString("```\n\n</details>\n\n")
		}
		if title == "" {
			title = filepath.Base(run.Args[0])
			if len(run.Args) > 1 && !strings.HasPrefix(run.Args[1], "-") {
				title += " " + run.Args[1]
			}
			if run.Error != "" {
				title += ": " + truncate(r.hide(run.Error), 80)
			}
		}
	}
	b.WriteString("## Machine\n\n```\n")
	var m strings.Builder
	st.Write(&m, false) // without the processes' command lines: they may hold prompts
	b.WriteString(r.hide(m.String()))
	b.WriteString("```\n")
	if title == "" {
		title = "hwnpu: "
	}
	return r.hide(title), b.String()
}

// issueURL is the page of a new issue of Repo with title and body filled in.
func issueURL(title, body string) string {
	q := url.Values{"title": {title}, "body": {body}}
	return "https://github.com/" + Repo + "/issues/new?" + q.Encode()
}

// openBrowser opens u in the browser: open on macOS, xdg-open on Linux and the BSDs, the URL
// handler on Windows (start would read the & of the query as cmd's).
func openBrowser(u string) error {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("open", u)
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", u)
	default:
		cmd = exec.Command("xdg-open", u)
	}
	cmd.Stdout, cmd.Stderr = nil, nil
	if err := cmd.Start(); err != nil {
		return err
	}
	go cmd.Wait()
	time.Sleep(200 * time.Millisecond) // xdg-open hands the URL on before the process ends
	return nil
}
