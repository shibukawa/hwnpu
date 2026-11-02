package agent

import (
	"context"
	"fmt"
	"html"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"
	"unicode/utf8"
)

// Fetch reads a page for the fetch tool and returns its text: the HTML's tags, scripts and
// styles stripped, whitespace folded, at most maxChars characters (the rest is counted).
func Fetch(ctx context.Context, client *http.Client, u string, maxChars int) (string, error) {
	if client == nil {
		client = &http.Client{Timeout: 20 * time.Second}
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", userAgent)
	req.Header.Set("Accept", "text/html, text/plain;q=0.9, */*;q=0.1")
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("%s: %s", req.URL.Host, resp.Status)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return "", err
	}
	text := string(body)
	if ct := resp.Header.Get("Content-Type"); strings.Contains(ct, "html") || strings.Contains(text[:min(len(text), 512)], "<") {
		text = Text(text)
	}
	if !utf8.ValidString(text) {
		return "", fmt.Errorf("%s: not a text page", req.URL.Host)
	}
	text = strings.TrimSpace(text)
	if r := []rune(text); len(r) > maxChars {
		text = strings.TrimSpace(string(r[:maxChars])) + fmt.Sprintf("\n[... %d more characters omitted]", len(r)-maxChars)
	}
	if text == "" {
		return "", fmt.Errorf("%s: an empty page", req.URL.Host)
	}
	return text, nil
}

var (
	mainRe  = regexp.MustCompile(`(?is)<main\b.*?</main\s*>|<article\b.*?</article\s*>`)
	dropRe  = regexp.MustCompile(`(?is)<script\b.*?</script\s*>|<style\b.*?</style\s*>|<noscript\b.*?</noscript\s*>|<svg\b.*?</svg\s*>|<head\b.*?</head\s*>|<nav\b.*?</nav\s*>|<footer\b.*?</footer\s*>|<aside\b.*?</aside\s*>|<!--.*?-->`)
	blockRe = regexp.MustCompile(`(?i)</?(p|div|br|li|ul|ol|h[1-6]|tr|td|th|table|section|article|header|footer|nav|blockquote|pre|dl|dt|dd)\b[^>]*>`)
)

// Text turns HTML into text: the main element (or the article) when the page has one, so
// that a site's menus do not take the characters the model gets (Wikipedia's took 4K tokens of
// the 6000 characters); scripts, styles, navigation and footers go, block tags become line
// breaks, other tags go, entities are decoded, and blank lines go (the model needs no
// paragraphs).
func Text(h string) string {
	if m := mainRe.FindString(h); m != "" {
		h = m
	}
	h = dropRe.ReplaceAllString(h, "")
	h = blockRe.ReplaceAllString(h, "\n")
	h = html.UnescapeString(tagRe.ReplaceAllString(h, ""))
	var lines []string
	for _, l := range strings.Split(h, "\n") {
		if l = strings.Join(strings.Fields(l), " "); l != "" {
			lines = append(lines, l)
		}
	}
	return strings.Join(lines, "\n")
}
