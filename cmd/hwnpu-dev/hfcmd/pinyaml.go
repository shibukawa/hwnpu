package hfcmd

import (
	"fmt"
	"regexp"
	"strings"
)

// pinYAML writes into a definition file (models.yaml) that model's repository repo pins commit
// for target: models.MODEL.hub.REPO.TARGET. It changes that line alone, or adds the lines it
// needs (hub: after from:, the repository at the end of hub, the target at the end of the
// repository); comments, order and the other lines stay as they are. A model the file does not
// have is added at the end of models (an override of the model only, as a user's file has).
func pinYAML(src []byte, model, repo, target, commit string) ([]byte, error) {
	out, err := pinLines(strings.Split(string(src), "\n"), model, repo, target, commit)
	if err != nil {
		return nil, err
	}
	s := strings.Join(out, "\n")
	if !strings.HasSuffix(s, "\n") {
		s += "\n"
	}
	return []byte(s), nil
}

func pinLines(lines []string, model, repo, target, commit string) ([]string, error) {
	indent := func(l string) int { return len(l) - len(strings.TrimLeft(l, " ")) }
	content := func(l string) bool {
		t := strings.TrimSpace(l)
		return t != "" && !strings.HasPrefix(t, "#")
	}
	// key matches a line of key at an indent
	key := func(l string, at int, k string) (rest string, ok bool) {
		if indent(l) != at {
			return "", false
		}
		t := strings.TrimSpace(l)
		for _, q := range []string{k, `"` + k + `"`, `'` + k + `'`} {
			if r, ok := strings.CutPrefix(t, q+":"); ok && (r == "" || r[0] == ' ') {
				return strings.TrimSpace(r), true
			}
		}
		return "", false
	}
	// block is the end of the block that starts at line i (the lines after it more indented)
	block := func(i int) int {
		at := indent(lines[i])
		j := i + 1
		last := i
		for ; j < len(lines); j++ {
			if !content(lines[j]) {
				continue
			}
			if indent(lines[j]) <= at {
				break
			}
			last = j
		}
		return last + 1
	}
	// child is the indent of the block's lines (step more than the block's own when it has none)
	child := func(i, end, step int) int {
		for j := i + 1; j < end; j++ {
			if content(lines[j]) {
				return indent(lines[j])
			}
		}
		return indent(lines[i]) + step
	}
	insert := func(at int, add ...string) {
		lines = append(lines[:at], append(add, lines[at:]...)...)
	}
	pad := func(n int) string { return strings.Repeat(" ", n) }

	top := -1
	for i, l := range lines {
		if _, ok := key(l, 0, "models"); ok {
			top = i
			break
		}
	}
	if top < 0 {
		lines = append(trimEnd(lines), "", "models:")
		top = len(lines) - 1
	}
	if r, _ := key(lines[top], 0, "models"); r != "" && !strings.HasPrefix(r, "#") {
		return nil, fmt.Errorf("models: is written in one line; write it as a block to pin %s", model)
	}
	topEnd := block(top)
	step := child(top, topEnd, 2)
	m := -1
	for i := top + 1; i < topEnd; i++ {
		if _, ok := key(lines[i], step, model); ok {
			m = i
			break
		}
	}
	if m < 0 {
		insert(topEnd, pad(step)+model+":", pad(2*step)+"hub:", pad(3*step)+repo+":", pad(4*step)+target+": "+commit)
		return lines, nil
	}
	mEnd := block(m)
	fi := child(m, mEnd, step)
	unit := fi - step
	h := -1
	from := -1
	for i := m + 1; i < mEnd; i++ {
		if r, ok := key(lines[i], fi, "hub"); ok {
			if r != "" && !strings.HasPrefix(r, "#") {
				return nil, fmt.Errorf("%s: hub is written in one line; write it as a block (hub:, then a line for each repository)", model)
			}
			h = i
		}
		if _, ok := key(lines[i], fi, "from"); ok {
			from = i
		}
	}
	if h < 0 {
		at := m + 1
		if from >= 0 {
			at = from + 1
		}
		insert(at, pad(fi)+"hub:", pad(fi+unit)+repo+":", pad(fi+2*unit)+target+": "+commit)
		return lines, nil
	}
	hEnd := block(h)
	ri := child(h, hEnd, unit)
	r := -1
	for i := h + 1; i < hEnd; i++ {
		if _, ok := key(lines[i], ri, repo); ok {
			r = i
			break
		}
	}
	if r < 0 {
		insert(hEnd, pad(ri)+repo+":", pad(ri+unit)+target+": "+commit)
		return lines, nil
	}
	if rest, _ := key(lines[r], ri, repo); strings.HasPrefix(rest, "{") {
		// a repository in one line: {amd-xdna2: COMMIT, apple: COMMIT}
		l, err := pinFlow(lines[r], target, commit)
		if err != nil {
			return nil, fmt.Errorf("%s: hub: %s: %w", model, repo, err)
		}
		lines[r] = l
		return lines, nil
	}
	rEnd := block(r)
	ti := child(r, rEnd, unit)
	for i := r + 1; i < rEnd; i++ {
		if _, ok := key(lines[i], ti, target); ok {
			comment := ""
			if c := strings.Index(lines[i], " #"); c >= 0 {
				for c > 0 && lines[i][c-1] == ' ' {
					c--
				}
				comment = lines[i][c:]
			}
			lines[i] = pad(ti) + target + ": " + commit + comment
			return lines, nil
		}
	}
	insert(rEnd, pad(ti)+target+": "+commit)
	return lines, nil
}

var flowPin = regexp.MustCompile(`([{,]\s*)([A-Za-z0-9-]+)(\s*:\s*)([0-9a-z]+)`)

// pinFlow sets target to commit in the line of a repository written as a flow map.
func pinFlow(line, target, commit string) (string, error) {
	open, close := strings.Index(line, "{"), strings.LastIndex(line, "}")
	if open < 0 || close < open {
		return "", fmt.Errorf("a map in one line that does not close")
	}
	inner := line[open : close+1]
	done := false
	inner = flowPin.ReplaceAllStringFunc(inner, func(s string) string {
		p := flowPin.FindStringSubmatch(s)
		if p[2] != target {
			return s
		}
		done = true
		return p[1] + p[2] + p[3] + commit
	})
	if !done {
		body := strings.TrimSpace(inner[1 : len(inner)-1])
		if body == "" {
			inner = "{" + target + ": " + commit + "}"
		} else {
			inner = "{" + body + ", " + target + ": " + commit + "}"
		}
	}
	return line[:open] + inner + line[close+1:], nil
}

func trimEnd(lines []string) []string {
	for len(lines) > 0 && strings.TrimSpace(lines[len(lines)-1]) == "" {
		lines = lines[:len(lines)-1]
	}
	return lines
}
