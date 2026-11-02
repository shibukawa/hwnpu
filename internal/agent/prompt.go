package agent

import (
	"os"
	"strings"
)

// DefaultSystem opens the system prompt. It is short on purpose: the model has about 16K
// tokens of context, and what the task needs goes in the user message, where it does not
// change the beginning of the prompt the engine reuses between requests. It names no
// directory: a small model copies a long absolute path wrong, and relative paths need none.
const DefaultSystem = "You are a coding agent working in the current directory. You help with programming tasks by reading files, running commands and editing code with the tools below."

// SearchSystem opens the system prompt when search is the only tool: an assistant that looks
// facts up. A 1.7B model recalls facts badly (the height of Mount Fuji, the author of a
// novel) but copies them from a search result well.
const SearchSystem = "You are a helpful assistant with a search tool. You do not rely on your memory for facts: look them up with search, and answer only from what the results say."

// MaxInstructions is the most characters of project instructions (AGENTS.md) the prompt takes.
const MaxInstructions = 4000

// MaxFiles is the most entries of the working directory the prompt lists.
const MaxFiles = 40

// Prompt builds the system prompt: the opening (Config.System or DefaultSystem), the entries
// of the working directory, the rules of the tools offered, and the project instructions. The
// tools' own declarations are not in it; the chat template adds them.
func Prompt(cfg Config, tools []*Tool) string {
	var sb strings.Builder
	coding := false // a tool that works on the directory is offered
	for _, t := range tools {
		coding = coding || t.Name != "search" && t.Name != "fetch"
	}
	switch {
	case cfg.System != "":
		sb.WriteString(strings.TrimSpace(cfg.System))
	case coding:
		sb.WriteString(DefaultSystem)
	default:
		sb.WriteString(SearchSystem)
	}
	if files := listing(cfg.Dir); coding && files != "" {
		sb.WriteString("\n\nFiles in the current directory: " + files)
	}
	sb.WriteString("\n\nRules:\n- Work step by step: call a tool, look at its result, then decide the next step.\n")
	if coding {
		sb.WriteString("- Paths are relative to the current directory (main.go, src/app.go); do not use absolute paths.\n")
	}
	for _, t := range tools {
		if t.Rule != "" {
			sb.WriteString("- " + t.Rule + "\n")
		}
	}
	if coding {
		sb.WriteString("- When the task is done, stop calling tools and answer with a short summary of what you did.")
	} else {
		sb.WriteString("- When the results answer the question, stop calling tools and answer, in the language of the question.")
	}
	if s := strings.TrimSpace(cfg.Instructions); s != "" {
		if r := []rune(s); len(r) > MaxInstructions {
			s = string(r[:MaxInstructions]) + "\n[...]"
		}
		sb.WriteString("\n\nProject instructions:\n" + s)
	}
	return sb.String()
}

// listing names the entries of the directory, except hidden ones, a directory with a slash.
func listing(dir string) string {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return ""
	}
	var names []string
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".") {
			continue
		}
		if len(names) == MaxFiles {
			names = append(names, "...")
			break
		}
		n := e.Name()
		if e.IsDir() {
			n += "/"
		}
		names = append(names, n)
	}
	return strings.Join(names, ", ")
}
