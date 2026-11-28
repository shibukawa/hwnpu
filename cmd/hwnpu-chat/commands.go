package main

import (
	"fmt"
	"strings"
	"text/tabwriter"

	"github.com/shibukawa/hwnpu/internal/console"
)

// command is a command of a conversation in the terminal: a line that starts with / and one of
// its names (/model, /exit), as the other chat programs have them.
type command struct {
	// names are what calls it, without the /: the first is the one the help shows first.
	names []string
	// args is what follows the name, as the help writes it ("[NAME]"; "" for none).
	args  string
	about string
	// unless says why the command is not for the model at hand ("" or nil: it is): the help
	// leaves it out, and calling it says so.
	unless func() string
	// run does it with what follows the name; quit ends the conversation.
	run func(arg string) (quit bool)
}

// commands are the commands of a conversation, in the order the help lists them.
type commands []command

// commandOf is the name and the rest of a line that calls a command: / and lowercase letters
// (or ?) up to the first space. A line that only starts with a / is not one (/usr/bin/env, a
// path): it is a message.
func commandOf(line string) (name, arg string, ok bool) {
	rest, ok := strings.CutPrefix(line, "/")
	if !ok {
		return "", "", false
	}
	name, arg, _ = strings.Cut(rest, " ")
	if i := strings.IndexAny(name, "\t\n"); i >= 0 {
		name, arg = name[:i], name[i:]+" "+arg
	}
	if name == "" {
		return "", "", false
	}
	for _, r := range name {
		if (r < 'a' || r > 'z') && r != '?' {
			return "", "", false
		}
	}
	return name, strings.TrimSpace(arg), true
}

// do runs the command line calls. handled is false for a line that calls none (a message);
// quit is true when the command ends the conversation. A command that is not for the model at
// hand is an error written here, and so is a name one slip of the keys from a command's
// (/modle): it is not sent to the model. Any other name is a message (/tmp is full).
func (cs commands) do(line string) (handled, quit bool) {
	name, arg, ok := commandOf(line)
	if !ok {
		return false, false
	}
	if name == "help" || name == "?" {
		cs.help()
		return true, false
	}
	for _, c := range cs {
		for _, n := range c.names {
			if n != name {
				continue
			}
			if c.unless != nil {
				if why := c.unless(); why != "" {
					printError(fmt.Errorf("/%s: %s", name, why))
					return true, false
				}
			}
			return true, c.run(arg)
		}
	}
	if near := cs.near(name); near != "" {
		printError(fmt.Errorf("no command /%s: /%s? (/help lists them)", name, near))
		return true, false
	}
	return false, false
}

// near is the command's name that name is one slip of the keys from (a letter changed, added,
// dropped or swapped with the next); "" for none. The folders of a Unix system (/tmp, /dev,
// /sys) are further than that from every command.
func (cs commands) near(name string) string {
	for _, c := range append(cs, command{names: []string{"help"}}) {
		for _, n := range c.names {
			if distance(name, n) == 1 {
				return n
			}
		}
	}
	return ""
}

// distance is the edit distance of a and b: the letters to change, add, drop or swap with the
// next (/modle is one from /model).
func distance(a, b string) int {
	d := make([][]int, len(a)+1)
	for i := range d {
		d[i] = make([]int, len(b)+1)
		d[i][0] = i
	}
	for j := range d[0] {
		d[0][j] = j
	}
	for i := 1; i <= len(a); i++ {
		for j := 1; j <= len(b); j++ {
			sub := d[i-1][j-1]
			if a[i-1] != b[j-1] {
				sub++
			}
			d[i][j] = min(sub, d[i-1][j]+1, d[i][j-1]+1)
			if i > 1 && j > 1 && a[i-1] == b[j-2] && a[i-2] == b[j-1] {
				d[i][j] = min(d[i][j], d[i-2][j-2]+1)
			}
		}
	}
	return d[len(a)][len(b)]
}

// help writes the commands that are for the model at hand.
func (cs commands) help() {
	var b strings.Builder
	tw := tabwriter.NewWriter(&b, 0, 0, 2, ' ', 0)
	for _, c := range cs {
		if c.unless != nil && c.unless() != "" {
			continue
		}
		names := "/" + strings.Join(c.names, ", /")
		if c.args != "" {
			names = "/" + c.names[0] + " " + c.args
			if len(c.names) > 1 {
				names += " (/" + strings.Join(c.names[1:], ", /") + ")"
			}
		}
		fmt.Fprintf(tw, "%s\t%s\n", names, c.about)
	}
	fmt.Fprintf(tw, "/help, /?\t%s\n", "this list")
	tw.Flush()
	console.Notef("%s", b.String())
	console.Notef("Ctrl-D quits too, ↑ ↓ go through what was typed, a line ending in \\ goes on in the next\n")
}

// quitCommand is /exit, as every conversation has it.
func quitCommand() command {
	return command{names: []string{"exit", "quit", "bye"}, about: "quit", run: func(string) bool { return true }}
}
