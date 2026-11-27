package main

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/catalog"
	"github.com/shibukawa/hwnpu/internal/daemon"
	"github.com/shibukawa/hwnpu/internal/envfile"
)

// ConfigCmd reads and writes hwnpu's settings (docs/design/hwnpud.md, Settings) by their keys
// (model-source, model-allowlist: envfile.Keys), as git config does. They are kept in the user's
// ~/.config/hwnpu/.env (--file another), which every command and hwnpud read; hwnpud applies a
// change at once: its engine is stopped when no request goes on, and the next request starts one
// with the file as it is.
//
// A single value is set, got and removed; a list (model-source, model-allowlist, model-denylist)
// is changed an item at a time: add, get (the items numbered), remove N. model-source's items are
// the sources in the order they are looked in (HWNPU_MODEL_SOURCE, _1, _2, ...); model-allowlist
// and model-denylist are the +items and the -items of HWNPU_MODEL_ALLOWLIST (the policy:
// docs/model-provenance.md).
type ConfigCmd struct {
	List   ConfigListCmd   `cmd:"" default:"1" help:"The settings in effect here and where each comes from."`
	Keys   ConfigKeysCmd   `cmd:"" help:"The keys there are, and what each sets."`
	Get    ConfigGetCmd    `cmd:"" help:"A setting's value; a list's items, numbered for remove."`
	Set    ConfigSetCmd    `cmd:"" help:"Set a setting of a single value (cache, memory, ...); hwnpud applies it at once."`
	Add    ConfigAddCmd    `cmd:"" help:"Add an item to a list (model-source URL, model-allowlist nsfw, model-denylist license:non-commercial); hwnpud applies it at once."`
	Remove ConfigRemoveCmd `cmd:"" help:"Remove a setting, or a list's item by its number (or the item itself); hwnpud applies it at once."`
}

// fileFlag is the file a command reads and writes.
type fileFlag struct {
	File string `type:"path" placeholder:"PATH" help:"The .env file (default: ~/.config/hwnpu/.env, the user's, which hwnpud reads)."`
}

func (f fileFlag) path() string { return cmp.Or(f.File, envfile.UserFile()) }

// lookupKey is the key of name.
func lookupKey(name string) (envfile.Key, error) {
	if k, ok := envfile.KeyByName(name); ok {
		return k, nil
	}
	if k, ok := envfile.KeyOf(name); ok {
		return k, fmt.Errorf("%s is a variable of the key %s: hwnpu config uses the key", name, k.Name)
	}
	var names []string
	for _, k := range envfile.Keys {
		names = append(names, k.Name)
	}
	return envfile.Key{}, fmt.Errorf("no key %q (hwnpu config keys: %s)", name, strings.Join(names, ", "))
}

// envFiles are the .env files read, in order, that are there.
func envFiles() []string {
	var out []string
	seen := map[string]bool{}
	for _, d := range envfile.Dirs() {
		for _, n := range []string{".env.local", ".env"} {
			p := filepath.Join(d, n)
			if seen[p] {
				continue
			}
			seen[p] = true
			if _, err := os.Stat(p); err == nil {
				out = append(out, p)
			}
		}
	}
	return out
}

func envMap(env []string) map[string]string {
	m := map[string]string{}
	for _, kv := range env {
		if k, v, ok := strings.Cut(kv, "="); ok {
			if _, seen := m[k]; !seen {
				m[k] = v
			}
		}
	}
	return m
}

// shellItems are the key's items the environment this command was started with (not the files)
// sets.
func shellItems(k envfile.Key) ([]string, bool) { return k.Get(envMap(envfile.Original())) }

// source is where this process's items of the key come from: the shell, or the first file of
// envFiles that sets it ("" when nothing does).
func source(k envfile.Key) (where string, items []string) {
	if items, ok := shellItems(k); ok {
		return "this shell", items
	}
	for _, p := range envFiles() {
		if items, set, _ := envfile.GetIn(p, k); set {
			return p, items
		}
	}
	return "", nil
}

// shown are the items as they are shown: a credential as set, a URL without its password.
func shown(k envfile.Key, items []string) []string {
	out := make([]string, len(items))
	for i, it := range items {
		out[i] = daemon.Shown(k.Env, it)
	}
	return out
}

func printItems(w io.Writer, k envfile.Key, items []string, indent string) {
	if !k.List() {
		for _, it := range shown(k, items) {
			fmt.Fprintln(w, indent+it)
		}
		return
	}
	for i, it := range shown(k, items) {
		fmt.Fprintf(w, "%s%d: %s\n", indent, i+1, it)
	}
}

type ConfigListCmd struct{}

func (c *ConfigListCmd) Run() error {
	found := false
	for _, k := range envfile.Keys {
		where, items := source(k)
		if where == "" {
			continue
		}
		found = true
		if where == "this shell" {
			where += "; hwnpud does not see it"
		}
		if k.List() {
			fmt.Printf("%s   (%s):\n", k.Name, where)
			printItems(os.Stdout, k, items, "  ")
		} else {
			fmt.Printf("%s = %s   (%s)\n", k.Name, shown(k, items)[0], where)
		}
	}
	// what the files set that no key has
	seen := map[string]bool{}
	for _, p := range envFiles() {
		vars, _, err := envfile.Read(p)
		for _, v := range vars {
			if _, ok := envfile.KeyOf(v.Name); ok || seen[v.Name] {
				continue
			}
			seen[v.Name] = true
			found = true
			fmt.Printf("%s=%s   (%s; no key)\n", v.Name, daemon.Shown(v.Name, v.Value), p)
		}
		if err != nil {
			fmt.Printf("error: %v\n", err)
		}
	}
	if !found {
		fmt.Printf("nothing set (hwnpu config keys; hwnpu config set|add KEY VALUE writes %s)\n", envfile.UserFile())
	}
	return nil
}

type ConfigKeysCmd struct{}

func (c *ConfigKeysCmd) Run() error {
	for _, k := range envfile.Keys {
		kind := "set/get/remove"
		if k.List() {
			kind = "add/get/remove N"
		}
		fmt.Printf("%-22s %s (%s)\n", k.Name, k.Help, kind)
	}
	return nil
}

type ConfigGetCmd struct {
	fileFlag
	Key string `arg:"" help:"The key (hwnpu config keys)."`
}

func (c *ConfigGetCmd) Run(ctx context.Context) error {
	k, err := lookupKey(c.Key)
	if err != nil {
		return err
	}
	p := c.path()
	items, set, err := envfile.GetIn(p, k)
	if err != nil {
		return err
	}
	if !set {
		where, other := source(k)
		if where == "" {
			return fmt.Errorf("%s is not set", k.Name)
		}
		items = other
		fmt.Fprintf(os.Stderr, "(%s sets it, not %s)\n", where, p)
	}
	printItems(os.Stdout, k, items, "")
	if set {
		winners(os.Stderr, p, k)
	}
	if daemon.Mode(daemon.DefaultMode()) == "off" {
		return nil
	}
	hctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	h, err := daemon.NewClient("").Hello(hctx)
	if err != nil {
		return nil // hwnpud does not run: it reads the file when it starts
	}
	if there, ok := hwnpudItems(h, k); ok && !slices.Equal(there, shown(k, items)) {
		fmt.Fprintln(os.Stderr, "hwnpud has:")
		printItems(os.Stderr, k, there, "  ")
	}
	if pending(h, k) {
		fmt.Fprintln(os.Stderr, "hwnpud's engine still runs with the old value: it starts again with this one when no request goes on")
	}
	return nil
}

// hwnpudItems are hwnpud's items of the key (as Shown), when hwnpud reports the key.
func hwnpudItems(h daemon.Hello, k envfile.Key) ([]string, bool) {
	if _, reported := daemon.Settings([]string{k.Env + "=x"})[k.Env]; !reported {
		return nil, false
	}
	items, _ := k.Get(h.Settings)
	return items, true
}

func pending(h daemon.Hello, k envfile.Key) bool {
	return slices.ContainsFunc(h.Pending, k.Owns)
}

type ConfigSetCmd struct {
	fileFlag
	Key   string `arg:"" help:"The key (hwnpu config keys)."`
	Value string `arg:"" help:"Its value."`
}

func (c *ConfigSetCmd) Run(ctx context.Context) error {
	k, err := lookupKey(c.Key)
	if err != nil {
		return err
	}
	if k.List() {
		return fmt.Errorf("%s is a list: hwnpu config add %s ITEM, hwnpu config remove %s N", k.Name, k.Name, k.Name)
	}
	if strings.Contains(c.Value, "\n") {
		return fmt.Errorf("%s: a value of one line", k.Name)
	}
	p := c.path()
	if err := envfile.SetIn(p, k, []string{c.Value}); err != nil {
		return err
	}
	fmt.Printf("%s = %s (%s)\n", k.Name, daemon.Shown(k.Env, c.Value), p)
	afterWrite(ctx, p, k)
	return nil
}

type ConfigAddCmd struct {
	fileFlag
	At   int    `placeholder:"N" help:"Put it at this number (1: first), not at the end: model-source's are looked in in order."`
	Key  string `arg:"" help:"The key of a list (model-source, model-allowlist, model-denylist)."`
	Item string `arg:"" help:"The item (model-source file:///mnt/llm, model-allowlist nsfw, model-denylist license:non-commercial)."`
}

func (c *ConfigAddCmd) Run(ctx context.Context) error {
	k, err := lookupKey(c.Key)
	if err != nil {
		return err
	}
	if !k.List() {
		return fmt.Errorf("%s is a single value: hwnpu config set %s VALUE", k.Name, k.Name)
	}
	item := strings.TrimSpace(c.Item)
	if err := checkItem(k, item); err != nil {
		return err
	}
	p := c.path()
	items, _, err := envfile.GetIn(p, k)
	if err != nil {
		return err
	}
	if slices.Contains(items, item) {
		fmt.Printf("%s already has %s\n", k.Name, daemon.Shown(k.Env, item))
		return nil
	}
	switch {
	case c.At == 0:
		items = append(items, item)
	case c.At < 1 || c.At > len(items)+1:
		return fmt.Errorf("--at %d: %s has %d items (1 to %d)", c.At, k.Name, len(items), len(items)+1)
	default:
		items = slices.Insert(items, c.At-1, item)
	}
	if err := envfile.SetIn(p, k, items); err != nil {
		return err
	}
	fmt.Printf("%s (%s):\n", k.Name, p)
	printItems(os.Stdout, k, items, "  ")
	afterWrite(ctx, p, k)
	return nil
}

// checkItem says what is wrong with an item of the list k.
func checkItem(k envfile.Key, item string) error {
	if item == "" || strings.Contains(item, "\n") || k.Sep != "" && strings.Contains(item, k.Sep) {
		return fmt.Errorf("an item of %s, of one line, without %q: add them one at a time", k.Name, k.Sep)
	}
	if k.Sign != "" {
		if strings.HasPrefix(item, "+") || strings.HasPrefix(item, "-") {
			return fmt.Errorf("%s's items are written without + or -: hwnpu config add model-allowlist nsfw, hwnpu config add model-denylist license:non-commercial", k.Name)
		}
		_, err := catalog.ParsePolicy([]string{k.Sign + item}, k.Name)
		return err
	}
	if k.Numbered {
		u, err := url.Parse(item)
		if err != nil {
			return fmt.Errorf("%s: %w", k.Name, err)
		}
		switch u.Scheme {
		case "file", "http", "https", "s3":
		default:
			return fmt.Errorf("%s: %q is not a source's URL (file:///PATH, https://USER@host/dav, s3://BUCKET/PREFIX)", k.Name, item)
		}
	}
	return nil
}

type ConfigRemoveCmd struct {
	fileFlag
	Key  string `arg:"" help:"The key (hwnpu config keys)."`
	Item string `arg:"" optional:"" placeholder:"N|ITEM" help:"A list's item: its number in hwnpu config get, or the item itself."`
}

func (c *ConfigRemoveCmd) Run(ctx context.Context) error {
	k, err := lookupKey(c.Key)
	if err != nil {
		return err
	}
	p := c.path()
	items, set, err := envfile.GetIn(p, k)
	if err != nil {
		return err
	}
	if !set {
		return fmt.Errorf("%s is not set in %s", k.Name, p)
	}
	switch {
	case !k.List() && c.Item != "":
		return fmt.Errorf("%s is a single value: hwnpu config remove %s", k.Name, k.Name)
	case !k.List():
		items = nil
	case c.Item == "":
		return fmt.Errorf("%s is a list: hwnpu config remove %s N (hwnpu config get %s numbers them)", k.Name, k.Name, k.Name)
	default:
		i := slices.Index(items, c.Item)
		if i < 0 {
			i = slices.Index(shown(k, items), c.Item) // a URL as shown, without its password
		}
		if n, err := strconv.Atoi(c.Item); err == nil {
			if n < 1 || n > len(items) {
				return fmt.Errorf("%s has %d items: no %d (hwnpu config get %s)", k.Name, len(items), n, k.Name)
			}
			i = n - 1
		} else if i < 0 {
			return fmt.Errorf("%s has no %s (hwnpu config get %s)", k.Name, c.Item, k.Name)
		}
		fmt.Printf("removed %s\n", daemon.Shown(k.Env, items[i]))
		items = slices.Delete(items, i, i+1)
	}
	if err := envfile.SetIn(p, k, items); err != nil {
		return err
	}
	if len(items) == 0 {
		fmt.Printf("%s removed (%s)\n", k.Name, p)
	} else {
		fmt.Printf("%s (%s):\n", k.Name, p)
		printItems(os.Stdout, k, items, "  ")
	}
	afterWrite(ctx, p, k)
	return nil
}

// winners writes what wins over the file p here: the shell, a file read before it.
func winners(w io.Writer, p string, k envfile.Key) {
	if items, ok := shellItems(k); ok {
		fmt.Fprintf(w, "this shell's %s wins over it here (hwnpud does not see the shell): %s\n", k.Name, strings.Join(shown(k, items), ", "))
	}
	for _, f := range envFiles() {
		if f == p {
			break
		}
		if items, ok, _ := envfile.GetIn(f, k); ok {
			fmt.Fprintf(w, "%s, read first, wins over it: %s\n", f, strings.Join(shown(k, items), ", "))
			break
		}
	}
}

// afterWrite tells what wins over the file p here, and has hwnpud apply it.
func afterWrite(ctx context.Context, p string, k envfile.Key) {
	winners(os.Stdout, p, k)
	if daemon.Mode(daemon.DefaultMode()) == "off" {
		return
	}
	if strings.HasPrefix(k.Env, "HWNPU_DAEMON") {
		fmt.Println("hwnpud reads it when it starts: hwnpu service restart")
		return
	}
	rctx, cancel := context.WithTimeout(ctx, 60*time.Second) // stopping the engine takes up to 30 seconds
	defer cancel()
	h, err := daemon.NewClient("").Reload(rctx)
	switch {
	case errors.Is(err, daemon.ErrNotRunning):
		fmt.Println("hwnpud does not run: it reads the file when it starts")
		return
	case err != nil:
		fmt.Printf("hwnpud did not apply it (%v): hwnpu service restart\n", err)
		return
	}
	if !slices.Contains(h.Files, p) {
		fmt.Printf("hwnpud does not read %s (it reads %s)\n", p, strings.Join(h.Files, ", "))
		return
	}
	if there, ok := hwnpudItems(h, k); ok {
		items, _, _ := envfile.GetIn(p, k)
		if !slices.Equal(there, shown(k, items)) {
			fmt.Printf("hwnpud has %s: the environment it was started with, or a file read before, wins\n", cmp.Or(strings.Join(there, ", "), "it unset"))
			return
		}
	}
	if pending(h, k) {
		fmt.Println("hwnpud: requests go on; its engine starts again with it when they end")
		return
	}
	fmt.Println("hwnpud applied it: its next engine starts with it")
}
