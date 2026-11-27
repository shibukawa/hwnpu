package dev

import (
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

func TestGet(t *testing.T) {
	t.Setenv(Env, " moe.time, moe.prefetch=0 ,iron.lookup.cost=2,iron.lookup.cost=3,nope=1")
	if v, ok := Get("iron.lookup.cost"); !ok || v != "3" {
		t.Errorf("iron.lookup.cost: %q %v, want the last", v, ok)
	}
	if !On("moe.time") || On("moe.prefetch") || !IsOff("moe.prefetch") || IsOff("moe.time") {
		t.Error("On and IsOff")
	}
	if _, ok := Get("moe.row"); ok {
		t.Error("moe.row is not set")
	}
	if Int("moe.prefetch", 4) != 0 || Int("moe.row", 4) != 4 {
		t.Error("Int")
	}
	if _, ok := Get("nope"); ok {
		t.Error("a key not in Keys")
	}
}

// TestKeysUsed checks that every key the code reads is in Keys, and every key of Keys is read.
func TestKeysUsed(t *testing.T) {
	root := filepath.Join("..", "..")
	reads := regexp.MustCompile(`devenv\.(?:On|Value|IsOff|Get|Int)\("([a-z0-9.\-]+)"`)
	moe := regexp.MustCompile(`DebugOpt\("([a-z0-9\-]+)"\)`)
	mxfp4 := regexp.MustCompile(`(?:mxfp4\.|[^A-Za-z.])Opt\("([a-z0-9\-]+)"\)`)
	used := map[string]bool{}
	filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() && (d.Name() == "testdata" || strings.HasPrefix(d.Name(), ".")) && path != root {
			if d != nil && d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}
		b, _ := os.ReadFile(path)
		for _, m := range reads.FindAllSubmatch(b, -1) {
			if !strings.HasSuffix(string(m[1]), ".") { // a group read by its item's name (DebugOpt, mxfp4.Opt)
				used[string(m[1])] = true
			}
		}
		for _, m := range moe.FindAllSubmatch(b, -1) {
			used["moe."+string(m[1])] = true
		}
		if strings.Contains(path, "mxfp4") {
			for _, m := range mxfp4.FindAllSubmatch(b, -1) {
				used["mxfp4."+string(m[1])] = true
			}
		}
		return nil
	})
	for k := range used {
		if _, ok := Keys[k]; !ok {
			t.Errorf("%s is read but not in Keys", k)
		}
	}
	for k := range Keys {
		if !used[k] {
			t.Errorf("%s is in Keys but nothing reads it", k)
		}
	}
}
