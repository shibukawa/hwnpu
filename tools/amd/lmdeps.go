package amdtools

import (
	"crypto/sha256"
	"encoding/hex"
	"io/fs"
	"path"
	"regexp"
	"slices"
	"strings"
)

// An export of a chat model (iron_export_lm.py MODEL ...) is keyed by the files of tools/amd it
// reads, not all of them (LMHash): a change to Gemma 4's per-layer table, or to an image model's
// exporter, makes the exports that read it again and leaves the others found. The files are
// lmCommon, the roots of the model's architecture in lmArch (the modules iron_export_lm.py
// imports for it: its MODELS entry and the installs of its branch), and those of the flags in
// lmFlags, each with the modules it imports, followed in tools/amd (lmClosure), and every kernel.

// lmCommon is what every export of iron_export_lm.py reads.
var lmCommon = []string{"iron_export_lm.py", "iron_export.py", "iron_stream.py", "iron_window.py"}

// lmArch is, by iron_export_lm.py's model (its first argument), the modules of tools/amd it
// imports for it: the module of its MODELS entry and those its branch of main installs.
// TestLMArchCoversModels checks every model of MODELS is here; a module the branch of a model
// starts to import is added by hand.
var lmArch = map[string][]string{
	"gemma4":       {"iron_gemma4.py", "iron_hostple.py"},
	"gemma4full":   {"iron_hostple.py"},
	"gemma4mm":     {"iron_gemma4.py", "iron_hostple.py"},
	"gemma4mmfull": {"iron_hostple.py"},
	"lfm3":         {"iron_lfm3.py"},
	"lfm3vl":       {"iron_lfm3vl.py", "iron_lfm3.py"},
	"lfm3audio":    {"iron_lfm3audio.py", "iron_lfm3.py"},
	"lfm3omni":     {"iron_lfm3omni.py", "iron_lfm3.py"},
	"plamo3":       {"iron_plamo3.py"},
	"ouro":         {"iron_ouro.py"},
	"qwen35":       {"iron_qwen35.py"},
	"qwen35moe":    {"iron_qwen35.py", "iron_moe.py"},
	"gptoss":       {"iron_gptoss.py"},
}

// lmFlags is the modules iron_export_lm.py imports for a flag.
var lmFlags = map[string][]string{
	"--verify":   {"iron_verify.py"},
	"--q4-codes": {"iron_q4.py"},
}

var importRe = regexp.MustCompile(`(?m)^\s*(?:from\s+(\w+)\s+import|import\s+(\w+))`)

// LMFiles is the files of tools/amd an export of iron_export_lm.py with args reads, sorted; nil
// for a model lmArch does not know (then all of them are its key: Hash).
func LMFiles(args []string) []string {
	if len(args) == 0 {
		return nil
	}
	roots, ok := lmArch[args[0]]
	if !ok {
		return nil
	}
	roots = append(slices.Clone(lmCommon), roots...)
	for _, a := range args[1:] {
		flag, _, _ := strings.Cut(a, "=")
		roots = append(roots, lmFlags[flag]...)
	}
	return lmClosure(files, roots)
}

// lmClosure is roots and the modules of fsys they import (at any depth, whatever the branch),
// and every kernel, sorted. iron_export_lm.py's own imports are not followed (the modules of
// all its models: the roots name the ones a model reads), nor are those of a module back to it.
// The kernels are all taken, as a module may name one it builds from its parts (q8_mv.cc).
func lmClosure(fsys fs.FS, roots []string) []string {
	seen := map[string]bool{}
	stack := slices.Clone(roots)
	for len(stack) > 0 {
		p := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if seen[p] {
			continue
		}
		b, err := fs.ReadFile(fsys, p)
		if err != nil {
			continue
		}
		seen[p] = true
		if p == "iron_export_lm.py" || !strings.HasSuffix(p, ".py") {
			continue
		}
		for _, m := range importRe.FindAllStringSubmatch(string(b), -1) {
			stack = append(stack, m[1]+m[2]+".py")
		}
	}
	fs.WalkDir(fsys, "kernels", func(p string, d fs.DirEntry, err error) error {
		if err == nil && !d.IsDir() {
			seen[p] = true
		}
		return nil
	})
	out := make([]string, 0, len(seen))
	for p := range seen {
		out = append(out, p)
	}
	slices.Sort(out)
	return out
}

// LMHash is the SHA-256 (hex) of the files an export of iron_export_lm.py with args reads
// (LMFiles: their paths and contents), or Hash for a model lmArch does not know.
func LMHash(args []string) (string, error) {
	names := LMFiles(args)
	if names == nil {
		return Hash()
	}
	h := sha256.New()
	for _, p := range names {
		b, err := files.ReadFile(path.Clean(p))
		if err != nil {
			return "", err
		}
		h.Write([]byte(p + "\x00"))
		h.Write(b)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
