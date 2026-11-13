package chat

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/llm"
	shared "github.com/shibukawa/hwnpu/internal/store"
	amdtools "github.com/shibukawa/hwnpu/tools/amd"
)

// IRONEnv names IRON's environment, in which PrepareAMD runs the exporter: a virtual
// environment's directory, its python, or a script that sets it up (sourced by bash, as
// INSTALL_AMD.md's). Without it, python3 on PATH is tried.
const IRONEnv = "HWNPU_IRON_ENV"

// AMDRoot is where the exports for AMD's NPU hwnpu makes go: entries/amd-xdna2 under cache.Root
// (NAME.amd-npu2, NAME.amd-npu2-q4). They are looked for there and then in ~/models, where
// exports were kept before and the exporters run by hand write (cache.ExportRoots).
func AMDRoot() string { return cache.Entries(cache.AMDTarget) }

// amdSuffix is the suffix of the export's directory iron_export_lm.py's options make: -q4 for
// --q4, -q6 when the MLP is in 6 bits (--mlp q6), none in bf16.
func amdSuffix(args []string) string {
	if !slices.Contains(args, "--q4") {
		return ""
	}
	if i := slices.Index(args, "--mlp"); i >= 0 && i+1 < len(args) {
		return "-" + args[i+1]
	}
	return "-q4"
}

// AMDExportDir is the directory of model m's export for AMD's NPU that models.yaml's
// amd_export makes, "" for a model without one.
func AMDExportDir(m Model) string {
	if len(m.AMDExport) == 0 {
		return ""
	}
	return filepath.Join(AMDRoot(), m.Name+".amd-npu2"+amdSuffix(m.AMDExport[1:]))
}

// isExport reports whether dir is a whole export of iron_export_lm.py: it has its manifest
// (PrepareAMD writes the export beside it and renames it into place when it is done).
func isExport(dir string) bool {
	_, err := os.Stat(filepath.Join(dir, "manifest.json"))
	return err == nil
}

// FindAMDExport is the export of model m here: the one its amd_export makes, else
// NAME.amd-npu2-q4, else NAME.amd-npu2 (exports made by hand); "" when there is none.
func FindAMDExport(m Model) string {
	var names []string
	if d := AMDExportDir(m); d != "" {
		names = append(names, filepath.Base(d))
	}
	for _, suffix := range []string{".amd-npu2-q4", ".amd-npu2"} {
		names = append(names, m.Name+suffix)
	}
	for _, n := range names {
		if d := cache.FindExport(n, isExport); d != "" {
			return d
		}
	}
	return ""
}

// OnlyAMD reports whether the NPU models run on is AMD's: no backend named (backend or
// HWNPU_BACKEND), no Apple or Intel NPU, and an AMD one.
func OnlyAMD(ctx context.Context, backend string) bool {
	if backend != "" || strings.TrimSpace(os.Getenv(llm.BackendEnv)) != "" {
		return false
	}
	if _, _, err := llm.FindDevice(ctx, ""); err == nil {
		return false
	}
	infos, _ := hwnpu.Enumerate(ctx)
	for _, info := range infos {
		if info.Vendor == hwnpu.VendorAMD {
			return true
		}
	}
	return false
}

// AMDExport is the export of the model use names (a name, a use or a family picked for the
// machine's memory, logged to log, or a repository OWNER/NAME) that hwnpu-chat runs when the
// only NPU is AMD's, which runs no programs of hwnpu-dev convert. It is "" when that is not so
// (OnlyAMD). As the other NPUs' models are prepared on their first run, a model of the
// definitions with amd_export and no export here is made ready by PrepareAMD: its checkpoint
// downloaded into dir and exported with IRON.
func AMDExport(ctx context.Context, use, backend, dir string, log io.Writer) (string, error) {
	if !OnlyAMD(ctx, backend) {
		return "", nil
	}
	name := strings.TrimSpace(use)
	if strings.Contains(name, ",") {
		return "", fmt.Errorf("--model %s: on AMD's NPU, one chat model runs at a time", use)
	}
	m, ok, err := lookupFor(name, log)
	if err != nil {
		return "", err
	}
	if !ok {
		if m, ok = repoModel(name); !ok {
			return "", errNoModel(name)
		}
	}
	if d := FindAMDExport(m); d != "" {
		return d, nil
	}
	if len(m.AMDExport) == 0 {
		msg := fmt.Sprintf("%s has no form for AMD's NPU: hwnpu has no recipe of tools/amd/iron_export_lm.py for it (amd_export in models.yaml), and %s has no %s.amd-npu2-q4 or %s.amd-npu2; --model takes an export's directory too",
			m.Name, strings.Join(cache.ExportRoots(), " or "), m.Name, m.Name)
		if names := amdExports(); len(names) > 0 {
			msg += ". The exports here: " + strings.Join(names, ", ")
		}
		return "", errors.New(msg)
	}
	return PrepareAMD(ctx, m, dir, log)
}

// amdExports are the names of the exports in cache.ExportRoots.
func amdExports() []string {
	var names []string
	for _, r := range cache.ExportRoots() {
		found, _ := filepath.Glob(filepath.Join(r, "*.amd-npu2*"))
		for _, d := range found {
			if isExport(d) && !slices.Contains(names, filepath.Base(d)) {
				names = append(names, filepath.Base(d))
			}
		}
	}
	return names
}

// checkpointFiles are the files of a checkpoint iron_export_lm.py reads and copies into the
// export, besides the weights (FetchWeights); config.json is the only one every repository has.
var checkpointFiles = []string{"tokenizer.json", "tokenizer.jsonl", "tokenizer_config.json", "special_tokens_map.json",
	"chat_template.jinja", "generation_config.json", "processor_config.json", "preprocessor_config.json"}

// PrepareAMD makes model m's export for AMD's NPU (AMDExportDir) as models.yaml's amd_export
// says: from the shared layers when one has it (SharedExport), else it checks that IRON's
// environment is there (IRONEnv), downloads the checkpoint into
// dir/NAME (only the files the exporter reads), and runs tools/amd/iron_export_lm.py (the copy
// built into hwnpu) in that environment into DIR.part, renamed to the export when it is done.
// Its output goes to DIR.log. A cut-off export is made again.
func PrepareAMD(ctx context.Context, m Model, dir string, log io.Writer) (string, error) {
	logf := func(format string, args ...any) {
		if log != nil {
			fmt.Fprintf(log, format+"\n", args...)
		}
	}
	out := AMDExportDir(m)
	// the shared layers first: a machine without IRON runs an export made elsewhere (no layers:
	// the hub is not asked for the commit)
	var recipe map[string]string
	if len(shared.Layers()) > 0 {
		recipe = ExportRecipe(ctx, m, "iron_export_lm.py", m.AMDExport)
	}
	err := SharedExport(ctx, "iron-lm", recipe, out, log, func() error {
		python, err := ironPython(ctx)
		if err != nil {
			return fmt.Errorf("%s has no export for AMD's NPU here (%s) or in the shared layers, and making one needs IRON (amd/IRON with mlir-aie and Peano, Python 3.12): %w. Install it as INSTALL_AMD.md says and set %s to its virtual environment or to a script that sets it up, or copy an export made elsewhere to %s",
				m.Name, out, err, IRONEnv, out)
		}
		return exportAMD(ctx, python, m, dir, out, logf, log)
	})
	if err != nil {
		return "", err
	}
	return out, nil
}

// exportAMD downloads model m's checkpoint into dir and exports it with IRON into out.
func exportAMD(ctx context.Context, python []string, m Model, dir, out string, logf func(string, ...any), log io.Writer) error {
	logf("%s: no export for AMD's NPU here; downloading the checkpoint and exporting it with IRON (%s)", m.Name, strings.Join(m.AMDExport, " "))
	ck := filepath.Join(dir, m.DirName())
	if err := os.MkdirAll(ck, 0o755); err != nil {
		return err
	}
	if err := fetch(ctx, m.Source(), "config.json", filepath.Join(ck, "config.json"), log); err != nil {
		return err
	}
	if err := FetchWeights(ctx, m.Source(), ck, log); err != nil {
		return err
	}
	for _, f := range checkpointFiles {
		if err := fetch(ctx, m.Source(), f, filepath.Join(ck, f), io.Discard); err != nil && !NotFound(err) {
			return err
		}
	}
	args := append([]string{m.AMDExport[0], ck, ExportOut}, m.AMDExport[1:]...)
	return RunIRON(ctx, python, m.Name, "iron_export_lm.py", args, out, log)
}

// ExportOut stands in RunIRON's args for the directory the exporter writes.
const ExportOut = "{out}"

// RunIRON runs one of hwnpu's IRON exporters (tools/amd, built into hwnpu) in IRON's
// environment (python, of IRONPython): script with args, ExportOut the export's directory,
// made beside out (out.part, its log out.log) and renamed to out when whole, out.done written
// last. The exporter's latest line follows a bar on log.
func RunIRON(ctx context.Context, python []string, name, script string, args []string, out string, log io.Writer) error {
	tools, err := amdtools.Extract(cache.Sub("tools"))
	if err != nil {
		return err
	}
	unlock, err := cache.Lock(ctx, out, "the export "+filepath.Base(out), log)
	if err != nil {
		return err
	}
	defer unlock()
	if _, err := os.Stat(out + ".done"); err == nil {
		return nil // another process made it meanwhile
	}
	part, logPath := out+".part", out+".log"
	os.RemoveAll(part) // cut off before: made again
	if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
		return err
	}
	lf, err := os.Create(logPath)
	if err != nil {
		return err
	}
	defer lf.Close()
	all := append(slices.Clone(python[1:]), filepath.Join(tools, script))
	for _, a := range args {
		if a == ExportOut {
			a = part
		}
		all = append(all, a)
	}
	cmd := exec.CommandContext(ctx, python[0], all...)
	cmd.Dir = tools
	cmd.Env = append(os.Environ(), "PYTHONUNBUFFERED=1")
	if os.Getenv(ironCacheEnv) == "" {
		moveIRONCache()
		cmd.Env = append(cmd.Env, ironCacheEnv+"="+IRONCache())
	}
	bar := console.NewBar(log, fmt.Sprintf("exporting %s for AMD's NPU with IRON", name))
	last := &lastLine{w: lf}
	cmd.Stdout, cmd.Stderr = last, last
	stop := make(chan struct{})
	go func() { // the exporter's latest line after the bar
		t := time.NewTicker(500 * time.Millisecond)
		defer t.Stop()
		for {
			select {
			case <-stop:
				return
			case <-t.C:
				bar.Set(-1, last.get())
			}
		}
	}()
	err = cmd.Run()
	close(stop)
	if err != nil {
		bar.Done("failed")
		os.RemoveAll(part)
		return fmt.Errorf("exporting %s for AMD's NPU failed (%v); the end of %s:\n%s", name, err, logPath, tail(logPath, 15))
	}
	os.RemoveAll(out) // one without its manifest: cut off by hand
	if err := os.Rename(part, out); err != nil {
		bar.Done("failed")
		return err
	}
	if err := os.WriteFile(out+".done", nil, 0o644); err != nil {
		return err
	}
	bar.Done(fmt.Sprintf("in %s to %s", console.Duration(bar.Elapsed()), shortPath(out)))
	return nil
}

// ironCacheEnv is where IRON (mlir-aie's python/utils/compile) keeps the kernels it compiled.
const ironCacheEnv = "NPU_CACHE_HOME"

// IRONCache is IRON's cache of compiled kernels the exporters run with: iron under cache.Root,
// unless NPU_CACHE_HOME says. IRON's own default, ~/.npu/cache, is moved there before the first
// export (moveIRONCache: a rename on the same file system; tens of GB on a machine that has
// exported much) and left a link to it, so that IRON run by hand shares it.
func IRONCache() string {
	if d := os.Getenv(ironCacheEnv); d != "" {
		return d
	}
	return cache.Sub("iron")
}

// moveIRONCache moves IRON's own cache to IRONCache once (IRONCache), before an exporter runs.
func moveIRONCache() {
	if os.Getenv(ironCacheEnv) != "" {
		return
	}
	dir := IRONCache()
	ironCacheOnce.Do(func() {
		if _, err := os.Lstat(dir); err == nil {
			return
		}
		home, err := os.UserHomeDir()
		if err != nil {
			return
		}
		old := filepath.Join(home, ".npu", "cache")
		if fi, err := os.Lstat(old); err != nil || !fi.IsDir() {
			return
		}
		if err := os.MkdirAll(filepath.Dir(dir), 0o755); err != nil {
			return
		}
		if os.Rename(old, dir) == nil {
			os.Symlink(dir, old)
		}
	})
}

var ironCacheOnce sync.Once

// IRONPython is the command that runs Python in IRON's environment (IRONEnv, else python3 on
// PATH), checked by importing IRON and mlir-aie.
func IRONPython(ctx context.Context) ([]string, error) { return ironPython(ctx) }

// ironPython is the command that runs Python in IRON's environment (IRONEnv, else python3 on
// PATH), checked by importing IRON and mlir-aie.
func ironPython(ctx context.Context) ([]string, error) {
	var cmd []string
	switch v := os.Getenv(IRONEnv); {
	case v == "":
		p, err := exec.LookPath("python3")
		if err != nil {
			return nil, fmt.Errorf("%s is not set and there is no python3 on PATH", IRONEnv)
		}
		cmd = []string{p}
	default:
		v = expandTilde(v)
		fi, err := os.Stat(v)
		switch {
		case err != nil:
			return nil, fmt.Errorf("%s=%s: %w", IRONEnv, v, err)
		case fi.IsDir():
			cmd = []string{filepath.Join(v, "bin", "python3")}
		case fi.Mode()&0o111 != 0 && !strings.HasSuffix(v, ".sh"):
			cmd = []string{v}
		default: // a script to source
			cmd = []string{"bash", "-c", `source "$0" >/dev/null 2>&1 && exec python3 "$@"`, v}
		}
	}
	probe, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()
	c := exec.CommandContext(probe, cmd[0], append(slices.Clone(cmd[1:]), "-c", "import iron, aie.iron")...)
	if b, err := c.CombinedOutput(); err != nil {
		how := "python3 on PATH"
		if v := os.Getenv(IRONEnv); v != "" {
			how = IRONEnv + "=" + v
		}
		msg := strings.TrimSpace(string(b))
		if i := strings.LastIndex(msg, "\n"); i >= 0 {
			msg = msg[i+1:]
		}
		return nil, fmt.Errorf("%s cannot import IRON (%s)", how, cmpOr(msg, err.Error()))
	}
	return cmd, nil
}

func cmpOr(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

func expandTilde(p string) string {
	if p == "~" || strings.HasPrefix(p, "~/") {
		home, _ := os.UserHomeDir()
		return filepath.Join(home, p[1:])
	}
	return p
}

// lastLine writes through to w and keeps the last line written, for the bar.
type lastLine struct {
	w    io.Writer
	mu   sync.Mutex
	line string
	buf  []byte
}

func (l *lastLine) Write(p []byte) (int, error) {
	l.mu.Lock()
	l.buf = append(l.buf, p...)
	if i := strings.LastIndexAny(string(l.buf), "\r\n"); i >= 0 {
		lines := strings.FieldsFunc(string(l.buf[:i]), func(r rune) bool { return r == '\n' || r == '\r' })
		for j := len(lines) - 1; j >= 0; j-- {
			if s := strings.TrimSpace(lines[j]); s != "" {
				l.line = s
				break
			}
		}
		l.buf = slices.Clone(l.buf[i+1:])
	}
	l.mu.Unlock()
	return l.w.Write(p)
}

func (l *lastLine) get() string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.line
}

// tail is the last n lines of the file at path.
func tail(path string, n int) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	var lines []string
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		lines = append(lines, sc.Text())
		if len(lines) > n {
			lines = lines[1:]
		}
	}
	return strings.Join(lines, "\n")
}
