package coreai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu/internal/coreai/llir"
	"github.com/shibukawa/hwnpu/internal/mem"
)

// BuildOptions says how Build compiles.
type BuildOptions struct {
	// Archs are the architectures to compile for (Archs; the device's own when empty).
	Archs []string
	// Jobs are the compiles run at once (coreai-build takes one thread and 1 to 3 GB each;
	// DefaultJobs when 0).
	Jobs int
	// Force compiles again over a prebuilt program that is there.
	Force bool
	// DryRun lists what would be compiled without compiling.
	DryRun bool
	// Log gets a line for each program as it finishes, when not nil.
	Log io.Writer
	// Packed writes the programs packed beside their export (PackedPath: what the shared layers
	// carry for an architecture), not whole beside their sources.
	Packed bool
	// Left, when set, is why the program at asset is left to the device on arch, "" when it is
	// not: a rule of the model's recipe (internal/recipe). Build compiles none of those.
	Left func(asset, arch string) string
	// Tool is the version of coreai-build (BuildToolVersion's, asked when ""): a packed build
	// takes no prebuilt program another one made.
	Tool string
	// Model is the model of the exports, for the record of a packed build (RecordFile).
	Model string
}

// DefaultJobs is the compiles Build runs at once by default: one for every 4 GB of memory beyond
// 6, at most half the cores, at least one.
func DefaultJobs() int {
	n := int((mem.Physical()>>30 - 6) / 4)
	if h := runtime.NumCPU() / 2; n > h {
		n = h
	}
	if n < 1 {
		n = 1
	}
	return n
}

// BuildResult is what became of one program for one architecture.
type BuildResult struct {
	Asset, Arch, Out string
	Elapsed          time.Duration
	// Skipped: the prebuilt program was there already (not Force).
	Skipped bool
	// Reused is the export whose prebuilt program of the same source was taken (a packed
	// build, not Force), "" when none was.
	Reused string
	// Why is why the prebuilt programs of the same source there were were not taken (it was
	// compiled), "" when there was none.
	Why string
	// Left is why the program is left to the device on the architecture (BuildOptions.Left):
	// not compiled.
	Left string
	// Code and Weights are the sizes of the compiled regions (the ANE's LLIR bundle, what a
	// distribution would carry) and of the weights (resources.bin) in Out.
	Code, Weights int64
	Err           error

	jobs      int        // the compiles that ran at once with it
	ran       bool       // compiled (and packed)
	from, bad *candidate // what it was taken from; what failed on the device
}

// FindCoreAIBuild finds coreai-build, the Metal Toolchain's compiler of Core AI programs: xcrun's
// answer, with Xcode's developer directory when the command line tools' lacks it.
func FindCoreAIBuild() (string, error) {
	try := func(env []string) (string, error) {
		cmd := exec.Command("xcrun", "-f", "coreai-build")
		cmd.Env = env
		out, err := cmd.Output()
		if err != nil {
			return "", err
		}
		return strings.TrimSpace(string(out)), nil
	}
	if p, err := try(os.Environ()); err == nil && p != "" {
		return p, nil
	}
	env := append([]string{}, os.Environ()...)
	env = append(env, "DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer")
	if p, err := try(env); err == nil && p != "" {
		return p, nil
	}
	return "", errors.New("coreai-build was not found: it comes with Xcode 27's Metal Toolchain (xcodebuild -downloadComponent MetalToolchain)")
}

// Build compiles the programs under paths ahead of time for each architecture with coreai-build,
// writing X.ARCH.aimodelc beside X.aimodel (Prebuilt loads it on a device of that architecture
// while Core AI's cache has no entry). The compiles run o.Jobs at a time, each its own process (a
// crash of the compiler on one architecture, as h14s and h14d with the Neural Engine preferred in
// Metal Toolchain 27.2, fails that one alone). The results come in the order of the programs then
// the architectures; err is the first failure of the compiler, after everything ran.
func Build(ctx context.Context, paths []string, o BuildOptions) ([]BuildResult, error) {
	all, err := Assets(paths)
	if err != nil {
		return nil, err
	}
	var assets, leftAll []string
	for _, a := range all {
		if LeftToDevice(a) {
			if o.Log != nil {
				fmt.Fprintf(o.Log, "%s: left to the device (LeftToDevice)\n", filepath.Base(a))
			}
			leftAll = append(leftAll, a)
			continue
		}
		assets = append(assets, a)
	}
	archs := o.Archs
	if len(archs) == 0 {
		if a := Arch(); a != "" {
			archs = []string{a}
		} else {
			return nil, errors.New("no architecture given and Core AI is not available here to name the device's")
		}
	}
	for _, a := range archs {
		if !known(a) {
			return nil, fmt.Errorf("unknown architecture %s (coreai-build knows %s)", a, strings.Join(Archs, " "))
		}
	}
	version := o.Tool
	if version == "" && o.Packed {
		version, _ = BuildToolVersion()
	}
	jobs := o.Jobs
	if jobs <= 0 {
		jobs = DefaultJobs()
	}
	results := make([]BuildResult, 0, len(assets)*len(archs))
	for _, a := range assets {
		for _, arch := range archs {
			out := PrebuiltPath(a, arch)
			if o.Packed {
				if out = PackedPath(a, arch); out == "" {
					return nil, fmt.Errorf("%s is in no export (NAME.coreai) to pack its prebuilt programs beside", a)
				}
			}
			results = append(results, BuildResult{Asset: a, Arch: arch, Out: out})
		}
	}
	logf := func(format string, a ...any) {
		if o.Log != nil {
			fmt.Fprintf(o.Log, format, a...)
		}
	}
	u := &reuser{tool: version, jobs: jobs, dry: o.DryRun, log: logf}
	build := buildOne
	if o.Packed {
		build = func(ctx context.Context, tool, asset, arch, out string) error {
			return buildPacked(ctx, tool, asset, arch, out, u.stash)
		}
	}
	// what is kept, taken or left; the rest is compiled
	var queue []*BuildResult
	for i := range results {
		r := &results[i]
		name := filepath.Base(r.Asset)
		if o.Left != nil {
			if why := o.Left(r.Asset, r.Arch); why != "" {
				r.Left = why
				logf("%s %s: left to the device by the recipe (%s)\n", name, r.Arch, why)
				if o.Packed && !o.DryRun {
					// one prebuilt before the rule goes aside: a device would take it from the
					// bundle
					if err := u.stash(r.Asset, r.Arch, r.Out); err != nil {
						logf("%s %s: %v\n", name, r.Arch, err)
					}
				}
				continue
			}
		}
		if !o.Force && !o.Packed && sameSource(r.Asset, r.Out) && onANE(r.Out) && !oldOverrun(r.Arch, r.Out) {
			r.Skipped = true
			r.Code, r.Weights = sizes(r.Out)
			logf("%s %s: prebuilt already (%s)\n", name, r.Arch, filepath.Base(r.Out))
			continue
		}
		if !o.Force && o.Packed {
			c, why, err := u.decide(r)
			switch {
			case err != nil:
				r.Err = err
				logf("%s %s: not compiled: %v\n", name, r.Arch, err)
				continue
			case c != nil && oldOverrun(r.Arch, c.dir):
				// compiled again, which says so
				r.Why = "the one of " + c.label() + ": its tasks read past their kernels"
			case c != nil && c.dir == r.Out:
				r.Skipped = true
				r.Code, r.Weights = sizes(r.Out)
				logf("%s %s: prebuilt already (%s)\n", name, r.Arch, filepath.Base(r.Out))
				continue
			case c != nil && o.DryRun:
				logf("%s %s: would take the one of %s\n", name, r.Arch, c.label())
				continue
			case c != nil:
				if err := u.take(c, r.Asset, r.Arch, r.Out); err != nil {
					r.Why = "the one of " + c.label() + ": " + err.Error()
					break
				}
				r.Reused, r.from = c.export, c
				r.Code, r.Weights = sizes(r.Out)
				logf("%s %s: taken from %s (same source, coreai-build %s)\n", name, r.Arch, c.label(), version)
				continue
			default:
				r.Why = why
			}
		}
		if o.DryRun {
			if r.Why != "" {
				logf("%s %s: would compile to %s (not %s)\n", name, r.Arch, r.Out, r.Why)
			} else {
				logf("%s %s: would compile to %s\n", name, r.Arch, r.Out)
			}
			continue
		}
		if r.Why != "" {
			logf("%s %s: compiling (not %s)\n", name, r.Arch, r.Why)
		}
		queue = append(queue, r)
	}
	var tool string
	if len(queue) > 0 {
		if tool, err = FindCoreAIBuild(); err != nil {
			if o.Packed {
				u.write(results, leftAll, archs, o.Model) // what was taken
			}
			return results, err
		}
	}
	var (
		wg  sync.WaitGroup
		sem = make(chan struct{}, jobs)
		mu  sync.Mutex
	)
	for _, r := range queue {
		r.jobs = min(jobs, len(queue))
		wg.Add(1)
		go func() {
			defer wg.Done()
			select {
			case sem <- struct{}{}:
			case <-ctx.Done():
				r.Err = ctx.Err()
				return
			}
			defer func() { <-sem }()
			t0 := time.Now()
			r.Err = build(ctx, tool, r.Asset, r.Arch, r.Out)
			r.Elapsed = time.Since(t0)
			if r.Err == nil {
				r.ran = true
				r.Code, r.Weights = sizes(r.Out)
			}
			if o.Log != nil {
				mu.Lock()
				if r.Err != nil {
					fmt.Fprintf(o.Log, "%s %s: FAILED after %s: %v\n", filepath.Base(r.Asset), r.Arch, r.Elapsed.Round(time.Second), r.Err)
				} else {
					fmt.Fprintf(o.Log, "%s %s: %s (%.1f MB of compiled regions, %.0f MB of weights)\n", filepath.Base(r.Asset), r.Arch, r.Elapsed.Round(time.Second), float64(r.Code)/1e6, float64(r.Weights)/1e6)
				}
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	for _, r := range queue {
		if !errors.Is(r.Err, errOffANE) || ctx.Err() != nil {
			continue
		}
		logf("%s %s: compiling again alone\n", filepath.Base(r.Asset), r.Arch)
		r.jobs = 1
		t0 := time.Now()
		r.Err = build(ctx, tool, r.Asset, r.Arch, r.Out)
		r.Elapsed += time.Since(t0)
		if r.Err == nil {
			r.ran = true
			r.Code, r.Weights = sizes(r.Out)
		}
		if r.Err != nil {
			logf("%s %s: FAILED again after %s: %v\n", filepath.Base(r.Asset), r.Arch, r.Elapsed.Round(time.Second), r.Err)
		} else {
			logf("%s %s: %s alone (%.1f MB of compiled regions, %.0f MB of weights)\n", filepath.Base(r.Asset), r.Arch, r.Elapsed.Round(time.Second), float64(r.Code)/1e6, float64(r.Weights)/1e6)
		}
	}
	if o.Packed && !o.DryRun {
		if err := u.write(results, leftAll, archs, o.Model); err != nil {
			logf("the record of the build (%s): %v\n", RecordFile, err)
		}
	}
	for _, r := range results {
		if r.Err != nil {
			return results, fmt.Errorf("%s for %s: %w", filepath.Base(r.Asset), r.Arch, r.Err)
		}
	}
	return results, nil
}

// IsBadOnDevice reports whether err is a build's of a program whose prebuilt program of the same
// source a device failed (its record says so): compiled again it would be the same.
func IsBadOnDevice(err error) bool { return errors.Is(err, errBadOnDevice) }

// errOffANE: coreai-build wrote a program without a Neural Engine region. Seen on 2 of 5 double
// groups of Chroma1-Flash at 1024² with two compiles at once, and not on the same programs alone:
// Build compiles such a program again by itself after the others.
var errOffANE = errors.New("coreai-build left the program off the Neural Engine (no compiled region)")

func known(arch string) bool {
	for _, a := range Archs {
		if a == arch {
			return true
		}
	}
	return false
}

// sameSource reports whether the prebuilt program at out is of the asset's bytecode: both hashes
// there and equal.
func sameSource(asset, out string) bool {
	want, err := os.ReadFile(filepath.Join(asset, "main.hash"))
	if err != nil {
		return false
	}
	have, err := os.ReadFile(filepath.Join(out, "main.hash"))
	return err == nil && bytes.Equal(want, have)
}

// buildOne runs coreai-build on asset for arch into a temporary directory beside out, then moves
// the result to out (what was there is replaced once the new one is whole).
func buildOne(ctx context.Context, tool, asset, arch, out string) error {
	tmp, err := os.MkdirTemp(filepath.Dir(out), ".prebuild-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmp)
	cmd := exec.CommandContext(ctx, tool, "compile", asset, "--platform", "macOS", "--architecture", arch, "--preferred-compute", "neural-engine", "--output", tmp+"/")
	if os.Getenv("DEVELOPER_DIR") == "" {
		cmd.Env = append(os.Environ(), "DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer")
	}
	var stderr bytes.Buffer
	cmd.Stdout = io.Discard
	cmd.Stderr = &stderr
	t0 := time.Now()
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("coreai-build: %v", err)
	}
	pid := cmd.Process.Pid
	if err := cmd.Wait(); err != nil {
		msg := strings.TrimSpace(stderr.String())
		if lines := strings.Split(msg, "\n"); len(lines) > 3 {
			msg = strings.Join(lines[len(lines)-3:], "\n")
		}
		if ctx.Err() == nil {
			if why := logReason(pid, t0); why != "" {
				msg = strings.TrimSpace(msg + "\n" + why)
			}
		}
		var ee *exec.ExitError
		if errors.As(err, &ee) && ee.ExitCode() == -1 {
			if msg != "" {
				return fmt.Errorf("coreai-build crashed (%v): %s", ee.ProcessState, msg)
			}
			return fmt.Errorf("coreai-build crashed (%v)", ee.ProcessState)
		}
		if msg != "" {
			return fmt.Errorf("coreai-build: %v: %s", err, msg)
		}
		return fmt.Errorf("coreai-build: %v", err)
	}
	built := filepath.Join(tmp, filepath.Base(out))
	if !sameSource(asset, built) {
		entries, _ := os.ReadDir(tmp)
		var names []string
		for _, e := range entries {
			names = append(names, e.Name())
		}
		return fmt.Errorf("coreai-build wrote %v, not %s of the asset's hash", names, filepath.Base(out))
	}
	if !onANE(built) {
		// no Neural Engine region: the compiler left the program to the GPU (hwnpu would refuse
		// it as off the NPU), so nothing is kept, nor an older output of the same kind
		if _, err := os.Stat(out); err == nil && !onANE(out) {
			os.RemoveAll(out)
		}
		if why := logReason(pid, t0); why != "" {
			return fmt.Errorf("%w: %s", errOffANE, why)
		}
		return errOffANE
	}
	// a program whose tasks read past the kernels they load is not kept: the M1 does not end the
	// first call of one (or its finishing service crashes), and takes the Neural Engine from
	// every process meanwhile (llir.Overruns). An older output of the asset goes with it.
	if _, err := llir.CheckKernels(built); err != nil {
		if errors.Is(err, llir.ErrKernelOverrun) {
			os.RemoveAll(out)
		}
		return err
	}
	if err := os.RemoveAll(out); err != nil {
		return err
	}
	return os.Rename(built, out)
}

// oldOverrun reports whether the prebuilt program at out, whole or packed, that an earlier build
// made for arch has tasks that read past the kernels they load (llir.Overruns): one built before
// buildOne checked (2026-11-27). Only the programs for the M1's family are read again: the h13g
// compiler alone wrote such tasks (none in 790 h14g and h15g programs, nor in the h14g to h17g
// programs of the parts that have them on h13g), and reading the LLIR of an image model's
// programs takes a minute a run.
func oldOverrun(arch, out string) bool {
	if !strings.HasPrefix(arch, "h13") {
		return false
	}
	_, err := llir.CheckKernels(out)
	return errors.Is(err, llir.ErrKernelOverrun)
}

// buildPacked runs coreai-build on asset for arch, and packs the result at out (Pack), the one
// there given to stash first.
func buildPacked(ctx context.Context, tool, asset, arch, out string, stash func(asset, arch, out string) error) error {
	if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
		return err
	}
	tmp, err := os.MkdirTemp(filepath.Dir(out), ".full-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmp)
	full := filepath.Join(tmp, filepath.Base(out))
	if err := buildOne(ctx, tool, asset, arch, full); err != nil {
		return err
	}
	if err := stash(asset, arch, out); err != nil {
		return err
	}
	return Pack(full, asset, out)
}

// onANE reports whether the prebuilt program at out has its function main on the Neural Engine:
// its MPSGraph manifest's mark, as placement reads of a cache entry.
func onANE(out string) bool {
	ms, _ := filepath.Glob(filepath.Join(out, "main-*-delegates", "MPSGraph", "mpsExecutable.mpsgraphpackage", "manifest.plist"))
	for _, m := range ms {
		if b, err := os.ReadFile(m); err == nil && fullyPlaced(b, "main") {
			return true
		}
	}
	return false
}

// sizes are the bytes of the compiled regions (the LLIR bundles) and of the weights in a prebuilt
// program.
func sizes(out string) (code, weights int64) {
	filepath.WalkDir(out, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		fi, err := d.Info()
		if err != nil {
			return nil
		}
		switch {
		case d.Name() == "resources.bin":
			weights += fi.Size()
		case strings.HasSuffix(d.Name(), MapSuffix):
			var m resMap
			if b, err := os.ReadFile(p); err == nil && json.Unmarshal(b, &m) == nil {
				weights += m.Size
			}
		case strings.Contains(p, ".llir.bundle/"):
			code += fi.Size()
		}
		return nil
	})
	return
}

// logReason is what coreai-build (process pid, started at t0) logged of why it failed: its
// compiler's errors go to os_log, not to its output (docs/known-issues.md). The last few lines
// that say so, "" when none or the log is not to be read.
func logReason(pid int, t0 time.Time) string {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	out, err := exec.CommandContext(ctx, "/usr/bin/log", "show", "--info", "--style", "compact",
		"--start", t0.Add(-time.Second).Format("2006-01-02 15:04:05"),
		"--predicate", fmt.Sprintf("processID == %d", pid)).Output()
	if err != nil {
		return ""
	}
	return reasonLines(string(out))
}

// reasonLines are the last lines of a log that tell a failure, their messages alone, without
// repeats.
func reasonLines(log string) string {
	var lines []string
	for _, l := range strings.Split(log, "\n") {
		low := strings.ToLower(l)
		if !strings.Contains(low, "fail") && !strings.Contains(low, "unsupported") && !strings.Contains(low, "error") {
			continue
		}
		// compact lines: DATE TIME TYPE PROCESS[PID:TID] [SUBSYSTEM:CATEGORY] MESSAGE
		if i := strings.Index(l, "] "); i >= 0 {
			l = l[i+2:]
			if strings.HasPrefix(l, "[") {
				if j := strings.Index(l, "] "); j >= 0 {
					l = l[j+2:]
				}
			}
		}
		l = strings.TrimSpace(l)
		if l != "" && !slices.Contains(lines, l) {
			lines = append(lines, l)
		}
	}
	if len(lines) > 3 {
		lines = lines[len(lines)-3:]
	}
	return strings.Join(lines, "; ")
}

// BuildToolVersion is coreai-build's version ("3605.6.3" of "coreai-build 3605.6.3").
func BuildToolVersion() (string, error) {
	tool, err := FindCoreAIBuild()
	if err != nil {
		return "", err
	}
	cmd := exec.Command(tool, "--version")
	if os.Getenv("DEVELOPER_DIR") == "" {
		cmd.Env = append(os.Environ(), "DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer")
	}
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("coreai-build --version: %v", err)
	}
	f := strings.Fields(string(out))
	if len(f) == 0 {
		return "", fmt.Errorf("coreai-build --version wrote nothing")
	}
	return f[len(f)-1], nil
}
