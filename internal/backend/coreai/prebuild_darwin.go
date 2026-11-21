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
	"strings"
	"sync"
	"time"

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
	// Code and Weights are the sizes of the compiled regions (the ANE's LLIR bundle, what a
	// distribution would carry) and of the weights (resources.bin) in Out.
	Code, Weights int64
	Err           error
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
	var assets []string
	for _, a := range all {
		if LeftToDevice(a) {
			if o.Log != nil {
				fmt.Fprintf(o.Log, "%s: left to the device (LeftToDevice)\n", filepath.Base(a))
			}
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
	var tool string
	if !o.DryRun {
		if tool, err = FindCoreAIBuild(); err != nil {
			return nil, err
		}
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
	build := buildOne
	if o.Packed {
		build = buildPacked
	}
	var (
		wg  sync.WaitGroup
		sem = make(chan struct{}, jobs)
		mu  sync.Mutex
	)
	for i := range results {
		r := &results[i]
		if !o.Force && sameSource(r.Asset, r.Out) && onANE(r.Out) {
			r.Skipped = true
			r.Code, r.Weights = sizes(r.Out)
			if o.Log != nil {
				fmt.Fprintf(o.Log, "%s %s: prebuilt already (%s)\n", filepath.Base(r.Asset), r.Arch, filepath.Base(r.Out))
			}
			continue
		}
		if o.DryRun {
			if o.Log != nil {
				fmt.Fprintf(o.Log, "%s %s: would compile to %s\n", filepath.Base(r.Asset), r.Arch, r.Out)
			}
			continue
		}
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
	for i := range results {
		r := &results[i]
		if !errors.Is(r.Err, errOffANE) || ctx.Err() != nil {
			continue
		}
		if o.Log != nil {
			fmt.Fprintf(o.Log, "%s %s: compiling again alone\n", filepath.Base(r.Asset), r.Arch)
		}
		t0 := time.Now()
		r.Err = build(ctx, tool, r.Asset, r.Arch, r.Out)
		r.Elapsed += time.Since(t0)
		if r.Err == nil {
			r.Code, r.Weights = sizes(r.Out)
		}
		if o.Log != nil {
			if r.Err != nil {
				fmt.Fprintf(o.Log, "%s %s: FAILED again after %s: %v\n", filepath.Base(r.Asset), r.Arch, r.Elapsed.Round(time.Second), r.Err)
			} else {
				fmt.Fprintf(o.Log, "%s %s: %s alone (%.1f MB of compiled regions, %.0f MB of weights)\n", filepath.Base(r.Asset), r.Arch, r.Elapsed.Round(time.Second), float64(r.Code)/1e6, float64(r.Weights)/1e6)
			}
		}
	}
	for _, r := range results {
		if r.Err != nil {
			return results, fmt.Errorf("%s for %s: %w", filepath.Base(r.Asset), r.Arch, r.Err)
		}
	}
	return results, nil
}

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
	if err := cmd.Run(); err != nil {
		msg := strings.TrimSpace(stderr.String())
		if lines := strings.Split(msg, "\n"); len(lines) > 3 {
			msg = strings.Join(lines[len(lines)-3:], "\n")
		}
		var ee *exec.ExitError
		if errors.As(err, &ee) && ee.ExitCode() == -1 {
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
		return errOffANE
	}
	if err := os.RemoveAll(out); err != nil {
		return err
	}
	return os.Rename(built, out)
}

// buildPacked runs coreai-build on asset for arch, and packs the result at out (Pack).
func buildPacked(ctx context.Context, tool, asset, arch, out string) error {
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
