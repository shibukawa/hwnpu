//go:build darwin

package npuinfo

import (
	"bytes"
	"context"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"
)

func collect(ctx context.Context, opts Options, info *Info) error {
	di := &DarwinInfo{}
	info.Darwin = di
	di.Model, _ = syscall.Sysctl("hw.model")
	di.CPU, _ = syscall.Sysctl("machdep.cpu.brand_string")
	di.Version, _ = syscall.Sysctl("kern.osproductversion")
	guard("ioreg", info, func() error { return darwinANE(ctx, di) })
	di.Device = aneDevice(di.ANE)
	guard("processes", info, func() error {
		p, err := darwinProcesses(di.Clients)
		di.Processes = p
		return err
	})
	if opts.SampleInterval >= 0 {
		guard("ioreport", info, func() error {
			interval := opts.SampleInterval
			if interval == 0 {
				interval = 500 * time.Millisecond
			}
			r, err := darwinIOReport(ctx, interval)
			di.IOReport = r
			return err
		})
	}
	if !opts.SkipCaches {
		guard("caches", info, func() error {
			c, err := darwinCaches(ctx)
			di.Caches = c
			return err
		})
	}
	if os.Geteuid() == 0 {
		guard("powermetrics", info, func() error {
			m, err := darwinPowerMetrics(ctx)
			di.PowerMetrics = m
			return err
		})
	}
	return nil
}

func darwinANE(ctx context.Context, di *DarwinInfo) error {
	text, err := exec.CommandContext(ctx, "ioreg", "-w0", "-p", "IOService").Output()
	if err != nil {
		return err
	}
	di.ANEClasses = aneClasses(text)
	seen := map[string]bool{}
	for _, class := range di.ANEClasses {
		if seen[class] {
			continue // already inside a tree read before
		}
		out, err := exec.CommandContext(ctx, "ioreg", "-a", "-l", "-w0", "-r", "-c", class).Output()
		if err != nil {
			return err
		}
		if len(bytes.TrimSpace(out)) == 0 {
			continue
		}
		v, err := parsePlist(bytes.NewReader(out))
		if err != nil {
			return fmt.Errorf("ioreg -c %s: %w", class, err)
		}
		nodes := ioregNodes(v)
		nodeClasses(nodes, seen)
		di.ANE = append(di.ANE, nodes...)
	}
	di.Clients = aneClients(di.ANE)
	return nil
}

// darwinCaches finds the Neural Engine's compiled-program caches: per executable under
// ~/Library/Caches/NAME/com.apple.e5rt.e5bundlecache, a shared one in ~/Library/Caches, and
// the same under the per-user cache directory (DARWIN_USER_CACHE_DIR) for sandboxed apps and
// daemons.
func darwinCaches(ctx context.Context) ([]DarwinCache, error) {
	var bases []string
	if home, err := os.UserHomeDir(); err == nil {
		bases = append(bases, filepath.Join(home, "Library", "Caches"))
	}
	if out, err := exec.CommandContext(ctx, "getconf", "DARWIN_USER_CACHE_DIR").Output(); err == nil {
		if d := strings.TrimSpace(string(out)); d != "" {
			bases = append(bases, filepath.Clean(d))
		}
	}
	const name = "com.apple.e5rt.e5bundlecache"
	var paths []string
	for _, b := range bases {
		paths = append(paths, filepath.Join(b, name))
		m, _ := filepath.Glob(filepath.Join(b, "*", name))
		paths = append(paths, m...)
	}
	var out []DarwinCache
	for _, p := range paths {
		if ctx.Err() != nil {
			return out, ctx.Err()
		}
		fi, err := os.Stat(p)
		if err != nil || !fi.IsDir() {
			continue
		}
		c := DarwinCache{Path: p, Owner: filepath.Base(filepath.Dir(p))}
		if ents, err := os.ReadDir(p); err == nil {
			for _, e := range ents {
				if e.IsDir() {
					c.Entries = append(c.Entries, e.Name())
				}
			}
		}
		filepath.WalkDir(p, func(path string, d fs.DirEntry, err error) error {
			if err != nil || d.IsDir() {
				return nil
			}
			if i, err := d.Info(); err == nil {
				c.Bytes += i.Size()
				c.Files++
				if i.ModTime().After(c.Newest) {
					c.Newest = i.ModTime()
				}
			}
			return ctx.Err()
		})
		out = append(out, c)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Bytes > out[j].Bytes })
	return out, nil
}

func darwinPowerMetrics(ctx context.Context) (map[string]any, error) {
	out, err := exec.CommandContext(ctx, "powermetrics", "-n", "1", "-i", "500", "--samplers", "cpu_power", "-f", "plist").Output()
	if err != nil {
		return nil, err
	}
	out = bytes.Trim(out, "\x00 \n")
	v, err := parsePlist(bytes.NewReader(out))
	if err != nil {
		return nil, err
	}
	m, _ := v.(map[string]any)
	return m, nil
}
