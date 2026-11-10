package probe

import (
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/probe"
)

// The cache mode measures the Neural Engine's own cache of compiled programs (Core ML): what
// makes a program that hwnpu's compile cache already holds load in 0.1 s on one run and take a
// minute or two (ANECompilerService) on the next (docs/handoff-ane-cache.md). A configuration is
// N programs of S bytes each, written once and loaded in order by a new process per round;
// a load that takes longer than -cache-hit is a miss. Sweeping N × S tells the capacity, and
// whether it counts bytes or programs; -cache-hold, -cache-order, -cache-passes and
// -cache-names tell whether closing a program, the order, the process and the executable's
// name matter.

// cacheOpts are the cache mode's flags.
type cacheOpts struct {
	format        string
	sizes, totals []float64 // GiB
	rounds        int
	holds, orders []string
	passes        int
	names         []string
	hit           time.Duration
	pause         time.Duration
	keep          bool
}

// parse reads the lists of the flags and checks the rest.
func (o *cacheOpts) parse(sizes, totals, work string) error {
	var err error
	if o.sizes, err = gibs("-cache-sizes", sizes); err != nil {
		return err
	}
	if o.totals, err = gibs("-cache-totals", totals); err != nil {
		return err
	}
	slices.Sort(o.totals)
	if b, ok := probe.BytesPerParam[o.format]; !ok || b < 0.5 || o.format == "int4" {
		return fmt.Errorf("-cache-format: %q is not a format the Neural Engine runs (fp16, int8, lut4)", o.format)
	}
	for _, h := range o.holds {
		if h != "close" && h != "open" {
			return fmt.Errorf("-cache-hold: %q is not close or open", h)
		}
	}
	for _, x := range o.orders {
		if x != "same" && x != "reverse" {
			return fmt.Errorf("-cache-order: %q is not same or reverse", x)
		}
	}
	if len(o.holds) == 0 || len(o.orders) == 0 {
		return errors.New("-cache-hold and -cache-order: at least one each")
	}
	if o.rounds < 1 || o.passes < 1 {
		return errors.New("-cache-rounds and -cache-passes: at least 1")
	}
	if o.keep && work == "" {
		return errors.New("-cache-keep keeps the programs in -dir: give one")
	}
	return nil
}

func gibs(flagName, s string) ([]float64, error) {
	var out []float64
	for _, f := range split(s) {
		v, err := strconv.ParseFloat(f, 64)
		if err != nil || v <= 0 {
			return nil, fmt.Errorf("%s: %q is not a size in GiB", flagName, f)
		}
		out = append(out, v)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("%s: empty", flagName)
	}
	return out, nil
}

// cacheResult is one configuration: its programs and every round of loading them.
type cacheResult struct {
	Format       string       `json:"format"`
	Layers       int          `json:"layers"` // of each program
	ProgramBytes int64        `json:"program_bytes"`
	Programs     int          `json:"programs"`
	TotalBytes   int64        `json:"total_bytes"`
	Hold         string       `json:"hold"`  // close: each closed before the next loads; open: all open until the pass ends
	Order        string       `json:"order"` // same: every round in one order; reverse: every other round reversed
	HitS         float64      `json:"hit_s"` // a load at or under this is a hit
	BuildS       float64      `json:"build_s"`
	Rounds       []cacheRound `json:"rounds"`
	Error        string       `json:"error,omitempty"`
}

// cacheRound is one process loading the programs, -cache-passes times.
type cacheRound struct {
	Exe    string        `json:"exe"`   // the executable's name
	Order  []int         `json:"order"` // the programs in the order loaded, from 0
	Passes [][]cacheLoad `json:"passes"`
	MaxRSS int64         `json:"max_rss,omitempty"`
	// What the system's caches of compiled programs hold after the round (~/Library/Caches).
	ExeCacheBytes    int64  `json:"exe_cache_bytes,omitempty"`
	SharedCacheBytes int64  `json:"shared_cache_bytes,omitempty"`
	Error            string `json:"error,omitempty"`
}

// cacheLoad is one program loaded: the load (Core ML's MLModel, where the Neural Engine's
// compilation runs when its cache misses) and the first call.
type cacheLoad struct {
	Program int     `json:"program"`
	LoadS   float64 `json:"load_s"`
	RunS    float64 `json:"run_s"`
	Cached  bool    `json:"hwnpu_cached"` // hwnpu's compile cache held the .mlmodelc
	OnNPU   bool    `json:"on_npu"`
	Error   string  `json:"error,omitempty"`
}

// cacheRun is what the child sends back.
type cacheRun struct {
	Passes [][]cacheLoad `json:"passes"`
	MaxRSS int64         `json:"max_rss"`
}

func (r cacheResult) hit(l cacheLoad) bool { return l.Error == "" && l.LoadS+l.RunS <= r.HitS }

// hits counts the hits of a pass.
func (r cacheResult) hits(pass []cacheLoad) int {
	n := 0
	for _, l := range pass {
		if r.hit(l) {
			n++
		}
	}
	return n
}

// warm reports whether every pass after the first round's first hit all its programs, in the
// rounds under this executable's name before any under another (-cache-names).
func (r cacheResult) warm() bool {
	if len(r.Rounds) < 2 || r.Error != "" {
		return false
	}
	for i, rd := range r.Rounds {
		if rd.Exe != r.Rounds[0].Exe {
			break
		}
		for j, pass := range rd.Passes {
			if (i > 0 || j > 0) && r.hits(pass) < len(pass) {
				return false
			}
		}
		if rd.Error != "" {
			return false
		}
	}
	return true
}

// caches runs every configuration: each size, each total (ascending), each hold and order.
func (p *prober) caches(rep *report, o cacheOpts, save func()) {
	per := probe.WeightBytes(o.format, 1)
	fmt.Printf("cache: %s programs of %s GiB, %s GiB in all, %d rounds (a new process each), a hit at %.1f s or less\n",
		o.format, joinFloats(o.sizes), joinFloats(o.totals), o.rounds, o.hit.Seconds())
	for _, size := range o.sizes {
		layers := max(1, int(math.Round(size*(1<<30)/float64(per))))
		for _, total := range o.totals {
			n := max(1, int(math.Round(total/size)))
			for _, hold := range o.holds {
				for _, order := range o.orders {
					if p.ctx.Err() != nil {
						return
					}
					r := p.cacheConfig(o, layers, n, hold, order)
					rep.Cache = append(rep.Cache, r)
					save()
				}
			}
		}
	}
}

// cacheConfig writes the programs of one configuration and loads them round after round.
func (p *prober) cacheConfig(o cacheOpts, layers, n int, hold, order string) cacheResult {
	r := cacheResult{Format: o.format, Layers: layers, ProgramBytes: probe.WeightBytes(o.format, layers), Programs: n,
		Hold: hold, Order: order, HitS: o.hit.Seconds()}
	r.TotalBytes = int64(n) * r.ProgramBytes
	fmt.Printf("  %d × %.2f GiB = %.2f GiB, %s, %s order:\n", n, probe.GiB(r.ProgramBytes), probe.GiB(r.TotalBytes), hold, order)
	// Programs of their own per configuration (the seed and the directory name tell them apart),
	// so that a configuration starts cold; with -cache-keep they stay and later runs reuse them
	// (the package's bytes, and with them the caches' keys, change when it is written again).
	name := fmt.Sprintf("cache-%s-l%d-n%d-%s-%s", o.format, layers, n, hold, order)
	dir := filepath.Join(p.dir, name)
	if !o.keep {
		defer os.RemoveAll(dir)
	}
	seed := uint64(0)
	for _, c := range name {
		seed = seed*131 + uint64(c)
	}
	var paths []string
	t := time.Now()
	for i := range n {
		path := filepath.Join(dir, fmt.Sprintf("p%02d.mlpackage", i))
		if _, err := os.Stat(filepath.Join(path, "Manifest.json")); err != nil {
			if err := probe.WriteMLPackage(path, o.format, layers, seed+uint64(i)); err != nil {
				r.Error = "build: " + err.Error()
				fmt.Printf("    %s\n", r.Error)
				return r
			}
		}
		paths = append(paths, path)
	}
	r.BuildS = time.Since(t).Seconds()
	cacheDir := filepath.Join(dir, "hwnpu-cache") // hwnpu's compile cache, as the commands keep one

	exe, err := os.Executable()
	if err != nil {
		r.Error = err.Error()
		return r
	}
	type step struct{ exe, label string }
	var steps []step
	for range o.rounds {
		steps = append(steps, step{exe, filepath.Base(exe)})
	}
	for _, nm := range o.names {
		cp, err := p.exeCopy(exe, nm)
		if err != nil {
			r.Error = "copy of the executable: " + err.Error()
			return r
		}
		// a round under the other name, then one under this name again
		steps = append(steps, step{cp, nm}, step{exe, filepath.Base(exe)})
	}
	for i, s := range steps {
		if p.ctx.Err() != nil {
			break
		}
		if i > 0 && o.pause > 0 {
			time.Sleep(o.pause)
		}
		idx := make([]int, n)
		for j := range idx {
			idx[j] = j
		}
		if order == "reverse" && i%2 == 1 {
			slices.Reverse(idx)
		}
		var ordered []string
		for _, j := range idx {
			ordered = append(ordered, paths[j])
		}
		rd := cacheRound{Exe: s.label, Order: idx}
		var run cacheRun
		spec := childSpec{Kind: "cache", Format: o.format, Layers: layers, Paths: ordered, Layout: hold, N: o.passes, CacheDir: cacheDir}
		prev := p.exe
		p.exe = s.exe
		err := p.child(spec, &run)
		p.exe = prev
		if err != nil {
			rd.Error = err.Error()
		}
		for _, pass := range run.Passes {
			for k := range pass {
				pass[k].Program = idx[k]
			}
		}
		rd.Passes, rd.MaxRSS = run.Passes, run.MaxRSS
		rd.ExeCacheBytes, rd.SharedCacheBytes = systemCaches(s.label)
		r.Rounds = append(r.Rounds, rd)
		for j, pass := range rd.Passes {
			var times []string
			for _, l := range pass {
				times = append(times, loadCell(r, l))
			}
			label := fmt.Sprintf("round %d", i+1)
			if len(rd.Passes) > 1 {
				label += fmt.Sprintf(" pass %d", j+1)
			}
			fmt.Printf("    %-16s %-12s %2d/%d hits  %s\n", label, s.label, r.hits(pass), len(pass), strings.Join(times, " "))
		}
		if rd.Error != "" {
			fmt.Printf("    round %d: %s\n", i+1, short(rd.Error, 200))
		}
	}
	return r
}

// exeCopy is this executable copied to the work directory under name (once).
func (p *prober) exeCopy(exe, name string) (string, error) {
	dst := filepath.Join(p.dir, "exe", name)
	if _, err := os.Stat(dst); err == nil {
		return dst, nil
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return "", err
	}
	src, err := os.Open(exe)
	if err != nil {
		return "", err
	}
	defer src.Close()
	f, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o755)
	if err != nil {
		return "", err
	}
	if _, err := io.Copy(f, src); err != nil {
		f.Close()
		return "", err
	}
	return dst, f.Close()
}

// systemCaches are the sizes of the system's caches of compiled programs that are known
// (macOS): the executable's own under ~/Library/Caches/NAME and the one shared beside them.
func systemCaches(exe string) (own, shared int64) {
	home, err := os.UserHomeDir()
	if err != nil {
		return 0, 0
	}
	caches := filepath.Join(home, "Library", "Caches")
	return probe.DirSize(filepath.Join(caches, exe, "com.apple.e5rt.e5bundlecache")),
		probe.DirSize(filepath.Join(caches, "com.apple.e5rt.e5bundlecache"))
}

// loadCell is a load's time in seconds, * when it missed.
func loadCell(r cacheResult, l cacheLoad) string {
	if l.Error != "" {
		return "err"
	}
	s := fmt.Sprintf("%.1f", l.LoadS+l.RunS)
	if !r.hit(l) {
		s += "*"
	}
	return s
}

// cacheLoads is the child side: load the programs in order, run each once, and close each
// before the next (hold close) or all at the end of the pass (open); passes times over.
func (p *prober) cacheLoads(format string, layers int, paths []string, hold string, passes int) cacheRun {
	var run cacheRun
	for range max(passes, 1) {
		var open []*loaded
		var pass []cacheLoad
		for _, path := range paths {
			if p.ctx.Err() != nil {
				break
			}
			var r result
			t := time.Now()
			l := p.loadPart(format, probe.Whole(layers), "npu", 0, nil, &sharedPart{path: path, fileBytes: 0}, &r)
			c := cacheLoad{LoadS: time.Since(t).Seconds(), Error: r.Error}
			if l != nil {
				c.Cached = l.g.Cached()
				t = time.Now()
				if err := l.req.Run(p.ctx); err != nil {
					c.Error = "run: " + err.Error()
				}
				c.RunS = time.Since(t).Seconds()
				c.OnNPU = c.Error == "" && r.Verified && r.NPUOps > 0 && r.OtherOps == 0
				if hold == "open" {
					open = append(open, l)
				} else {
					l.close()
				}
			}
			pass = append(pass, c)
		}
		for _, l := range open {
			l.close()
		}
		run.Passes = append(run.Passes, pass)
	}
	run.MaxRSS = maxRSS()
	return run
}

func joinFloats(fs []float64) string {
	var s []string
	for _, f := range fs {
		s = append(s, fmt.Sprintf("%g", f))
	}
	return strings.Join(s, ", ")
}

// cacheMarkdown writes the cache section: a row per configuration with the hits of each round,
// the largest total that stays warm per size, hold and order, then every load's time.
func cacheMarkdown(sb *strings.Builder, rs []cacheResult) {
	sb.WriteString("\n## Neural Engine's cache of compiled programs\n\n")
	fmt.Fprintf(sb, "Each configuration is its own programs, loaded and run once in order by a new process per round (hwnpu's compile cache kept, so a load that takes long is the Neural Engine compiling again). A hit is a load and first call of %.1f s or less. The first round compiles everything; warm means every later round hit every program.\n\n", rs[0].HitS)
	sb.WriteString("| format | program | programs | total | hold | order | hits per round | warm | max RSS | error |\n|---|---|---|---|---|---|---|---|---|---|\n")
	for _, r := range rs {
		var hits []string
		var rss int64
		for _, rd := range r.Rounds {
			for _, pass := range rd.Passes {
				h := fmt.Sprintf("%d/%d", r.hits(pass), len(pass))
				if rd.Exe != r.Rounds[0].Exe {
					h += " (" + rd.Exe + ")"
				}
				hits = append(hits, h)
			}
			rss = max(rss, rd.MaxRSS)
			if rd.Error != "" && r.Error == "" {
				r.Error = rd.Error
			}
		}
		warm := "no"
		if r.warm() {
			warm = "yes"
		}
		fmt.Fprintf(sb, "| %s | %.2f GiB | %d | %.2f GiB | %s | %s | %s | %s | %.2f GiB | %s |\n", r.Format, probe.GiB(r.ProgramBytes), r.Programs, probe.GiB(r.TotalBytes),
			r.Hold, r.Order, strings.Join(hits, ", "), warm, probe.GiB(rss), strings.ReplaceAll(short(r.Error, 200), "|", "/"))
	}

	sb.WriteString("\nLargest total that stays warm, and smallest that does not:\n\n")
	type key struct {
		format      string
		size        int64
		hold, order string
	}
	var keys []key
	for _, r := range rs {
		k := key{r.Format, r.ProgramBytes, r.Hold, r.Order}
		if !slices.Contains(keys, k) {
			keys = append(keys, k)
		}
	}
	for _, k := range keys {
		var warm, cold int64 = -1, -1
		for _, r := range rs {
			if (key{r.Format, r.ProgramBytes, r.Hold, r.Order}) != k || r.Error != "" {
				continue
			}
			if r.warm() {
				warm = max(warm, r.TotalBytes)
			} else if cold < 0 || r.TotalBytes < cold {
				cold = r.TotalBytes
			}
		}
		gib := func(b int64) string {
			if b < 0 {
				return "none"
			}
			return fmt.Sprintf("%.2f GiB", probe.GiB(b))
		}
		fmt.Fprintf(sb, "- %s programs of %.2f GiB, %s, %s order: warm up to %s, not at %s\n", k.format, probe.GiB(k.size), k.hold, k.order, gib(warm), gib(cold))
	}

	sb.WriteString("\nEvery load, in seconds (load and first call; * a miss), in the order loaded; the program's number in brackets. The system's caches are ~/Library/Caches/NAME/com.apple.e5rt.e5bundlecache of the executable and ~/Library/Caches/com.apple.e5rt.e5bundlecache after the round.\n")
	for _, r := range rs {
		fmt.Fprintf(sb, "\n### %d × %.2f GiB %s, %s, %s order\n\n| round | executable | pass | loads | caches (own / shared) |\n|---|---|---|---|---|\n",
			r.Programs, probe.GiB(r.ProgramBytes), r.Format, r.Hold, r.Order)
		for i, rd := range r.Rounds {
			for j, pass := range rd.Passes {
				var cells []string
				for _, l := range pass {
					cells = append(cells, fmt.Sprintf("[%d] %s", l.Program, loadCell(r, l)))
				}
				fmt.Fprintf(sb, "| %d | %s | %d | %s | %.0f MiB / %.0f MiB |\n", i+1, rd.Exe, j+1, strings.Join(cells, " "),
					float64(rd.ExeCacheBytes)/(1<<20), float64(rd.SharedCacheBytes)/(1<<20))
			}
		}
	}
}
