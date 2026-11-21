package ai

import (
	"bytes"
	"cmp"
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
	shared "github.com/shibukawa/hwnpu/internal/store"
)

// Plan is what making an app's models ready takes. Making it downloads nothing.
type Plan struct {
	Items []Item
	// NoCompile leaves out loading each model once at the end of its preparation. The first
	// load compiles the programs for the NPU, which takes from seconds to minutes (100 s for
	// Gemma 4 on the Neural Engine) and belongs to the machine: Run does it so that the app's
	// first use does not wait for it. Without it Run only downloads and converts, which needs
	// no NPU (a build machine, or an NPU another program is using).
	NoCompile bool
	// AcceptLicenses accepts the licenses that ask for it (Licenses' Accept) as Run comes to
	// their models, for an app that has shown them and has the user's agreement. Without it Run
	// fails with a *LicenseError at a model whose license is not accepted yet.
	AcceptLicenses bool

	store   Store
	npu     airole.NPU
	machine int
}

// Item is a use of a plan.
type Item struct {
	// Use is the name the model is picked by and Role the role it fills.
	Use  string `json:"use"`
	Role string `json:"role"`
	// Model is the model's name in the definitions and Repo where it is downloaded from:
	// OWNER/NAME of Hugging Face.
	Model string `json:"model"`
	Repo  string `json:"repo"`
	About string `json:"about,omitempty"`
	// Why says how the model was picked: by the app for the machine's memory (Use.At, Use.As),
	// by a use or a family of the definitions (for 16 GB machines).
	Why string `json:"why,omitempty"`
	// Ready is true when the store has the model's files and programs: it loads without the
	// network. Bundle is the directory of Store.Prepared they are in, "" for this machine's
	// cache.
	Ready  bool   `json:"ready"`
	Bundle string `json:"bundle,omitempty"`
	// Compiled is true when the model has been loaded once on this machine's NPU, so that its
	// programs are compiled for it.
	Compiled bool `json:"compiled"`
	// Download is about how many bytes preparing it downloads: the model's size in the
	// definitions, 0 when its checkpoint is here (or it is Ready). It is the most: a shared
	// layer (Store.Models) may hold what was converted, and then the checkpoint is not
	// downloaded.
	Download int64 `json:"download_bytes"`
	// MemoryGB is what the model takes running, from the definitions (0: not known).
	MemoryGB float64 `json:"memory_gb"`

	model chat.Model
	req   requirement
}

// License is what a model's terms ask an app to show or agree to (models.yaml's license).
type License struct {
	Model string `json:"model"`
	Repo  string `json:"repo"`
	// ID names the license (apache-2.0, gemma); Class is what it allows of commercial use
	// (open, conditional, non-commercial); Terms says it in one line.
	ID    string `json:"id"`
	Class string `json:"class"`
	Terms string `json:"terms,omitempty"`
	// URL is where its text is: the license's, else the model's page.
	URL string `json:"url"`
	// Accept is true when the license asks to be agreed to before the model is downloaded;
	// Accepted when it has been (on this machine, or by HWNPU_ACCEPT_LICENSE).
	Accept   bool `json:"accept"`
	Accepted bool `json:"accepted"`
}

// LicenseError is Run's error at a model whose license asks to be accepted and is not.
type LicenseError struct{ License License }

func (e *LicenseError) Error() string {
	l := e.License
	terms := ""
	if l.Terms != "" {
		terms = ": " + l.Terms
	}
	return fmt.Sprintf("ai: %s is under the %s license (%s)%s, which asks to be accepted before the model is downloaded (Plan.AcceptLicenses, or %s=%s)",
		l.Model, l.ID, l.URL, terms, chat.LicenseEnv, l.ID)
}

// resolve finds the model a requirement means for a machine of memory GB (0: this one, with the
// memory chosen for each model, hwnpu memory), as hwnpu's commands find it: by the name of the
// requirement's tier for that memory (Use.At), picked as a machine of its own memory would when
// it has one (Use.As).
func resolve(r requirement, memory int) (m chat.Model, why string, err error) {
	name, tier := r.pick(cmp.Or(memory, chat.MachineMemory()))
	var log strings.Builder
	m, ok, err := chat.LookupOn(name, cmp.Or(r.as, memory), &log)
	if err != nil {
		return m, "", fmt.Errorf("ai: %w", err)
	}
	if !ok {
		return m, "", fmt.Errorf("ai: no model, family or use %q in the definitions (hwnpu-chat list shows them)", name)
	}
	if err := r.role.Accepts(m); err != nil {
		return m, "", err
	}
	var whys []string
	if tier > 0 {
		whys = append(whys, fmt.Sprintf("%s on machines of %d GB and up, by the app", name, tier))
	}
	if r.as > 0 {
		whys = append(whys, fmt.Sprintf("picked as on a machine of %d GB, by the app", r.as))
	}
	if l := strings.TrimSpace(log.String()); l != "" {
		whys = append(whys, l)
	}
	return m, strings.Join(whys, "; "), nil
}

// Plan resolves the app's uses for the machine (or Memory) and finds what the store lacks. It
// sets the process's layers and reads the definitions as the app's Store says (modelstore.Open).
func (a *App) Plan(ctx context.Context) (*Plan, error) {
	return a.plan(ctx, a.reqs)
}

func (a *App) plan(ctx context.Context, reqs []requirement) (*Plan, error) {
	openMu.Lock()
	defer openMu.Unlock()
	if err := a.Store.open(); err != nil {
		return nil, err
	}
	p := &Plan{store: a.Store, npu: airole.Here(ctx), machine: cmp.Or(a.memory, chat.MachineMemory())}
	for _, r := range reqs {
		m, why, err := resolve(r, a.memory)
		if err != nil {
			return nil, err
		}
		it := Item{Use: r.name, Role: r.role.Name, Why: why, model: m, req: r,
			Model: m.Name, Repo: m.Source(), About: m.About, MemoryGB: m.MemoryFor(p.npu.Vendor)}
		loc, err := a.Store.find(r.key(), p.npu, m)
		switch {
		case err == nil:
			// a prepared cache is read only: what it holds counts as compiled (the first load
			// compiles it, and nothing can be written there to say so)
			it.Ready, it.Compiled = true, !loc.Compiled.IsZero() || loc.bundle
			if loc.bundle {
				it.Bundle = loc.root
			}
		case err != nil && !errors.Is(err, ErrNotPrepared) && !errors.Is(err, ErrBundleTarget):
			return nil, err
		}
		if !it.Ready && !chat.Downloaded(m) {
			it.Download = int64(m.BytesFor(p.npu.Vendor))
		}
		p.Items = append(p.Items, it)
	}
	return p, nil
}

// Target names the NPU the plan prepares for, as the shared layers name it: coreai, intel or
// amd-xdna2; "" when no NPU runs the models here (Run then downloads and converts only).
func (p *Plan) Target() string { return p.npu.Target }

// MachineGB is the memory of the machine the plan is for, in GB: this one's (HWNPU_MEMORY, else
// what it has), or the app's Memory. The uses picked their models for it, and MemoryGB is to fit
// in it beside everything else the machine runs.
func (p *Plan) MachineGB() int { return p.machine }

// usable is about how much of a machine's memory is left for models, in GB, by the memory it
// has: MODEL.md's second row (the OS and the everyday applications taken off). The NPU's buffers
// are pinned memory that cannot be swapped out, so models over it push other applications to
// swap.
var usable = [][2]float64{{8, 3}, {16, 9}, {24, 15}, {32, 22}, {48, 35}, {64, 50}}

// usableGB is usable for a machine of machine GB: between the table's machines in proportion,
// below 8 GB what is over 5 GB, above 64 GB all but 14 GB.
func usableGB(machine int) float64 {
	m := float64(machine)
	first, last := usable[0], usable[len(usable)-1]
	switch {
	case m <= first[0]:
		return max(0, m-(first[0]-first[1]))
	case m >= last[0]:
		return m - (last[0] - last[1])
	}
	for i := 1; ; i++ {
		if a, b := usable[i-1], usable[i]; m <= b[0] {
			return a[1] + (m-a[0])*(b[1]-a[1])/(b[0]-a[0])
		}
	}
}

// UsableGB is about how much of the plan's machine's memory (MachineGB) is left for models,
// beside the OS and the everyday applications (MODEL.md: about 3 GB of 8, 9 of 16, 15 of 24, 22
// of 32, 35 of 48, 50 of 64). MemoryGB is to stay under it: the NPU's buffers are pinned memory
// that cannot be swapped out, and models over it push the other applications to swap.
func (p *Plan) UsableGB() float64 { return usableGB(p.machine) }

// Fits reports whether the plan's models running together (MemoryGB) stay within what the
// machine has for models (UsableGB). An app whose plan does not fit picks smaller models for
// that memory (Use.At, Use.As) or runs its models one after another (Use.Release).
func (p *Plan) Fits() bool { return p.MemoryGB() <= p.UsableGB() }

// Ready reports whether every use is ready to load: Run downloads nothing.
func (p *Plan) Ready() bool {
	for _, it := range p.Items {
		if !it.Ready {
			return false
		}
	}
	return true
}

// Compiled reports whether every use is ready and compiled for this machine's NPU: Run has
// nothing to do.
func (p *Plan) Compiled() bool {
	for _, it := range p.Items {
		if !it.Ready || !it.Compiled {
			return false
		}
	}
	return true
}

// DownloadBytes is about how many bytes Run downloads.
func (p *Plan) DownloadBytes() int64 {
	var n int64
	for _, it := range p.Items {
		n += it.Download
	}
	return n
}

// MemoryGB is what the plan's models take running at once, from the definitions: whether they
// fit the machine together (the three models of an interpreter take about 6.7 GB). An app that
// runs them one after another (Use.Release) needs the largest alone.
func (p *Plan) MemoryGB() float64 {
	var n float64
	for _, it := range p.Items {
		n += it.MemoryGB
	}
	return n
}

// Licenses are the models' terms to show before downloading them, one per model.
func (p *Plan) Licenses() []License {
	var out []License
	seen := map[string]bool{}
	for _, it := range p.Items {
		if seen[it.Model] {
			continue
		}
		seen[it.Model] = true
		out = append(out, license(it.model))
	}
	return out
}

func license(m chat.Model) License {
	ml := m.License
	l := License{Model: m.Name, Repo: m.Source(), ID: ml.ID, Class: ml.Class, Terms: ml.Terms, URL: ml.URL, Accept: ml.Accept}
	if l.URL == "" && m.Repo != "" {
		l.URL = "https://huggingface.co/" + m.Repo
	}
	l.Accepted = !l.Accept || chat.AcceptLicense(m, false, false, nil, io.Discard) == nil
	return l
}

// Stage is a step of preparing a use.
type Stage string

const (
	StagePrepare Stage = "prepare" // downloading and converting
	StageCompile Stage = "compile" // loading once: compiling the programs for the NPU
	StageDone    Stage = "done"
)

// Progress is what Run reports as it goes.
type Progress struct {
	Use, Role, Model string
	Stage            Stage
	Item, Items      int // the use's place among those Run works on, from 1
	// Message is a line of what is being done (a download, a conversion), for a log; "" when
	// the stage begins or ends.
	Message string
}

// lineWriter calls fn with each line written to it, on \n or \r (a progress bar redrawn).
type lineWriter struct {
	mu   sync.Mutex
	buf  []byte
	last string
	fn   func(string)
}

func (w *lineWriter) Write(b []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.buf = append(w.buf, b...)
	for {
		i := bytes.IndexAny(w.buf, "\r\n")
		if i < 0 {
			return len(b), nil
		}
		line := strings.TrimSpace(stripANSI(string(w.buf[:i])))
		w.buf = w.buf[i+1:]
		if line != "" && line != w.last {
			w.last = line
			w.fn(line)
		}
	}
}

// stripANSI removes the escape sequences of colors and cursor moves.
func stripANSI(s string) string {
	if !strings.Contains(s, "\x1b") {
		return s
	}
	var sb strings.Builder
	for i := 0; i < len(s); i++ {
		if s[i] != 0x1b {
			sb.WriteByte(s[i])
			continue
		}
		i++
		if i < len(s) && s[i] == '[' {
			for i++; i < len(s) && (s[i] < 0x40 || s[i] > 0x7e); i++ {
			}
		}
	}
	return sb.String()
}

// Run makes the uses ready, in order, as hwnpu's commands do on their first run. For each that
// is not ready it takes what is missing from the shared layers or downloads it (checked against
// the definitions' sha256 where they have one), converts the programs unless a layer has them,
// and puts what it downloaded and made into the layers that take it (Store.Models). Then, unless
// NoCompile, it loads each use that was not loaded on this machine before once, which compiles
// its programs for the NPU, and writes it into the store. fn (nil for none) is called in Run's
// goroutine. A Run cut off (ctx, a network error) goes on from where it stopped when run again.
func (p *Plan) Run(ctx context.Context, fn func(Progress)) error {
	if fn == nil {
		fn = func(Progress) {}
	}
	compile := !p.NoCompile && p.npu.Target != ""
	var todo []int
	for i, it := range p.Items {
		if !it.Ready || compile && !it.Compiled {
			todo = append(todo, i)
		}
	}
	for n, i := range todo {
		it := &p.Items[i]
		report := func(stage Stage, msg string) {
			fn(Progress{Use: it.Use, Role: it.Role, Model: it.Model, Stage: stage, Item: n + 1, Items: len(todo), Message: msg})
		}
		log := &lineWriter{fn: func(line string) { report(StagePrepare, line) }}
		var r *record
		if it.Ready {
			loc, err := p.store.find(it.req.key(), p.npu, it.model)
			if err != nil {
				return err
			}
			r = loc.record
		} else {
			if l := license(it.model); !l.Accepted {
				if !p.AcceptLicenses {
					return &LicenseError{License: l}
				}
				if err := chat.AcceptLicense(it.model, true, false, nil, log); err != nil {
					return fmt.Errorf("ai: %w", err)
				}
			}
			report(StagePrepare, "")
			dir := chat.ModelDir(it.model)
			files, err := it.req.role.Prepare(airole.PrepareEnv{Ctx: chat.WithModel(ctx, it.model), Model: it.model, Dir: dir, NPU: p.npu, Log: log})
			if err != nil {
				return fmt.Errorf("ai: preparing %s (%s): %w", it.Use, it.Model, err)
			}
			r = &record{Model: it.Model, Role: it.Role, Run: it.model.Run, MemoryGB: it.MemoryGB, Files: files,
				Prepared: time.Now().UTC().Truncate(time.Second)}
			// written before the load, so that a compile cut off does not download again
			if err := p.store.put(ctx, it.req.key(), p.npu, r); err != nil {
				return err
			}
			it.Ready, it.Download = true, 0
		}
		if compile && !it.Compiled {
			report(StageCompile, "")
			log.fn = func(line string) { report(StageCompile, line) }
			l, err := it.req.role.Load(ctx, r.Files, airole.LoadEnv{Model: it.Model, NPU: p.npu, Cache: p.store.cacheDir(), Log: log})
			if err != nil {
				return fmt.Errorf("ai: loading %s (%s) once: %w", it.Use, it.Model, err)
			}
			if err := l.Close(); err != nil {
				return err
			}
			r.Compiled = time.Now().UTC().Truncate(time.Second)
			if err := p.store.put(ctx, it.req.key(), p.npu, r); err != nil {
				return err
			}
			it.Compiled = true
		}
		report(StageDone, "")
	}
	// what was made here is on its way to the sources written back to: wait for it, so that a
	// process that ends after Run does not cut it off (ctx's end leaves the rest for the next run)
	shared.Flush(ctx)
	return ctx.Err()
}

// Prepare is the app's Plan and its Run: everything made ready in one call, for a command line
// tool. Apps that show what is downloaded first call Plan and Run.
func (a *App) Prepare(ctx context.Context, fn func(Progress)) error {
	p, err := a.Plan(ctx)
	if err != nil {
		return err
	}
	return p.Run(ctx, fn)
}
