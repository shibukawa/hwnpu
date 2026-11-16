package ai

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu/internal/chat"
)

// Role is what a use does, which decides the type that runs it.
type Role string

const (
	RoleTranscribe Role = "transcribe" // speech to text: Runtime.Transcriber
	RoleTranslate  Role = "translate"  // text to text in another language: Runtime.Translator
	RoleSpeech     Role = "speech"     // text to speech: Runtime.Speaker
	RoleVAD        Role = "vad"        // where the speech is: Runtime.VAD (no model)
)

// Registry is what an app uses: the uses, families or models of the definitions (models.yaml)
// it requires.
type Registry struct {
	names  []string
	memory int
}

// NewRegistry is an empty registry.
func NewRegistry() *Registry { return &Registry{} }

// Require adds uses (transcribe, translate, speech, vad), families (plamo3translate) or models
// (plamo-3-translate-2b) of the definitions. A use or a family picks its model for the machine's
// memory when the plan is made.
func (r *Registry) Require(names ...string) *Registry {
	for _, n := range names {
		if n = strings.TrimSpace(n); n != "" && !slices.Contains(r.names, n) {
			r.names = append(r.names, n)
		}
	}
	return r
}

// Memory makes uses and families pick their models, and the models their programs, for a machine
// of gb GB instead of this one (a cache to ship for smaller machines). 0 is this machine's, with
// the memory chosen for a model (hwnpu memory, modelstore.Store.SetMemory).
func (r *Registry) Memory(gb int) *Registry {
	r.memory = gb
	return r
}

// resolved is a name of the registry and what it means.
type resolved struct {
	use   string
	role  Role
	model chat.Model // none for RoleVAD
	why   string
}

// resolve finds what a name means for a machine of memory GB (0: this one, with the memory
// chosen for each model, hwnpu memory), as hwnpu's commands find it.
func resolve(name string, memory int) (resolved, error) {
	if strings.EqualFold(name, string(RoleVAD)) {
		return resolved{use: name, role: RoleVAD}, nil
	}
	var why strings.Builder
	m, ok, err := chat.LookupOn(name, memory, &why)
	if err != nil {
		return resolved{}, fmt.Errorf("ai: %w", err)
	}
	if !ok {
		return resolved{}, fmt.Errorf("ai: no model, family or use %q in the definitions (hwnpu-chat list shows them)", name)
	}
	role, err := roleOf(m)
	if err != nil {
		return resolved{}, err
	}
	return resolved{use: name, role: role, model: m, why: strings.TrimSpace(why.String())}, nil
}

// roleOf is the role of a model of the definitions.
func roleOf(m chat.Model) (Role, error) {
	switch {
	case m.Kind == chat.KindTranscribe:
		return RoleTranscribe, nil
	case m.Kind == chat.KindSpeech:
		return RoleSpeech, nil
	case m.Kind == chat.KindChat && slices.Contains(m.Features, "translation"):
		return RoleTranslate, nil
	}
	return "", fmt.Errorf("ai: %s is a %s model, which this version of the package does not run (transcribe, translate and speech do)", m.Name, m.Kind)
}

// Plan is what preparing the registry's uses in a store takes. Making it downloads nothing.
type Plan struct {
	Items []Item
	// NoWarmUp leaves out loading each model once at the end of its preparation. The first
	// load compiles the programs for the NPU, which takes from seconds to minutes (100 s for
	// Gemma 4 on the Neural Engine) and belongs to the machine: Run does it so that the app's
	// first use does not wait for it.
	NoWarmUp bool

	store Store
	npu   npu
}

// Item is a use of a plan.
type Item struct {
	Use   string
	Role  Role
	Model string // the model's name in the definitions, "" for a use without one (vad)
	Repo  string // where it is downloaded from: OWNER/NAME of Hugging Face, or SOURCE:NAME
	About string
	// Why says how a use or a family picked the model (for 16 GB machines).
	Why string
	// Ready is true when the store has it prepared, in Bundle (a directory of Store.Prepared)
	// when that is not "".
	Ready  bool
	Bundle string
	// Download is about how many bytes preparing it downloads: the model's size in the
	// definitions, 0 when its checkpoint is here (or it is Ready). It is the most: a shared
	// layer (Store.Models) may hold what was converted, and then the checkpoint is not
	// downloaded.
	Download int64
	// MemoryGB is what the model takes running, from the definitions (0: not known).
	MemoryGB float64

	model chat.Model
}

// License is what a model's terms ask an app to show or agree to (models.yaml's license).
type License struct {
	Model string
	Repo  string
	// ID names the license (apache-2.0, gemma); Class is what it allows of commercial use
	// (open, conditional, non-commercial); Terms says it in one line.
	ID, Class, Terms string
	// URL is where its text is: the license's, else the model's page.
	URL string
	// Accept is true when the license asks to be agreed to before the model is downloaded.
	Accept bool
}

// Plan resolves the registry's names for the machine (or Memory) and finds what store lacks. It
// sets the process's layers and reads the definitions as store.Models says (modelstore.Open).
func (r *Registry) Plan(ctx context.Context, store Store) (*Plan, error) {
	openMu.Lock()
	defer openMu.Unlock()
	if err := store.open(); err != nil {
		return nil, err
	}
	p := &Plan{store: store, npu: hereNPU(ctx)}
	vendor := chat.HereVendor()
	for _, name := range r.names {
		res, err := resolve(name, r.memory)
		if err != nil {
			return nil, err
		}
		it := Item{Use: res.use, Role: res.role, Why: res.why}
		if res.role == RoleVAD {
			it.Ready = true
			p.Items = append(p.Items, it)
			continue
		}
		m := res.model
		it.model, it.Model, it.Repo, it.About = m, m.Name, m.Source(), m.About
		it.MemoryGB = m.MemoryFor(vendor)
		if loc, err := store.find(name, p.npu.target); err == nil && loc.Model == m.Name && loc.Run == m.Run {
			it.Ready = true
			if loc.bundle {
				it.Bundle = loc.root
			}
		} else if err != nil && !errors.Is(err, ErrNotPrepared) && !errors.Is(err, ErrBundleTarget) {
			return nil, err
		}
		if !it.Ready && !chat.Downloaded(m) {
			it.Download = int64(m.BytesFor(vendor))
		}
		p.Items = append(p.Items, it)
	}
	return p, nil
}

// Ready reports whether every use is prepared: Run has nothing to do.
func (p *Plan) Ready() bool {
	for _, it := range p.Items {
		if !it.Ready {
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
// fit the machine together (three models of an interpreter take about 6.7 GB).
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
		if it.Model == "" || seen[it.Model] {
			continue
		}
		seen[it.Model] = true
		ml := it.model.License
		l := License{Model: it.Model, Repo: it.Repo, ID: ml.ID, Class: ml.Class, Terms: ml.Terms, URL: ml.URL, Accept: ml.Accept}
		if l.URL == "" && it.model.Repo != "" {
			l.URL = "https://huggingface.co/" + it.model.Repo
		}
		out = append(out, l)
	}
	return out
}

// Stage is a step of preparing a use.
type Stage string

const (
	StagePrepare Stage = "prepare" // downloading and converting
	StageWarmUp  Stage = "warmup"  // loading once: compiling the programs for the NPU
	StageDone    Stage = "done"
)

// Progress is what Run reports as it goes.
type Progress struct {
	Use, Model  string
	Stage       Stage
	Item, Items int // the use's place in the plan, from 1, of how many Run prepares
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

// Run prepares the uses that are not ready, in order, as hwnpu-chat and hwnpu-audio do on their
// first run: it takes what is missing from the shared layers or downloads it (checked against the
// definitions' sha256 where they have one), converts the programs unless a layer has them, puts
// what it downloaded and made into the layers that take it (Store.Models), loads each once
// (unless NoWarmUp) and writes it into the store. fn (nil for none) is called in Run's goroutine.
// A Run cut off (ctx, a network error) goes on from where it stopped when run again.
func (p *Plan) Run(ctx context.Context, fn func(Progress)) error {
	if fn == nil {
		fn = func(Progress) {}
	}
	var todo []int
	for i, it := range p.Items {
		if !it.Ready {
			todo = append(todo, i)
		}
	}
	for n, i := range todo {
		it := &p.Items[i]
		report := func(stage Stage, msg string) {
			fn(Progress{Use: it.Use, Model: it.Model, Stage: stage, Item: n + 1, Items: len(todo), Message: msg})
		}
		log := &lineWriter{fn: func(line string) { report(StagePrepare, line) }}
		report(StagePrepare, "")
		env := prepareEnv{ctx: ctx, model: it.model, dir: chat.ModelDir(it.model), npu: p.npu, log: log}
		r, err := roles[it.Role].prepare(env)
		if err != nil {
			return fmt.Errorf("ai: preparing %s (%s): %w", it.Use, it.Model, err)
		}
		r.Model, r.Role, r.Dir, r.Run, r.MemoryGB, r.Prepared = it.Model, it.Role, env.dir, it.model.Run, it.MemoryGB, time.Now().UTC().Truncate(time.Second)
		if !p.NoWarmUp && p.npu.target != "" {
			report(StageWarmUp, "")
			log.fn = func(line string) { report(StageWarmUp, line) }
			l, err := roles[it.Role].load(ctx, located{record: r, root: p.store.dir()}, loadEnv{cache: p.store.cacheDir(), device: p.npu.device, log: log})
			if err != nil {
				return fmt.Errorf("ai: loading %s (%s) once: %w", it.Use, it.Model, err)
			}
			if err := l.Close(); err != nil {
				return err
			}
		}
		if err := p.store.put(ctx, it.Use, p.npu.target, r); err != nil {
			return err
		}
		it.Ready, it.Download = true, 0
		report(StageDone, "")
	}
	return nil
}

// Prepare is NewRegistry().Require(names...), its plan in store, and Run: the uses prepared in
// one call, for a command line tool. Apps that show what is downloaded first call Plan and Run.
func Prepare(ctx context.Context, store Store, fn func(Progress), names ...string) error {
	p, err := NewRegistry().Require(names...).Plan(ctx, store)
	if err != nil {
		return err
	}
	return p.Run(ctx, fn)
}

// prepareEnv is what a role's prepare is given.
type prepareEnv struct {
	ctx   context.Context
	model chat.Model
	dir   string // the model's directory of programs (chat.ModelDir)
	npu   npu    // what the programs are converted for
	log   io.Writer
}

// loadEnv is what a role's load is given.
type loadEnv struct {
	cache  string // the compile cache
	device string // the device backend (npu.device)
	log    io.Writer
}

// role prepares and loads the models of a role.
type role struct {
	prepare func(prepareEnv) (*record, error)
	load    func(ctx context.Context, l located, env loadEnv) (io.Closer, error)
}

// roles are the roles' preparing and loading, by role; a variable so that tests replace them.
var roles = map[Role]role{
	RoleTranscribe: {prepareTranscribe, loadTranscribe},
	RoleSpeech:     {prepareSpeech, loadSpeech},
	RoleTranslate:  {prepareTranslate, loadTranslate},
}
