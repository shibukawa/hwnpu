package ai

import (
	"context"
	"errors"
	"fmt"
	"io"
	"maps"
	"path"
	"slices"
	"strings"

	"github.com/shibukawa/hwnpu/internal/aieng"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/airpc"
	"github.com/shibukawa/hwnpu/internal/catalog"
	"github.com/shibukawa/hwnpu/internal/daemon"
)

// Catalog is what the definitions name (models.yaml, the machine's, the user's, the app's
// files and Definitions): the models, the uses that pick one by the machine's memory, and the
// aliases. It is what a command's --model is checked and made a model's name with (Normalize),
// and what its list shows.
type Catalog struct {
	Models []ModelInfo
	Uses   []UseInfo
	// Aliases run the model of the value wherever the key is named.
	Aliases map[string]string
	// Files are the definition files read, the built-in first.
	Files []string

	machine int
	vendor  string
	// remote is hwnpud, whose definitions these are (Find asks it); store the Store asked
	remote *daemon.Client
	store  StoreSpec
}

// ModelInfo is a model of the definitions.
type ModelInfo struct {
	Name string `json:"name"`
	// Kind is what it is: chat, embedding, transcribe, speech, decision, image, depth,
	// face-detect, pose, detect, segment, ocr, label, pitch, notes, music, sfx.
	Kind string `json:"kind"`
	// Family and Tag name it as FAMILY:TAG; Machine is the least memory of a machine (GB) the
	// family picks it for.
	Family  string `json:"family,omitempty"`
	Tag     string `json:"tag,omitempty"`
	Machine int    `json:"machine,omitempty"`
	// Repo is where it is downloaded from: OWNER/NAME, or OWNER/NAME@REVISION when pinned.
	Repo     string   `json:"repo"`
	About    string   `json:"about,omitempty"`
	Features []string `json:"features,omitempty"`
	// MemoryGB is what it takes running and DownloadBytes about what preparing it downloads, for
	// this machine's NPU (0: not known).
	MemoryGB      float64 `json:"memory_gb,omitempty"`
	DownloadBytes int64   `json:"download_bytes,omitempty"`
	// Device is where it runs on this machine's NPU: npu or cpu.
	Device  string  `json:"device"`
	License License `json:"license"`
	// Downloaded reports whether its checkpoint or its converted programs are here.
	Downloaded bool `json:"downloaded"`
	// Develop marks a model in development; Defined one the app made (Definition).
	Develop bool `json:"develop,omitempty"`
	Defined bool `json:"defined,omitempty"`
}

// UseInfo is a use of the definitions (translate, transcribe, speech): a name that picks a model
// by the machine's memory.
type UseInfo struct {
	Name    string   `json:"name"`
	Aliases []string `json:"aliases,omitempty"`
	About   string   `json:"about,omitempty"`
	// Picks are its models by the least memory of a machine, smallest first.
	Picks []Pick `json:"picks"`
	// Model is what it picks on this machine ("" for none).
	Model string `json:"model,omitempty"`
}

// Pick is a use's model on machines of at least MachineGB GB.
type Pick struct {
	MachineGB int    `json:"machine_gb"`
	Model     string `json:"model"`
}

// Catalog reads the definitions as the Store says (its Models' files and its Definitions) and
// lists them: hwnpud's, through it (HWNPU_DAEMON), else this process's, setting its layers and
// definitions as Plan does.
func (s Store) Catalog() (*Catalog, error) {
	if !s.local {
		ctx := context.Background()
		dc, err := hwnpud(ctx)
		if err != nil {
			return nil, err
		}
		if dc != nil {
			return s.catalogRemote(ctx, dc)
		}
	}
	return s.catalog()
}

// catalog is Catalog of this process's definitions.
func (s Store) catalog() (*Catalog, error) {
	openMu.Lock()
	defer openMu.Unlock()
	e, err := aieng.Get()
	if err != nil {
		return nil, fmt.Errorf("ai: %w", err)
	}
	if err := s.open(); err != nil {
		return nil, err
	}
	models, uses, aliases, files := e.Table()
	c := &Catalog{Aliases: maps.Clone(aliases), Files: slices.Clone(files), machine: e.MachineMemory(), vendor: e.HereVendor()}
	defined := map[string]bool{}
	for _, d := range s.Definitions {
		defined[d.ID()] = true
	}
	for _, m := range models {
		c.Models = append(c.Models, c.info(m, defined[m.Name]))
	}
	for _, cat := range uses {
		u := UseInfo{Name: cat.Name, Aliases: slices.Clone(cat.Aliases), About: cat.About}
		for _, p := range cat.Picks {
			u.Picks = append(u.Picks, Pick{MachineGB: p.Machine, Model: p.Model})
		}
		slices.SortFunc(u.Picks, func(a, b Pick) int { return a.MachineGB - b.MachineGB })
		if m, _, ok, err := e.Pick(cat.Name, c.machine); ok && err == nil {
			u.Model = m.Name
		}
		c.Uses = append(c.Uses, u)
	}
	return c, nil
}

func (c *Catalog) info(m catalog.Model, defined bool) ModelInfo {
	return ModelInfo{Name: m.Name, Kind: string(m.Kind), Family: m.Family, Tag: m.Tag, Machine: m.MachineFor(c.vendor),
		Repo: m.Source(), About: m.About, Features: slices.Clone(m.Features), MemoryGB: m.MemoryFor(c.vendor),
		DownloadBytes: int64(m.BytesFor(c.vendor)), Device: m.DeviceFor(c.vendor), License: license(m),
		Downloaded: aieng.Must().Downloaded(m), Develop: m.Develop, Defined: defined}
}

// OfKind are the models of the kinds named (all of them when none is).
func (c *Catalog) OfKind(kinds ...string) []ModelInfo {
	var out []ModelInfo
	for _, m := range c.Models {
		if len(kinds) == 0 || slices.Contains(kinds, m.Kind) {
			out = append(out, m)
		}
	}
	return out
}

// Model is the model of the definitions named name (or its repository), ignoring case.
func (c *Catalog) Model(name string) (ModelInfo, bool) {
	i := slices.IndexFunc(c.Models, func(m ModelInfo) bool {
		repo, _, _ := strings.Cut(m.Repo, "@")
		return strings.EqualFold(m.Name, name) || strings.EqualFold(repo, name)
	})
	if i < 0 {
		return ModelInfo{}, false
	}
	return c.Models[i], true
}

// NoModelError is the error of a name that the definitions have no model, family or use of for
// a role. Models are those the role runs, for a list to show.
type NoModelError struct {
	Name, Role string
	Models     []ModelInfo
}

func (e *NoModelError) Error() string {
	return fmt.Sprintf("ai: no %s model, family or use %q in the definitions", e.Role, e.Name)
}

// Find is the model name means for u's role on this machine, as hwnpu's commands find it: a
// model by its name or repository, or the one a family (with a tag, or for the machine's
// memory) or a use picks; through the aliases. A name the definitions do not have is a
// *NoModelError; a model the role does not run is an error that says so.
func (c *Catalog) Find(name string, u Requirement) (ModelInfo, error) {
	r := u.requirement()
	if c.remote != nil {
		return c.findRemote(context.Background(), name, r.role)
	}
	return c.find(name, r.role)
}

// find is Find in this process's definitions.
func (c *Catalog) find(name string, role *airole.Role) (ModelInfo, error) {
	m, ok, err := aieng.Must().LookupOn(strings.TrimSpace(name), 0, io.Discard)
	if err != nil {
		return ModelInfo{}, fmt.Errorf("ai: %w", err)
	}
	if !ok {
		return ModelInfo{}, c.noModel(name, role)
	}
	if err := role.Accepts(m); err != nil {
		return ModelInfo{}, err
	}
	info, _ := c.Model(m.Name)
	if info.Name == "" {
		info = c.info(m, false)
	}
	return info, nil
}

func (c *Catalog) noModel(name string, role *airole.Role) error {
	e := &NoModelError{Name: name, Role: role.Name}
	for _, m := range c.Models {
		if slices.Contains(role.Kinds, catalog.Kind(m.Kind)) {
			e.Models = append(e.Models, m)
		}
	}
	return e
}

// Normalize is a command's --model made a use of the model it means (Find): the use of the
// model's name, run as on a machine of the memory a tag of memory names (lfm3:8g). A Hugging Face
// repository the definitions do not have (OWNER/NAME, OWNER/NAME@REVISION) is a model of the
// role's kind from there: def is its Definition, which the app defines (App.Define) before it
// plans or opens the use; nil otherwise. The use keeps u's At and As.
func Normalize[T io.Closer](c *Catalog, u Use[T]) (Use[T], *Definition, error) {
	name := u.Name()
	m, err := c.Find(name, u)
	var none *NoModelError
	if errors.As(err, &none) {
		repo, _, _ := strings.Cut(name, "@")
		if strings.Count(repo, "/") != 1 || strings.HasPrefix(repo, "/") || strings.HasSuffix(repo, "/") || strings.Contains(repo, ":") {
			return u, nil, err
		}
		d := &Definition{Name: strings.ReplaceAll(path.Base(name), "@", "-"), Kind: string(u.req.role.Kinds[0]), From: "hf:" + name,
			About: "the Hugging Face repository " + name}
		u.req.name = d.ID()
		return u, d, nil
	}
	if err != nil {
		return u, nil, err
	}
	if _, tag, ok := strings.Cut(name, ":"); ok {
		if gb, ok := catalog.MemoryTag(tag); ok && u.req.as == 0 {
			u = u.As(gb)
		}
	}
	u.req.name = m.Name
	return u, nil, nil
}

// Kinds are the kinds of the definitions' models the use's role runs (Catalog.OfKind).
func (u Use[T]) Kinds() []string {
	var out []string
	for _, k := range u.req.role.Kinds {
		out = append(out, string(k))
	}
	return out
}

// Spec is the catalog as hwnpud sends it.
func (c *Catalog) Spec() CatalogSpec {
	return CatalogSpec{Models: c.Models, Uses: c.Uses, Aliases: c.Aliases, Files: c.Files, Machine: c.machine, Vendor: c.vendor}
}

// catalogRemote is Catalog of hwnpud's definitions.
func (s Store) catalogRemote(ctx context.Context, dc *daemon.Client) (*Catalog, error) {
	st, err := s.spec()
	if err != nil {
		return nil, fmt.Errorf("ai: the store's %w", err)
	}
	sess, host, err := hostSession(ctx, dc)
	if err != nil {
		return nil, err
	}
	defer sess.Close()
	cs, err := airpc.Call1[CatalogSpec](host, ctx, "Catalog", st)
	if err != nil {
		return nil, fmt.Errorf("ai: hwnpud's catalog: %w", olderHwnpud(dc, err))
	}
	return &Catalog{Models: cs.Models, Uses: cs.Uses, Aliases: cs.Aliases, Files: cs.Files, machine: cs.Machine, vendor: cs.Vendor,
		remote: dc, store: st}, nil
}

// findRemote is Find in hwnpud's definitions.
func (c *Catalog) findRemote(ctx context.Context, name string, role *airole.Role) (ModelInfo, error) {
	sess, host, err := hostSession(ctx, c.remote)
	if err != nil {
		return ModelInfo{}, err
	}
	defer sess.Close()
	m, err := airpc.Call1[ModelInfo](host, ctx, "Find", c.store, name, role.Name)
	return m, olderHwnpud(c.remote, err)
}

// olderHwnpud is err, told as what to do when hwnpud is of a build before the method (its Host
// has no method of the call).
func olderHwnpud(dc *daemon.Client, err error) error {
	if err == nil || !strings.Contains(err.Error(), "has no method") {
		return err
	}
	return fmt.Errorf("%w: hwnpud (build %s) is older than this client (go install ./cmd/hwnpud && hwnpu service restart)", err, dc.Server.Build)
}

// HostCatalog is Store.Catalog of the store a client sent, in hwnpud's definitions (Host.Catalog).
func HostCatalog(st StoreSpec) (CatalogSpec, error) {
	c, err := st.store().catalog()
	if err != nil {
		return CatalogSpec{}, err
	}
	return c.Spec(), nil
}

// HostFind is Catalog.Find of the store a client sent, for the role named role (Host.Find).
func HostFind(st StoreSpec, name, role string) (ModelInfo, error) {
	r := airole.Lookup(role)
	if r == nil {
		return ModelInfo{}, fmt.Errorf("ai: hwnpud knows no role %q: hwnpud is of an older build than the app (go install ./cmd/hwnpud, then hwnpu service restart)", role)
	}
	c, err := st.store().catalog()
	if err != nil {
		return ModelInfo{}, err
	}
	return c.find(name, r)
}
