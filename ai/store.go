package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm"
)

// Store is where prepared models are kept.
type Store struct {
	// Dir is where Plan.Run downloads and converts the models, each in a directory of its
	// own, and writes what it prepared (StoreFile).
	Dir string
	// Bundles are prepared stores shipped with the app, read only: directories with a
	// StoreFile whose paths are relative to them. They are looked in before Dir.
	Bundles []string
	// Cache is the compile cache of the NPU's programs ("": hwnpu in the user cache
	// directory). Apple's compiled programs belong to the machine and its OS, so they are not
	// bundled.
	Cache string
}

// DefaultStore is the store of hwnpu's commands (hwnpu-chat, hwnpu-audio): what they downloaded
// and converted is used as it is.
func DefaultStore() Store { return Store{Dir: chat.ModelsDir()} }

func (s Store) cacheDir() string {
	if s.Cache != "" {
		return s.Cache
	}
	if d, err := os.UserCacheDir(); err == nil {
		return filepath.Join(d, "hwnpu")
	}
	return ""
}

// StoreFile is the file in a store that says what each use was prepared as.
const StoreFile = "store.json"

// storeFormat is the version of StoreFile's format this package writes and reads.
const storeFormat = 1

var (
	// ErrNotPrepared is the error of a use the store has not prepared: Plan.Run prepares it.
	ErrNotPrepared = errors.New("ai: not prepared")
	// ErrBundleTarget is the error of a store prepared for another NPU (a bundle for Apple's
	// Neural Engine on a machine with AMD's NPU), or of a format this package does not read.
	ErrBundleTarget = errors.New("ai: prepared for another target")
)

// manifest is StoreFile.
type manifest struct {
	Format int `json:"format"`
	// Target is the NPU the programs are for (hereTarget): apple, intel, amd-npu2.
	Target string             `json:"target"`
	Uses   map[string]*record `json:"uses"`
}

// record is what a use was prepared as. Its paths are relative to the store's directory, or
// absolute for files outside it (an export of AMD's NPU in its own directory).
type record struct {
	Model    string    `json:"model"`
	Role     Role      `json:"role"`
	Dir      string    `json:"dir"`
	MemoryGB float64   `json:"memory_gb,omitempty"`
	Prepared time.Time `json:"prepared"`

	Transcribe *transcribeRecord `json:"transcribe,omitempty"`
	Speech     *speechRecord     `json:"speech,omitempty"`
	Chat       *chatRecord       `json:"chat,omitempty"`
}

type transcribeRecord struct {
	Dir, Encoder, Cross, Decoder, Weights string
}

type speechRecord struct {
	Text         []string
	Depth, Codec string
	Dir, Weights string
	SampleRate   int `json:",omitempty"`
}

type chatRecord struct {
	Packages                        []string
	Tiers                           [][]string `json:",omitempty"`
	MTP                             []string   `json:",omitempty"`
	ModelConfig, Weights, Tokenizer string
	IRON                            string `json:",omitempty"`
	Name, Backend                   string
}

// paths are pointers to every path of the record, to make them relative or absolute.
func (r *record) paths() []*string {
	ps := []*string{&r.Dir}
	if t := r.Transcribe; t != nil {
		ps = append(ps, &t.Dir, &t.Encoder, &t.Cross, &t.Decoder, &t.Weights)
	}
	if s := r.Speech; s != nil {
		ps = append(ps, &s.Depth, &s.Codec, &s.Dir, &s.Weights)
		for i := range s.Text {
			ps = append(ps, &s.Text[i])
		}
	}
	if c := r.Chat; c != nil {
		ps = append(ps, &c.ModelConfig, &c.Weights, &c.Tokenizer, &c.IRON)
		for i := range c.Packages {
			ps = append(ps, &c.Packages[i])
		}
		for i := range c.MTP {
			ps = append(ps, &c.MTP[i])
		}
		for _, t := range c.Tiers {
			for i := range t {
				ps = append(ps, &t[i])
			}
		}
	}
	return ps
}

// relative makes the record's paths under root relative to it.
func (r *record) relative(root string) {
	for _, p := range r.paths() {
		if *p == "" || !filepath.IsAbs(*p) {
			continue
		}
		if rel, err := filepath.Rel(root, *p); err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			*p = filepath.ToSlash(rel)
		}
	}
}

// absolute makes the record's relative paths absolute under root.
func (r *record) absolute(root string) {
	for _, p := range r.paths() {
		if *p != "" && !filepath.IsAbs(*p) {
			*p = filepath.Join(root, filepath.FromSlash(*p))
		}
	}
}

// missing is the first of the record's files (absolute) that is not there, "" when all are.
func (r *record) missing() string {
	for _, p := range r.paths() {
		if *p == "" {
			continue
		}
		if _, err := os.Stat(*p); err != nil {
			return *p
		}
	}
	return ""
}

// clone is a deep copy of the record.
func (r *record) clone() *record {
	raw, _ := json.Marshal(r)
	var c record
	_ = json.Unmarshal(raw, &c)
	return &c
}

// readManifest reads the StoreFile of root; an empty one when there is none.
func readManifest(root string) (*manifest, error) {
	m := &manifest{Format: storeFormat, Uses: map[string]*record{}}
	raw, err := os.ReadFile(filepath.Join(root, StoreFile))
	if errors.Is(err, os.ErrNotExist) {
		return m, nil
	}
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(raw, m); err != nil {
		return nil, fmt.Errorf("ai: %s: %w", filepath.Join(root, StoreFile), err)
	}
	if m.Format != storeFormat {
		return nil, fmt.Errorf("%w: %s is of format %d, this hwnpu reads %d", ErrBundleTarget, filepath.Join(root, StoreFile), m.Format, storeFormat)
	}
	if m.Uses == nil {
		m.Uses = map[string]*record{}
	}
	return m, nil
}

// write writes the manifest as root's StoreFile, through a temporary file.
func (m *manifest) write(root string) error {
	raw, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(root, 0o755); err != nil {
		return err
	}
	tmp := filepath.Join(root, StoreFile+".tmp")
	if err := os.WriteFile(tmp, append(raw, '\n'), 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, filepath.Join(root, StoreFile))
}

// located is a use's record found in a store, its paths absolute.
type located struct {
	*record
	root   string // the store's directory
	bundle bool
}

// find looks for use's record in the bundles, then in Dir, with its files all there. A record
// of another target is skipped, and reported as ErrBundleTarget when nothing else is found.
func (s Store) find(use, target string) (located, error) {
	var mismatch error
	roots := append(append([]string(nil), s.Bundles...), s.Dir)
	for i, root := range roots {
		if root == "" {
			continue
		}
		m, err := readManifest(root)
		if err != nil {
			return located{}, err
		}
		r, ok := m.Uses[use]
		if !ok {
			continue
		}
		if target != "" && m.Target != "" && m.Target != target {
			mismatch = fmt.Errorf("%w: %s in %s is for %s, this machine is %s", ErrBundleTarget, use, root, m.Target, target)
			continue
		}
		r = r.clone()
		r.absolute(root)
		if f := r.missing(); f != "" {
			continue
		}
		return located{record: r, root: root, bundle: i < len(s.Bundles)}, nil
	}
	if mismatch != nil {
		return located{}, mismatch
	}
	return located{}, fmt.Errorf("%w: %s (Plan and Run prepare it)", ErrNotPrepared, use)
}

// put writes use's record into Dir's StoreFile.
func (s Store) put(use, target string, r *record) error {
	m, err := readManifest(s.Dir)
	if err != nil {
		return err
	}
	if m.Target != "" && m.Target != target {
		return fmt.Errorf("%w: %s was prepared for %s, this machine is %s", ErrBundleTarget, s.Dir, m.Target, target)
	}
	m.Target = target
	r = r.clone()
	r.relative(s.Dir)
	m.Uses[use] = r
	return m.write(s.Dir)
}

// hereTarget is the target of this machine's NPU: apple, intel or amd-npu2; "" when there is
// none (a machine without an NPU runs on the CPU).
func hereTarget(ctx context.Context) string {
	if chat.OnlyAMD(ctx, "") {
		return "amd-npu2"
	}
	f, err := llm.ProgramFormat(ctx, "")
	if err != nil {
		return ""
	}
	switch f {
	case llm.BackendCoreML:
		return "apple"
	case llm.BackendIntel:
		return "intel"
	}
	return f
}
