package hfcmd

import (
	"archive/tar"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/klauspost/compress/zstd"

	"github.com/shibukawa/bonsaiai/internal/backend/coreai"
	"github.com/shibukawa/bonsaiai/internal/cache"
	"github.com/shibukawa/bonsaiai/internal/chat"
	shared "github.com/shibukawa/bonsaiai/internal/store"
)

// stage puts f's files in dir (the folder of the repository being made: dir/TARGET/EXPORT/VERSION)
// as its layer holds them (packed or not), checked against its entry.json, and its entry.json.
func stage(ctx context.Context, ls []shared.Source, f found, dir string, log io.Writer) error {
	s := ls[f.layer]
	out := filepath.Join(dir, filepath.FromSlash(f.rel()))
	if err := os.MkdirAll(out, 0o755); err != nil {
		return err
	}
	files := f.got.Files
	if f.got.Pack != nil {
		files = []shared.File{*f.got.Pack}
	}
	var need int64
	for _, x := range files {
		need += x.Size
	}
	if err := cache.Room(need, out, nil); err != nil {
		return err
	}
	for _, x := range files {
		dst := filepath.Join(out, filepath.FromSlash(x.Path))
		if fi, err := os.Stat(dst); err == nil && fi.Size() == x.Size {
			if sum, err := shared.FileSHA256(dst); err == nil && sum == x.SHA256 {
				continue // staged before
			}
		}
		t0 := time.Now()
		fmt.Fprintf(log, "  %s/%s (%.2f GB) from %s\n", f.rel(), x.Path, float64(x.Size)/1e9, s)
		if _, err := s.Store.Get(ctx, f.from+"/"+x.Path, dst, nil); err != nil {
			return err
		}
		sum, err := shared.FileSHA256(dst)
		if err != nil {
			return err
		}
		if sum != x.SHA256 {
			os.Remove(dst)
			return fmt.Errorf("%s/%s: SHA-256 %s, not the %s of its entry.json", f.from, x.Path, sum, x.SHA256)
		}
		if d := time.Since(t0); d > 2*time.Second {
			fmt.Fprintf(log, "    %.1f MB/s\n", float64(x.Size)/1e6/d.Seconds())
		}
	}
	raw := f.raw
	if raw == nil {
		b, err := json.MarshalIndent(f.got, "", "  ")
		if err != nil {
			return err
		}
		raw = append(b, '\n')
	}
	return os.WriteFile(filepath.Join(out, shared.EntryFile), raw, 0o644)
}

// programsDigest is coreai.ProgramsDigest of the Core AI export f staged in dir: read from its
// pack without unpacking it, or from its files.
func programsDigest(f found, dir string) (string, error) {
	out := filepath.Join(dir, filepath.FromSlash(f.rel()))
	if f.got.Pack == nil {
		return coreai.ProgramsDigest(out), nil
	}
	file, err := os.Open(filepath.Join(out, f.got.Pack.Path))
	if err != nil {
		return "", err
	}
	defer file.Close()
	zr, err := zstd.NewReader(file)
	if err != nil {
		return "", err
	}
	defer zr.Close()
	hashes := map[string][]byte{} // by the program's folder
	bytecode := map[string]bool{}
	tr := tar.NewReader(zr)
	for {
		h, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", err
		}
		d, name := path.Split(h.Name)
		d = strings.TrimSuffix(d, "/")
		switch name {
		case "main.hash":
			b, err := io.ReadAll(io.LimitReader(tr, 1<<20))
			if err != nil {
				return "", err
			}
			hashes[d] = b
		case "main.mlirb":
			bytecode[d] = h.Size > 0
		}
	}
	return digestOf(hashes, bytecode), nil
}

// digestOf is coreai.ProgramsDigest of the programs whose folders (in the export) hold
// main.hash (hashes) and main.mlirb (bytecode, true when not empty): the folders NAME.aimodel
// that have both and are in no program, prebuilt program or other such folder.
func digestOf(hashes map[string][]byte, bytecode map[string]bool) string {
	var assets []string
	for d := range hashes {
		if !strings.EqualFold(path.Ext(d), ".aimodel") || !bytecode[d] {
			continue
		}
		assets = append(assets, d)
	}
	var keep []string
	for _, d := range assets {
		inside := false
		for p := path.Dir(d); p != "." && p != "/" && p != ""; p = path.Dir(p) {
			if strings.HasSuffix(p, ".aimodelc") || strings.HasSuffix(p, coreai.PackedSuffix) || slices.Contains(assets, p) {
				inside = true
				break
			}
		}
		if !inside {
			keep = append(keep, d)
		}
	}
	if len(keep) == 0 {
		return ""
	}
	sort.Strings(keep)
	h := sha256.New()
	for _, d := range keep {
		fmt.Fprintf(h, "%s\x00%x\n", d, hashes[d])
	}
	return hex.EncodeToString(h.Sum(nil))
}

// manifest is the repository's config.json: what bonsaiai-dev models hf push put there (bonsaiai reads it
// first; Hugging Face counts its downloads by it).
type manifest struct {
	Bonsaiai struct {
		Format  int             `json:"format"`
		Model   string          `json:"model"`
		Source  manifestSource  `json:"source"`
		Entries []manifestEntry `json:"entries"`
	} `json:"bonsaiai"`
}

type manifestSource struct {
	Repo   string `json:"repo"`
	Commit string `json:"commit,omitempty"`
}

type manifestEntry struct {
	Path    string `json:"path"` // TARGET/EXPORT/VERSION
	Kind    string `json:"kind"`
	Target  string `json:"target"`
	Key     string `json:"key"`
	Version int    `json:"version,omitempty"`
	Bytes   int64  `json:"bytes"` // as stored (packed)
	Made    string `json:"made,omitempty"`
}

// merged is the manifest of what was there (old, at main) and what goes now.
func merged(old *manifest, m chat.Model, commit string, fs []found) manifest {
	var out manifest
	if old != nil {
		out = *old
	}
	out.Bonsaiai.Format, out.Bonsaiai.Model = 1, m.Name
	out.Bonsaiai.Source = manifestSource{Repo: m.Repo, Commit: commit}
	byPath := map[string]manifestEntry{}
	for _, e := range out.Bonsaiai.Entries {
		byPath[e.Path] = e
	}
	for _, f := range fs {
		n := f.got.Bytes()
		if f.got.Pack != nil {
			n = f.got.Pack.Size
		}
		byPath[f.rel()] = manifestEntry{Path: f.rel(), Kind: f.got.Kind, Target: f.got.Target, Key: f.got.Key,
			Version: f.got.Version, Bytes: n, Made: f.got.Made.UTC().Format(time.RFC3339)}
	}
	out.Bonsaiai.Entries = nil
	for _, p := range slices.Sorted(mapKeys(byPath)) {
		out.Bonsaiai.Entries = append(out.Bonsaiai.Entries, byPath[p])
	}
	return out
}

// hfLicenses are the licenses Hugging Face knows by these ids (the rest are other).
var hfLicenses = map[string]bool{"apache-2.0": true, "mit": true, "cc-by-4.0": true, "cc-by-nc-4.0": true, "bsd-3-clause": true, "openrail": true}

// card is the repository's README.md: what it holds and where it comes from, the model's own
// license (docs/prebuilt.md 3, 6).
func card(m chat.Model, commit string, man manifest, gated string) string {
	var b strings.Builder
	b.WriteString("---\n")
	if hfLicenses[m.License.ID] {
		fmt.Fprintf(&b, "license: %s\n", m.License.ID)
	} else {
		fmt.Fprintf(&b, "license: other\nlicense_name: %s\n", m.License.ID)
		if m.License.URL != "" {
			fmt.Fprintf(&b, "license_link: %s\n", m.License.URL)
		}
	}
	fmt.Fprintf(&b, "base_model: %s\nbase_model_relation: quantized\nlibrary_name: bonsaiai\n", m.Repo)
	tags := []string{"bonsaiai", "npu"}
	targets := map[string]bool{}
	for _, e := range man.Bonsaiai.Entries {
		t, _, _ := strings.Cut(e.Path, "/")
		targets[t] = true
	}
	if targets["amd-xdna2"] {
		tags = append(tags, "amd-xdna2", "ryzen-ai")
	}
	if targets["coreai"] {
		tags = append(tags, "coreai", "apple-neural-engine")
	}
	fmt.Fprintf(&b, "tags: [%s]\n", strings.Join(tags, ", "))
	if gated != "" {
		fmt.Fprintf(&b, "extra_gated_prompt: \"%s is gated as %s is: agree to its license (%s) to download it.\"\n", m.Name, m.Repo, m.License.ID)
	}
	b.WriteString("---\n\n")
	fmt.Fprintf(&b, "# %s for NPUs (bonsaiai)\n\n", m.Name)
	src := "https://huggingface.co/" + m.Repo
	fmt.Fprintf(&b, "[%s](%s)", m.Repo, src)
	if commit != "" {
		fmt.Fprintf(&b, " (commit `%s`)", commit)
	}
	b.WriteString(" converted for the NPUs [bonsaiai](https://github.com/shibukawa/bonsaiai) runs it on. For what the model does and how well, read its own card; this repository holds only what bonsaiai needs to run it on an NPU without converting it first.\n\n")
	fmt.Fprintf(&b, "**License**: the model's own, %s", m.License.ID)
	if m.License.URL != "" {
		fmt.Fprintf(&b, " ([%s](%s))", m.License.URL, m.License.URL)
	}
	b.WriteString(". ")
	if m.License.Terms != "" {
		fmt.Fprintf(&b, "Its terms: %s. ", m.License.Terms)
	}
	b.WriteString("What is here is derived from its weights (quantized) and goes under the same license.\n")
	if strings.HasPrefix(m.License.ID, "plamo") {
		b.WriteString("\nBuilt with PLaMo. Commercial use needs PFN's registration first (see the license).\n")
	}
	fmt.Fprintf(&b, "\n## Use\n\n```\n%s -m %s\n```\n\nbonsaiai finds this repository by itself (models.yaml names it) and downloads only what the machine's NPU runs. To fetch a part by hand: `hf download %s --include \"amd-xdna2/*\"`.\n\n", command(m), m.Name, HubRepoName(m))
	b.WriteString("## What is here\n\n| folder | what | size |\n|---|---|---|\n")
	for _, e := range man.Bonsaiai.Entries {
		fmt.Fprintf(&b, "| `%s` | %s | %.2f GB |\n", e.Path, what(e), float64(e.Bytes)/1e9)
	}
	b.WriteString("\n- `amd-xdna2`: AMD's Ryzen AI NPU (XDNA2: Strix Point and later). The same files on Linux (checked) and on Windows through XRT (not checked yet).\n")
	b.WriteString("- `coreai`: every Mac with Core AI (macOS 27): compiled on the Mac the first time. `coreai-<chip>`: compiled ahead of time for that Neural Engine (h13g M1, h14g M2, h15g M3, h16g M4, h17g M5, and their c/s/d variants); a Mac without one uses `coreai`.\n")
	b.WriteString("- `any`: tables read on every NPU (the embeddings), so that the checkpoint is not downloaded.\n")
	b.WriteString("- `config.json`: bonsaiai's list of what is here (not a transformers config).\n")
	return b.String()
}

func what(e manifestEntry) string {
	switch {
	case e.Kind == chat.CoreAIKind:
		return "Core AI export"
	case e.Kind == chat.PrebuiltKind:
		return "prebuilt programs (" + strings.TrimPrefix(e.Target, "coreai-") + ")"
	case e.Kind == chat.TablesKind:
		return "tables"
	case strings.HasPrefix(e.Kind, "iron-"):
		return "export for AMD's NPU (IRON)"
	}
	return e.Kind
}

// command is the command that runs model m.
func command(m chat.Model) string {
	switch m.Kind {
	case chat.KindChat, chat.KindEmbed, chat.KindDecision:
		return "bonsaiai-chat"
	case chat.KindTranscribe:
		return "bonsaiai-audio transcribe"
	case chat.KindSpeech:
		return "bonsaiai-audio tts"
	case chat.KindMusic, chat.KindSFX, chat.KindPitch, chat.KindNotes:
		return "bonsaiai-audio"
	}
	return "bonsaiai-image"
}

// license fetches the model's license file at commit into dir/LICENSE: the file license.url
// names in the model's repository, else LICENSE, LICENSE.md or LICENSE.txt. It reports whether
// there was one.
func license(ctx context.Context, m chat.Model, commit, dir string) bool {
	rev := commit
	if rev == "" {
		rev = "main"
	}
	var names []string
	if rest, ok := strings.CutPrefix(m.License.URL, "https://huggingface.co/"+m.Repo+"/blob/"); ok {
		if _, file, ok := strings.Cut(rest, "/"); ok {
			names = append(names, file)
		}
	}
	names = append(names, "LICENSE", "LICENSE.md", "LICENSE.txt")
	for _, n := range names {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimSuffix(shared.HubEndpoint, "/")+"/"+m.Repo+"/resolve/"+rev+"/"+n, nil)
		if err != nil {
			continue
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			continue
		}
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		resp.Body.Close()
		if resp.StatusCode == http.StatusOK && len(b) > 0 {
			return os.WriteFile(filepath.Join(dir, "LICENSE"), b, 0o644) == nil
		}
	}
	return false
}
