package flux2klein

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
)

// MergeAMD makes out, one IRON export of the sizes of first and of others (each an export of
// tools/amd/iron_flux2klein.py of one size, made with the same options), as
// iron_flux2klein.py --size FIRST OTHERS... lays it out: first's files, and of each other export
// the blocks' graphs of the row counts first has not (on first's banks), its dit_out and VAE in
// SizesDir/WxH, and its encoders of the sizes first has none of. The files are hard links of the
// exports' (copies on another volume), so out takes the room of what the others add (0.4 GB for
// 512 x 512 beside 1024 x 1024, of 11 GB) once first and others are removed. It refuses exports
// whose shared parts differ (the text encoder's, the blocks' banks): another checkpoint or other
// options. out is written as out.merging and renamed when whole; first and others are only read.
func MergeAMD(out, first string, others ...string) error {
	fail := func(err error) error { return fmt.Errorf("flux2klein: merging into %s: %w", filepath.Base(out), err) }
	if !isIronExport(first) {
		return fail(fmt.Errorf("%s is no IRON export", first))
	}
	base, err := ironConfig(first)
	if err != nil {
		return fail(err)
	}
	tmp := out + ".merging"
	if err := os.RemoveAll(tmp); err != nil {
		return fail(err)
	}
	// first's files as they are, but its entry and marks (out is described anew)
	err = filepath.WalkDir(first, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(first, p)
		if d.IsDir() {
			return os.MkdirAll(filepath.Join(tmp, rel), 0o755)
		}
		if rel == "entry.json" || rel == ".files-shared" || !d.Type().IsRegular() {
			return nil
		}
		return linkFile(filepath.Join(tmp, rel), p)
	})
	if err != nil {
		os.RemoveAll(tmp)
		return fail(err)
	}
	sizes := IronSizes(first)
	baseRows := base.Tokens() + base.TextRows
	for _, o := range others {
		if err := mergeSize(tmp, first, o, base, baseRows, &sizes); err != nil {
			os.RemoveAll(tmp)
			return fail(err)
		}
	}
	if err := writeSizes(filepath.Join(tmp, "model.json"), sizes); err != nil {
		os.RemoveAll(tmp)
		return fail(err)
	}
	os.RemoveAll(out)
	if err := os.Rename(tmp, out); err != nil {
		os.RemoveAll(tmp)
		return fail(err)
	}
	return nil
}

// ironConfig is the DiT's shape in the IRON export dir (its first size).
func ironConfig(dir string) (Config, error) {
	b, err := os.ReadFile(filepath.Join(dir, "doubles", "manifest.json"))
	if err != nil {
		return Config{}, err
	}
	var m struct {
		Meta struct {
			Config Config `json:"config"`
		} `json:"meta"`
	}
	if err := json.Unmarshal(b, &m); err != nil {
		return Config{}, err
	}
	return m.Meta.Config, nil
}

// mergeSize puts the parts of the export o (of one size) that out (first's, being made) has not
// in out.
func mergeSize(out, first, o string, base Config, baseRows int, sizes *[][2]int) error {
	if !isIronExport(o) {
		return fmt.Errorf("%s is no IRON export", o)
	}
	if len(IronSizes(o)) != 1 {
		return fmt.Errorf("%s holds several sizes already", o)
	}
	c, err := ironConfig(o)
	if err != nil {
		return err
	}
	size := [2]int{c.Width, c.Height}
	if slices.Contains(*sizes, size) {
		return nil
	}
	a, b := base, c
	a.Width, a.Height, b.Width, b.Height = 0, 0, 0, 0
	if !slices.Equal(a.Axes, b.Axes) || fmt.Sprint(a) != fmt.Sprint(b) {
		return fmt.Errorf("%s's DiT (%+v) is not %s's (%+v)", o, c, first, base)
	}
	// the parts every size shares: the same bytes in both, else another checkpoint or options
	for _, f := range []string{"text/embed.bin", "text/tokenizer.json"} {
		if err := sameFile(filepath.Join(first, f), filepath.Join(o, f)); err != nil {
			return err
		}
	}
	for _, d := range []string{"text", "mods", "ctx"} {
		if err := sameBanks(filepath.Join(first, d), filepath.Join(o, d)); err != nil {
			return err
		}
	}
	oRows := c.Tokens() + c.TextRows
	checked := map[[2]string]bool{} // the banks found the same: each graph of a row count reads them all
	same := func(a, b string) error {
		if checked[[2]string{a, b}] {
			return nil
		}
		if err := sameFile(a, b); err != nil {
			return err
		}
		checked[[2]string{a, b}] = true
		return nil
	}
	entries, err := os.ReadDir(o)
	if err != nil {
		return err
	}
	for _, e := range entries {
		name := e.Name()
		part, rest, _ := strings.Cut(name, "_n")
		if name == "dit_in" || strings.HasPrefix(name, "dit_in_n") {
			part, rest, _ = strings.Cut(strings.TrimPrefix(name, "dit_in"), "_n")
			part = "dit_in"
		}
		if !e.IsDir() || (part != "doubles" && part != "singles" && part != "dit_in") {
			continue
		}
		rows := oRows
		if rest != "" {
			if rows, err = strconv.Atoi(rest); err != nil || name != fmt.Sprintf("%s_n%d", part, rows) {
				continue
			}
		}
		to := part
		if rows != baseRows {
			to = fmt.Sprintf("%s_n%d", part, rows)
		}
		if _, err := os.Stat(filepath.Join(out, to, "manifest.json")); err == nil {
			continue // a graph of that many rows: the same for every size
		}
		if err := moveGraph(out, filepath.Join(o, name), to, same); err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
	}
	sd := sizeDir(c.Width, c.Height)
	for _, part := range []string{"dit_out", "vae"} {
		if err := linkTree(filepath.Join(out, sd, part), filepath.Join(o, part)); err != nil {
			return err
		}
	}
	// the encoders: o's own picture's, and of its sides over 2 and 4
	have := func(w, h int) bool {
		for _, n := range encoderNames(base, w, h) {
			if _, err := os.Stat(filepath.Join(out, n, "manifest.json")); err == nil {
				return true
			}
		}
		return false
	}
	for _, d := range []int{1, 2, 4} {
		src := "vae_enc"
		if d > 1 {
			src = fmt.Sprintf("vae_enc_d%d", d)
		}
		if _, err := os.Stat(filepath.Join(o, src, "manifest.json")); err != nil {
			continue
		}
		w, h := c.Width/d, c.Height/d
		if have(w, h) {
			continue
		}
		if err := linkTree(filepath.Join(out, sizeDir(w, h), "vae_enc"), filepath.Join(o, src)); err != nil {
			return err
		}
	}
	*sizes = append(*sizes, size)
	return nil
}

// encoderNames are the directories an export whose first size is base holds the encoder of
// references of w x h in (Pipeline.encoder).
func encoderNames(base Config, w, h int) []string {
	var names []string
	if w == base.Width && h == base.Height {
		names = append(names, "vae_enc")
	}
	for _, d := range []int{2, 4} {
		if w*d == base.Width && h*d == base.Height {
			names = append(names, fmt.Sprintf("vae_enc_d%d", d))
		}
	}
	return append(names, filepath.Join(sizeDir(w, h), "vae_enc"))
}

// moveGraph puts the graph in src (of another export) in out as to: its own files, and its
// manifest's banks named as out's files with the same bytes (the blocks' banks of out's first
// size; a bank of src's own export that out has not is refused).
func moveGraph(out, src, to string, same func(a, b string) error) error {
	dst := filepath.Join(out, to)
	if err := os.MkdirAll(dst, 0o755); err != nil {
		return err
	}
	b, err := os.ReadFile(filepath.Join(src, "manifest.json"))
	if err != nil {
		return err
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber() // the offsets and sizes as they are
	var m map[string]any
	if err := dec.Decode(&m); err != nil {
		return err
	}
	own := map[string]bool{}
	// rename is a bank's path in the merged graph: its own file where it is src's, else out's file
	// of the same name in the same place of out's first size (the same bytes)
	rename := func(rel string) (string, error) {
		file := filepath.Clean(filepath.Join(src, rel))
		if filepath.Dir(file) == filepath.Clean(src) && !isBlockBank(filepath.Base(src), rel) {
			own[filepath.Base(file)] = true
			return rel, nil
		}
		// a bank of the blocks (src's own, or ../doubles/... of its export): out's first size's
		part := filepath.Base(src)
		if filepath.Dir(file) != filepath.Clean(src) {
			part = filepath.Base(filepath.Dir(file))
		}
		part, _, _ = strings.Cut(part, "_n")
		mine := filepath.Join(out, part, filepath.Base(file))
		if err := same(mine, file); err != nil {
			return "", err
		}
		return filepath.ToSlash(filepath.Join("..", part, filepath.Base(file))), nil
	}
	if banks, ok := m["banks"].(map[string]any); ok {
		for _, v := range banks {
			bk, ok := v.(map[string]any)
			if !ok {
				continue
			}
			if img, ok := bk["image"].(string); ok && img != "" {
				if bk["image"], err = rename(img); err != nil {
					return err
				}
			}
		}
	}
	if vs, ok := m["variants"].([]any); ok {
		for _, v := range vs {
			vm, ok := v.(map[string]any)
			if !ok {
				continue
			}
			for k, f := range vm {
				if s, ok := f.(string); ok {
					if vm[k], err = rename(s); err != nil {
						return err
					}
				}
			}
		}
	}
	m["name"] = to
	entries, err := os.ReadDir(src)
	if err != nil {
		return err
	}
	for _, e := range entries {
		n := e.Name()
		if !e.Type().IsRegular() || n == "manifest.json" || (strings.HasSuffix(n, ".bin") && !own[n]) {
			continue
		}
		if err := linkFile(filepath.Join(dst, n), filepath.Join(src, n)); err != nil {
			return err
		}
	}
	nb, err := json.MarshalIndent(m, "", " ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dst, "manifest.json"), append(nb, '\n'), 0o644)
}

// isBlockBank reports whether rel, a bank of the graph dir named part, is a bank of the blocks
// (doubles, singles): those every size's graphs share.
func isBlockBank(part, rel string) bool {
	p, _, _ := strings.Cut(part, "_n")
	return (p == "doubles" || p == "singles") && strings.HasSuffix(rel, ".bin")
}

// sameBanks checks that the graphs in a and b read the same bytes from their banks.
func sameBanks(a, b string) error {
	entries, err := os.ReadDir(a)
	if err != nil {
		return err
	}
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".bin") {
			if err := sameFile(filepath.Join(a, e.Name()), filepath.Join(b, e.Name())); err != nil {
				return err
			}
		}
	}
	return nil
}

// sameFile checks that a and b hold the same bytes.
func sameFile(a, b string) error {
	fa, err := os.Open(a)
	if err != nil {
		return err
	}
	defer fa.Close()
	fb, err := os.Open(b)
	if err != nil {
		return err
	}
	defer fb.Close()
	sa, _ := fa.Stat()
	sb, _ := fb.Stat()
	if os.SameFile(sa, sb) {
		return nil
	}
	if sa.Size() != sb.Size() {
		return fmt.Errorf("%s and %s differ (%d and %d bytes): exports of another checkpoint or other options", a, b, sa.Size(), sb.Size())
	}
	ba, bb := make([]byte, 4<<20), make([]byte, 4<<20)
	for {
		na, ea := io.ReadFull(fa, ba)
		nb, eb := io.ReadFull(fb, bb)
		if na != nb || !bytes.Equal(ba[:na], bb[:nb]) {
			return fmt.Errorf("%s and %s differ: exports of another checkpoint or other options", a, b)
		}
		if ea == io.EOF || ea == io.ErrUnexpectedEOF {
			return nil
		}
		if ea != nil {
			return ea
		}
		if eb != nil {
			return eb
		}
	}
}

// linkTree puts the files of src in dst (made), as linkFile.
func linkTree(dst, src string) error {
	return filepath.WalkDir(src, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(src, p)
		if d.IsDir() {
			return os.MkdirAll(filepath.Join(dst, rel), 0o755)
		}
		if !d.Type().IsRegular() {
			return nil
		}
		return linkFile(filepath.Join(dst, rel), p)
	})
}

// linkFile makes dst a hard link of src, or a copy where src is on another volume.
func linkFile(dst, src string) error {
	if err := os.Link(src, dst); err == nil {
		return nil
	} else if errors.Is(err, fs.ErrExist) {
		return err
	}
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	fi, err := in.Stat()
	if err != nil {
		return err
	}
	o, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_EXCL, fi.Mode().Perm())
	if err != nil {
		return err
	}
	if _, err := io.Copy(o, in); err != nil {
		o.Close()
		return err
	}
	return o.Close()
}

// writeSizes sets model.json's sizes (iron_flux2klein.write_model) to sizes.
func writeSizes(path string, sizes [][2]int) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	var m map[string]any
	if err := dec.Decode(&m); err != nil {
		return err
	}
	m["sizes"] = sizes
	nb, err := json.MarshalIndent(m, "", " ")
	if err != nil {
		return err
	}
	os.Remove(path) // a hard link of first's: written anew, not through
	return os.WriteFile(path, append(nb, '\n'), 0o644)
}
